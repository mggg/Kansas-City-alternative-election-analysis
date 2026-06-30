"""
Summarize election simulation outputs and generate visualization figures.

Aggregates district-level election results produced by the
pipeline into a single summary dataset and generates histogram
visualizations of representation outcomes. Joins election results
with district-level population data from the corresponding settings
files, computes focal-group representation statistics, and writes a
summary CSV along with figures showing the distribution of seats won
across voter models and election methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import geopandas as gpd

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pipeline.utils.helpers import (
    parse_district_configs,
    parse_plan_district_rep_from_path,
    count_focal_winners,
    load_json,
    find_settings_file,
    get_non_focal_group,
)


# The three voter models the pipeline simulates. These match the subdirectory
# names created upstream by generate_profiles / simulate_elections.
MODES = ["slate_pl", "slate_bt", "cambridge"]

# Map the raw method keys emitted by simulate_elections to display names.
METHOD_NAME_MAP = {
    "stv": "STV",
    "plurality": "Plurality",
    "irv": "IRV",
}

# Fixed colors / labels so every figure reads the same way.
MODE_COLORS = {
    "cambridge": "#E32636",
    "slate_bt": "#FFBF00",
    "slate_pl": "#8DB600",
}

# Pseudo-mode that pools occurrences across every voter model into one row.
COMBINED_MODE = "combined"

LEGEND_MAPPING = {
    "slate_bt": "Deliberative",
    "slate_pl": "Impulsive",
    "cambridge": "Cambridge",
    COMBINED_MODE: "Combined",
}

DESIRED_ORDER = ["slate_pl", "slate_bt", "cambridge"]


# --- Representation baselines --------------------------------------------------


def _focal_population_share(config) -> float:
    """
    Statewide focal-group population proportion, straight from the geodata.

    This is the *overall* focal share before any districting, used as the
    "proportional representation" population baseline on each figure.
    """
    geodata_path = Path(config["geodata_path"])
    gdf = gpd.read_file(geodata_path)
    # vap  = total (voting-age) population across all precincts
    # ivap = population of the group of interest (the focal group) across all precincts
    vap = sum(gdf[config["population_column"]])
    ivap = sum(gdf[config["pop_of_interest_column"]])
    return ivap / vap  # raw focal-group population proportion


def _turnout_adjusted_share(config, iprop: float) -> float:
    """
    Reweight the raw population share by per-bloc turnout to get the effective
    share of *voters* that belong to the focal group.
    """
    focal_group = str(config["focal_group"])
    turnout = config["turnout"]
    if len(turnout) != 2:
        raise ValueError("Turnout does not have exactly two keys")
    non_focal_group = get_non_focal_group(config)
    # Bayes-style reweighting: focal voters / (focal voters + non-focal voters).
    return (
        iprop * turnout[focal_group]
        / (iprop * turnout[focal_group] + (1 - iprop) * turnout[non_focal_group])
    )


def _combined_support(config, iprop_turnout: float) -> float:
    """
    Share of the *vote* that flows to focal candidates.

    Blends two sources: focal voters who back focal candidates (cohesion), plus
    non-focal voters who cross over to focal candidates. cohesion_parameters[g][h]
    is the share of bloc g's votes that go to bloc h's slate.
    """
    focal_group = str(config["focal_group"])
    non_focal_group = get_non_focal_group(config)
    cohesion_parameters = config["cohesion_parameters"]
    focal_group_cohesion = cohesion_parameters[focal_group]
    non_focal_group_cohesion = cohesion_parameters[non_focal_group]
    return (
        iprop_turnout * focal_group_cohesion[focal_group]
        + (1 - iprop_turnout) * non_focal_group_cohesion[focal_group]
    )


def _compute_representation_baselines(config) -> Tuple[float, float, float]:
    """
    Compute the three representation baselines used throughout this step.

    Returns:
        (iprop, iprop_turnout, i_cs_turnout):
            raw focal population share, turnout-adjusted voter share, and
            combined support (vote share for focal candidates).
    """
    iprop = _focal_population_share(config)
    iprop_turnout = _turnout_adjusted_share(config, iprop)
    i_cs_turnout = _combined_support(config, iprop_turnout)
    return iprop, iprop_turnout, i_cs_turnout


# --- Filesystem layout ---------------------------------------------------------


def _prepare_directories(run_name: str) -> Tuple[Path, Path, Path]:
    """
    Resolve the input results directory and create the output directories.

    Returns:
        (results_dir, summary_dir, figs_dir).

    Raises:
        FileNotFoundError: If the election results directory does not exist.
    """
    # simulate_elections writes one JSON per (mode, district config) under here.
    results_dir = Path("outputs") / f"{run_name}" / "election_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Could not find election results directory: {results_dir}")

    # Layout mirrors the rest of the pipeline: outputs/<run_name>/summaries/...
    # (run.py's has_valid_summaries() looks for exactly this CSV and figures dir.)
    summary_dir = Path("outputs") / f"{run_name}" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = summary_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, summary_dir, figs_dir


# --- Tidy-table construction ---------------------------------------------------


def _district_population(settings_dir: Path, config, plan, district) -> Tuple[Any, Any]:
    """
    Join back to the settings file for a district to recover the population
    totals that profile was built from.

    Returns:
        (total_vap, total_ivap), either of which may be None if no settings
        file is found.
    """
    settings_path = find_settings_file(
        settings_dir, config["run_name"], plan=plan, district=district
    )
    settings_data = load_json(settings_path) if settings_path else {}
    total_vap = settings_data.get(config["population_column"], None)
    total_ivap = settings_data.get(config["pop_of_interest_column"], None)
    return total_vap, total_ivap


def _rows_from_results_file(
    rf: Path,
    dc,
    mode: str,
    settings_dir: Path,
    config,
    i_cs_turnout: float,
) -> List[Dict[str, Any]]:
    """
    Build the summary rows contributed by a single election-results JSON file.

    Returns an empty list for files that do not match the district config/mode
    currently being iterated (guards against stale or mixed-in files).

    Raises:
        ValueError: If profile_files is missing or its length does not match
            election_results.
    """
    run_name = str(config["run_name"])
    focal_group = str(config["focal_group"])
    # slate_to_candidates maps a bloc label (e.g. "A") to the candidate ids it ran.
    # It is optional here: count_focal_winners can fall back to a prefix match.
    slate_to_candidates = config.get("slate_to_candidates", {}) or {}

    data = load_json(rf)

    # The results file self-describes its district count, seats, and mode.
    # We re-read them and skip files that don't match the config we are
    # currently iterating on (guards against stale or mixed-in files).
    district_num = int(data.get("district_num", dc.num_districts))
    winners_per_district = int(data.get("winners_per_district", dc.winners))
    voter_mode = str(data.get("voter_mode", mode))
    if (
        district_num != dc.num_districts
        or winners_per_district != dc.winners
        or voter_mode != mode
    ):
        return []

    # election_results[i] holds the winners for the i-th simulated profile;
    # profile_files[i] is the path to that profile. They must line up 1:1.
    election_results: List[Dict[str, List[str]]] = data.get("election_results", [])
    profile_files: Optional[List[str]] = data.get("profile_files")

    if profile_files is None:
        raise ValueError(f"Missing profile_files in results file: {rf}")

    if len(election_results) != len(profile_files):
        raise ValueError(
            f"Length mismatch in {rf}: "
            f"{len(election_results)=} vs {len(profile_files)=}"
        )

    rows: List[Dict[str, Any]] = []
    # --- One row per simulated profile (and per election method) ----
    for idx, result in enumerate(election_results):
        # Recover (plan, district, replicate) by parsing the profile path,
        # e.g. ..._district_plan_003_district_07_v1.csv -> (3, 7, 1).
        plan, district, rep = parse_plan_district_rep_from_path(profile_files[idx])

        total_vap, total_ivap = _district_population(settings_dir, config, plan, district)

        # A single profile may be scored under several methods (e.g. a
        # single-winner district under Plurality and IRV), so we emit one
        # row per method, each with its own focal-seat count.
        for method_key, winners in result.items():
            focal_seats = count_focal_winners(
                winners,
                focal_group,
                slate_to_candidates,
            )
            rows.append({
                "run_name": run_name,
                "plan": plan,
                "num_districts": district_num,
                "seats_per_district": winners_per_district,
                "election_method": METHOD_NAME_MAP.get(method_key, method_key.upper()),
                "mode": mode,
                "district_id": district,
                "rep": rep,
                "simulation_index": idx,
                "focal_group": focal_group,
                "focal_seats": focal_seats,
                config["population_column"]: total_vap,
                config["pop_of_interest_column"]: total_ivap,
                "combined_support": i_cs_turnout,
            })

    return rows


def build_summary_dataframe(config, results_dir: Path, i_cs_turnout: float) -> pd.DataFrame:
    """
    Walk every district config x voter model x results file and build the tidy,
    district-level summary DataFrame (sorted, one row per
    (replicate, plan, district, election_method) tuple).
    """
    run_name = str(config["run_name"])
    # district_configs may use either the new {"num_districts", "winners"} schema or
    # the legacy {<n>: <winners>} schema; the helper normalizes both into objects.
    district_configs = parse_district_configs(config["district_configs"])

    # We accumulate one dict per row and build the DataFrame once at the end; this
    # is much faster than growing a DataFrame incrementally.
    rows: List[Dict[str, Any]] = []

    for dc in district_configs:
        # Settings files are grouped by district count (one folder per num_districts),
        # matching settings_generator's outputs/<run>/settings/<n>/ layout.
        settings_dir = Path("outputs") / f"{run_name}" / "settings" / str(dc.num_districts)

        for mode in MODES:
            mode_dir = results_dir / mode
            if not mode_dir.exists():
                continue

            for rf in sorted(mode_dir.glob("*.json")):
                rows.extend(
                    _rows_from_results_file(rf, dc, mode, settings_dir, config, i_cs_turnout)
                )

    df = pd.DataFrame(rows)
    df = df.sort_values(["mode", "rep", "num_districts", "plan", "district_id"])
    return df


def aggregate_to_plan_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the district-level table up to the plan level.

    A "plan" is one sampled districting map; representation is naturally a
    whole-map quantity, so we sum focal seats across that plan's districts. Each
    (plan, mode, method, replicate) becomes one data point in the histograms.
    """
    return (
        df.groupby(
            ["plan", "num_districts", "seats_per_district", "mode", "election_method", "rep"],
            as_index=False,
        )
        .agg({"focal_seats": "sum"})
    )


# --- Plotting ------------------------------------------------------------------


def _draw_mode_histograms(ax, group_distn: pd.DataFrame) -> float:
    """
    Draw a grouped (dodged) bar histogram with one series per voter model.

    For each integer focal-seat count, each mode gets its own bar placed
    side-by-side, so the series are read by comparison rather than overlapping
    translucently. Modes are ordered by DESIRED_ORDER (with any unexpected modes
    appended) so colors line up left-to-right with the legend.

    Returns:
        The tallest bar height across all modes, so the caller can scale the
        y-axis (and place text labels) consistently.
    """
    present_modes = set(group_distn["mode"].unique())
    # Canonical order first, then any modes not anticipated by DESIRED_ORDER.
    modes_in_order = [m for m in DESIRED_ORDER if m in present_modes]
    modes_in_order += [m for m in present_modes if m not in DESIRED_ORDER]

    n_modes = len(modes_in_order)
    if n_modes == 0:
        return 0

    # Total group width of 0.8 keeps a gap between adjacent seat counts; each
    # mode's bar is an equal slice of that width.
    bar_width = 0.8 / n_modes
    max_bin_height = 0

    for i, mode in enumerate(modes_in_order):
        seats = group_distn.loc[group_distn["mode"] == mode, "focal_seats"]
        if seats.empty:
            continue

        # One bar per possible focal-seat count in this group.
        counts = seats.value_counts().sort_index()

        # Center the cluster of bars on each integer seat value.
        offset = (i - (n_modes - 1) / 2) * bar_width

        ax.bar(
            counts.index + offset,
            counts.values,
            width=bar_width,
            edgecolor="gray",
            linewidth=0.5,
            color=MODE_COLORS.get(mode, "xkcd:light gray"),
            alpha=0.9,
            label=mode,
        )

        if len(counts) > 0:
            max_bin_height = max(max_bin_height, counts.values.max())

    return max_bin_height


def _style_axes(ax, config, focal_group: str, num_dist, seats_per_district, elm, ylim: float) -> None:
    """Apply spines, limits, ticks, labels, and title for one histogram figure."""
    # Thin, uniform spines.
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # x-axis spans 0..total_seats; give the y-axis 20% headroom for the labels.
    total_seats = config["total_seats"]

    ax.set_xlim(-1, total_seats + 1)
    ax.set_ylim(0, ylim)
    ax.set_xticks(range(0, total_seats + 1, 1))
    # Only label multiples of 5 to keep the axis uncluttered.
    ax.set_xticklabels([str(x) if x % 5 == 0 else "" for x in range(0, total_seats + 1)])
    ax.set_xlabel("Seats")
    ax.set_title(
        f"Representation for {focal_group}-preferred candidates, "
        f"{num_dist} x {seats_per_district} {elm}"
    )
    ax.tick_params(axis="both", which="major", labelsize=8)


def _build_mode_legend(ax) -> None:
    """Draw a legend of modes only, renamed via LEGEND_MAPPING and in DESIRED_ORDER."""
    handles, labels = ax.get_legend_handles_labels()
    handle_map = {label: handle for handle, label in zip(handles, labels) if label in LEGEND_MAPPING}

    ordered_handles, ordered_labels = [], []
    for mode_key in DESIRED_ORDER:
        if mode_key in handle_map:
            ordered_handles.append(handle_map[mode_key])
            ordered_labels.append(LEGEND_MAPPING[mode_key])

    ax.legend(ordered_handles, ordered_labels, title="Mode", fontsize=8)


def _draw_reference_lines(ax, config, iprop: float, i_cs_turnout: float, ylim: float) -> None:
    """
    Draw the two "proportional representation" reference lines and their labels.

    i_cs_share : seats implied by combined *support* (votes for focal cands).
    i_share    : seats implied by raw focal-group *population* share.
    Comparing where the histogram mass falls against these lines is the whole
    point of the figure.
    """
    total_seats = config["total_seats"]
    color_cs = "xkcd:brownish grey"
    color_iprop = "xkcd:purplish brown"

    i_cs_share = i_cs_turnout * total_seats
    i_share = iprop * total_seats

    # Nudge the two text labels apart so they don't overlap when the lines are
    # close: the leftmost line gets a right-aligned label and vice versa.
    if i_cs_share < i_share:
        i_cs_alignment = -0.3
        i_share_alignment = 0.3
        i_cs_ha = "right"
        i_share_ha = "left"
    else:
        i_cs_alignment = 0.3
        i_share_alignment = -0.3
        i_cs_ha = "left"
        i_share_ha = "right"

    ax.axvline(i_cs_share, color=color_cs, linewidth=1)
    ax.text(
        i_cs_share + i_cs_alignment,
        ylim * 0.90,
        f"Combined support\n{i_cs_turnout * 100:.2f}%\n({i_cs_share:.2f} seats)",
        va="center",
        ha=i_cs_ha,
        fontsize=8,
        color=color_cs,
    )

    ax.axvline(i_share, color=color_iprop, linestyle=":", linewidth=1)
    ax.text(
        i_share + i_share_alignment,
        ylim * 0.90,
        f"Focal group VAP\n{iprop * 100:.2f}%\n({i_share:.2f} seats)",
        va="center",
        ha=i_share_ha,
        fontsize=8,
        color=color_iprop,
    )


def _plot_one_histogram(
    group_distn: pd.DataFrame,
    num_dist,
    seats_per_district,
    elm,
    config,
    focal_group: str,
    iprop: float,
    i_cs_turnout: float,
    figs_dir: Path,
    run_name: str,
) -> None:
    """Create and save a single by-mode representation histogram figure."""
    fig, ax = plt.subplots(figsize=(6, 4))

    max_bin_height = _draw_mode_histograms(ax, group_distn)
    ylim = max_bin_height * 1.2 if max_bin_height > 0 else 1

    _style_axes(ax, config, focal_group, num_dist, seats_per_district, elm, ylim)
    _build_mode_legend(ax)
    _draw_reference_lines(ax, config, iprop, i_cs_turnout, ylim)

    fig_path = figs_dir / f"{run_name}_{num_dist}x{seats_per_district}_{elm}_bymode.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_representation_histograms(
    df_plan: pd.DataFrame,
    config,
    focal_group: str,
    iprop: float,
    i_cs_turnout: float,
    figs_dir: Path,
    run_name: str,
) -> None:
    """Produce one histogram per (district count, seats, election method)."""
    for (num_dist, seats_per_district, elm), group_distn in df_plan.groupby(
        ["num_districts", "seats_per_district", "election_method"]
    ):
        _plot_one_histogram(
            group_distn,
            num_dist,
            seats_per_district,
            elm,
            config,
            focal_group,
            iprop,
            i_cs_turnout,
            figs_dir,
            run_name,
        )


# --- Bubble plot ---------------------------------------------------------------

# Marker areas (points^2): the most frequent cell uses BUBBLE_MAX_AREA, and a
# floor keeps rare cells visible.
BUBBLE_MAX_AREA = 300
BUBBLE_MIN_AREA = 20

# Color of the focal-group proportional-representation reference line (matches
# the "Focal group VAP" line on the histograms).
PROP_LINE_COLOR = "orangered"

# Single fill color for all bubbles; the voter mode is conveyed by the y-axis row.
BUBBLE_COLOR = "#4C72B0"


def _occurrence_counts(df_plan: pd.DataFrame) -> pd.DataFrame:
    """
    Count plan-level occurrences per (election_method, mode, focal_seats), plus
    a pooled ``COMBINED_MODE`` row that averages those counts across every voter
    model so the figure can show the combined distribution on the same scale as
    the individual models.
    """
    per_mode = (
        df_plan.groupby(["election_method", "mode", "focal_seats"])
        .size()
        .reset_index(name="count")
    )
    # Average across models: sum the counts then divide by the number of voter
    # models for that method, so seats where only some models landed aren't
    # over-counted (a missing (mode, seats) cell counts as zero, not absent).
    n_models = per_mode.groupby("election_method")["mode"].transform("nunique")
    combined = (
        per_mode.assign(count=per_mode["count"] / n_models)
        .groupby(["election_method", "focal_seats"], as_index=False)["count"]
        .sum()
    )
    combined["mode"] = COMBINED_MODE
    return pd.concat([per_mode, combined], ignore_index=True)


def _draw_method_bubbles(
    ax,
    method_counts: pd.DataFrame,
    modes_in_order: List[str],
    size_scale: float,
    iprop: float,
    config,
) -> None:
    """
    Draw the bubble grid (mode x seats, area sized by occurrence count) for one
    election method, overlay the focal-group proportional-representation line,
    and style the axes.
    """
    y_index = {mode: i for i, mode in enumerate(modes_in_order)}

    for mode in modes_in_order:
        sub = method_counts[method_counts["mode"] == mode]
        if sub.empty:
            continue
        ax.scatter(
            sub["focal_seats"],
            [y_index[mode]] * len(sub),
            s=BUBBLE_MIN_AREA + sub["count"] * size_scale,
            color=BUBBLE_COLOR,
            alpha=0.7,
            edgecolor="gray",
            linewidth=0.5,
        )

    total_seats = config["total_seats"]

    # Seats the focal group would win under strict population-proportional
    # representation: their population share times the total number of seats.
    i_share = iprop * total_seats
    ax.axvline(i_share, color=PROP_LINE_COLOR, linestyle=":", linewidth=1.2)

    ax.set_xlim(0, total_seats + 1)
    ax.set_xticks(range(0, total_seats + 2, 1))
    # Only label even seat counts to keep the axis uncluttered.
    ax.set_xticklabels([str(x) if x % 2 == 0 else "" for x in range(0, total_seats + 2)])
    ax.set_xlabel("City Council Seats")

    ax.set_ylim(-0.5, len(modes_in_order) - 0.5)
    ax.set_yticks(range(len(modes_in_order)))
    ax.set_yticklabels([LEGEND_MAPPING.get(m, m) for m in modes_in_order])

    ax.tick_params(axis="both", which="major", labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def plot_representation_bubbles(
    df_plan: pd.DataFrame,
    config,
    focal_group: str,
    iprop: float,
    figs_dir: Path,
    run_name: str,
) -> None:
    """
    One bubble figure per districting configuration (district count x
    magnitude), each with one subplot per election method. Splitting by
    configuration keeps the filenames aligned with the histograms and prevents
    different configurations from overwriting a single shared image.
    """
    for (num_dist, seats_per_district), config_plans in df_plan.groupby(
        ["num_districts", "seats_per_district"]
    ):
        _plot_bubbles_for_config(
            config_plans,
            config,
            iprop,
            figs_dir,
            run_name,
            num_dist,
            seats_per_district,
        )


def _plot_bubbles_for_config(
    df_plan: pd.DataFrame,
    config,
    iprop: float,
    figs_dir: Path,
    run_name: str,
    num_dist,
    seats_per_district,
) -> None:
    """
    Single figure with one bubble subplot per election method.

    Each subplot has focal seats on the x-axis and voter modes on the y-axis;
    bubble area encodes how many plans produced that focal-seat count under that
    mode. A dotted line marks the focal group's proportional-representation seat
    share. Subplots share the y-axis so modes line up across methods.
    """
    counts = _occurrence_counts(df_plan)
    if counts.empty:
        return

    methods = sorted(counts["election_method"].unique())

    present_modes = set(counts["mode"].unique())
    # Canonical order first, then any modes not anticipated by DESIRED_ORDER,
    # with the pooled "Combined" row pinned to the top.
    modes_in_order = [m for m in DESIRED_ORDER if m in present_modes]
    modes_in_order += [
        m for m in present_modes if m not in DESIRED_ORDER and m != COMBINED_MODE
    ]
    if COMBINED_MODE in present_modes:
        modes_in_order.append(COMBINED_MODE)

    # Scale bubble area from the per-model counts only; the pooled "Combined"
    # row sums those, so including it would shrink every individual bubble.
    per_model_counts = counts.loc[counts["mode"] != COMBINED_MODE, "count"]
    max_count = int(per_model_counts.max()) if not per_model_counts.empty else 0
    size_scale = (BUBBLE_MAX_AREA - BUBBLE_MIN_AREA) / max_count if max_count > 0 else 0

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(4 * len(methods), 3),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, method in zip(axes, methods):
        _draw_method_bubbles(
            ax,
            counts[counts["election_method"] == method],
            modes_in_order,
            size_scale,
            iprop,
            config,
        )
        # Title includes the districting configuration (district count x
        # magnitude), e.g. "4 X 3 STV".
        ax.set_title(f"{num_dist} X {seats_per_district} {method}", fontsize=10)

    # One shared legend for the proportional-representation line (the same seat
    # share applies to every subplot since it depends only on population).
    prop_handle = Line2D(
        [0], [0],
        color=PROP_LINE_COLOR,
        linestyle=":",
        linewidth=1.2,
        label=f"Proportional representation ({iprop * 100:.1f}%)",
    )
    # Lay out subplots first, reserving the top strip for the legend so it
    # sits above the titles instead of overlapping them.
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(
        handles=[prop_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        fontsize=7,
        frameon=True,
    )
    fig_path = (
        figs_dir
        / f"{run_name}_{num_dist}x{seats_per_district}_bubbles_by_method.png"
    )
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _district_occurrence_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count district-level occurrences per
    (election_method, district_id, mode, focal_seats).

    Unlike _occurrence_counts this keeps results at the individual-district
    level (no plan-level aggregation) so each district can be its own row.
    """
    return (
        df.groupby(["election_method", "district_id", "mode", "focal_seats"])
        .size()
        .reset_index(name="count")
    )


def _district_row_layout(
    districts: List[Any], modes_in_order: List[str]
) -> Tuple[Dict[Tuple[Any, str], int], Dict[Any, float], int]:
    """
    Assign each (district, voter model) its own y-row so bubbles never overlap.

    Models are stacked within a district, districts are separated by a one-row
    gap, and rows are numbered top-down (District 1 on top).

    Returns:
        (y_pos, district_centers, n_rows):
            y_pos maps (district, mode) -> y coordinate; district_centers maps a
            district -> the y at the middle of its model rows (for the tick
            label); n_rows is the total span including gaps.
    """
    # Top-to-bottom ordering, with a None placeholder for the gap between groups.
    ordered: List[Optional[Tuple[Any, str]]] = []
    for district in districts:
        for mode in modes_in_order:
            ordered.append((district, mode))
        ordered.append(None)  # gap row after each district group
    if ordered and ordered[-1] is None:
        ordered.pop()

    n_rows = len(ordered)
    # Index 0 is the top, so the largest y goes to the first entry.
    y_pos = {item: (n_rows - 1) - i for i, item in enumerate(ordered) if item is not None}
    district_centers = {
        district: sum(y_pos[(district, m)] for m in modes_in_order) / len(modes_in_order)
        for district in districts
    }
    return y_pos, district_centers, n_rows


def _draw_district_model_bubbles(
    ax,
    method_counts: pd.DataFrame,
    districts: List[Any],
    modes_in_order: List[str],
    y_pos: Dict[Tuple[Any, str], int],
    district_centers: Dict[Any, float],
    n_rows: int,
    size_scale: float,
    district_props: Dict[Any, float],
    seats_per_district: int,
) -> None:
    """
    Draw the district-by-seats bubble grid for one election method. Every
    (district, voter model) pair is its own row, so the colored bubbles are
    fully separated with margin to spare. Alternate districts get a shaded band
    (striped-table style), and each district's proportional-representation line
    is placed by its own focal-group population share.
    """
    y_bottom, y_top = -0.7, n_rows - 0.3

    # Vertical extent of each district's block of model rows.
    ymin_d = {d: min(y_pos[(d, m)] for m in modes_in_order) for d in districts}
    ymax_d = {d: max(y_pos[(d, m)] for m in modes_in_order) for d in districts}

    # Zebra shading: districts run top-to-bottom; split adjacent groups at the
    # midpoint of the gap between them so the stripes are contiguous.
    for i, d in enumerate(districts):
        upper = y_top if i == 0 else (ymin_d[districts[i - 1]] + ymax_d[d]) / 2
        lower = y_bottom if i == len(districts) - 1 else (ymin_d[d] + ymax_d[districts[i + 1]]) / 2
        if i % 2 == 1:
            ax.axhspan(lower, upper, color="0.93", zorder=-1)

    # Faint dotted guide lines rising from each integer seat tick.
    for x in range(0, seats_per_district + 2):
        ax.axvline(x, color="0.85", linestyle=":", linewidth=0.5, zorder=0)

    for mode in modes_in_order:
        sub = method_counts[method_counts["mode"] == mode]
        if sub.empty:
            continue
        ax.scatter(
            sub["focal_seats"],
            [y_pos[(d, mode)] for d in sub["district_id"]],
            s=BUBBLE_MIN_AREA + sub["count"] * size_scale,
            color=MODE_COLORS.get(mode, BUBBLE_COLOR),
            alpha=0.6,
            edgecolor="gray",
            linewidth=0.5,
            zorder=2,
        )

    # Per-district proportional-representation line: the district's own focal
    # population share times its seat count, drawn across just that district's
    # rows.
    for d in districts:
        prop = district_props.get(d)
        if prop is None:
            continue
        x = prop * seats_per_district
        ax.plot(
            [x, x],
            [ymin_d[d] - 0.5, ymax_d[d] + 0.5],
            color=PROP_LINE_COLOR,
            linestyle=":",
            linewidth=1.2,
            zorder=1,
        )

    ax.set_xlim(0, seats_per_district + 1)
    ax.set_xticks(range(0, seats_per_district + 2))
    ax.set_xlabel("Focal seats in district")

    # One tick per district, centered on its block of model rows; 1-indexed.
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.set_yticks([district_centers[d] for d in districts])
    ax.set_yticklabels([f"District {d + 1}" for d in districts])

    ax.tick_params(axis="both", which="major", labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def _district_focal_proportions(df: pd.DataFrame, config) -> Dict[Any, float]:
    """
    Mean focal-group population share for each district id, pooled across the
    sampled plans (focal VAP / total VAP). Rows with missing population are
    ignored; districts with no usable population data are omitted.
    """
    pop_col = config["population_column"]
    ipop_col = config["pop_of_interest_column"]
    usable = df.dropna(subset=[pop_col, ipop_col])
    if usable.empty:
        return {}
    totals = usable.groupby("district_id")[[ipop_col, pop_col]].sum()
    totals = totals[totals[pop_col] > 0]
    return (totals[ipop_col] / totals[pop_col]).to_dict()


def _plot_district_bubbles_for_config(
    df: pd.DataFrame,
    config,
    iprop: float,
    figs_dir: Path,
    run_name: str,
    num_dist,
    seats_per_district,
) -> None:
    """
    Single figure for one districting configuration with one subplot per
    election method. Each district is a y-axis row; bubble area encodes how
    often that district elected a given number of focal seats, and each voter
    model is drawn in its own translucent color so overlaps are easy to spot.
    """
    counts = _district_occurrence_counts(df)
    if counts.empty:
        return

    methods = sorted(counts["election_method"].unique())

    present_modes = set(counts["mode"].unique())
    modes_in_order = [m for m in DESIRED_ORDER if m in present_modes]
    modes_in_order += [m for m in present_modes if m not in DESIRED_ORDER]

    districts = sorted(counts["district_id"].unique())
    district_props = _district_focal_proportions(df, config)

    max_count = int(counts["count"].max())
    size_scale = (BUBBLE_MAX_AREA - BUBBLE_MIN_AREA) / max_count if max_count > 0 else 0

    # One row per (district, model). Sizing the figure by row count keeps the
    # row spacing (~ROW_INCHES) comfortably larger than the biggest bubble
    # diameter, so nothing overlaps.
    y_pos, district_centers, n_rows = _district_row_layout(districts, modes_in_order)
    ROW_INCHES = 0.4
    fig_height = ROW_INCHES * n_rows + 1.8

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(4 * len(methods), fig_height),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, method in zip(axes, methods):
        _draw_district_model_bubbles(
            ax,
            counts[counts["election_method"] == method],
            districts,
            modes_in_order,
            y_pos,
            district_centers,
            n_rows,
            size_scale,
            district_props,
            int(seats_per_district),
        )
        ax.set_title(f"{num_dist} X {seats_per_district} {method}", fontsize=10)

    # Legend: one swatch per voter model plus the proportional-representation line.
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=MODE_COLORS.get(m, BUBBLE_COLOR),
            markeredgecolor="gray",
            alpha=0.5,
            label=LEGEND_MAPPING.get(m, m),
        )
        for m in modes_in_order
    ]
    handles.append(
        Line2D(
            [0], [0],
            color=PROP_LINE_COLOR,
            linestyle=":",
            linewidth=1.2,
            label=f"Proportional representation ({iprop * 100:.1f}%)",
        )
    )
    # Reserve a fixed strip (not a fraction) for the legend so it sits just above
    # the axes regardless of how tall the figure grows with the row count.
    rect_top = 1 - 0.5 / fig_height
    fig.tight_layout(rect=[0, 0, 1, rect_top])
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, rect_top),
        ncol=len(handles),
        fontsize=7,
        frameon=True,
    )

    fig_path = (
        figs_dir
        / f"{run_name}_{num_dist}x{seats_per_district}_district_bubbles_by_model.png"
    )
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_district_model_bubbles(
    df: pd.DataFrame,
    config,
    focal_group: str,
    iprop: float,
    figs_dir: Path,
    run_name: str,
) -> None:
    """
    One figure per districting configuration showing per-district focal-seat
    distributions. Each district is a y-axis row and each voter model a
    translucent color, so the spread (and overlap) across models is visible per
    district. Operates on the district-level table (not the plan aggregate).
    """
    for (num_dist, seats_per_district), config_rows in df.groupby(
        ["num_districts", "seats_per_district"]
    ):
        _plot_district_bubbles_for_config(
            config_rows,
            config,
            iprop,
            figs_dir,
            run_name,
            num_dist,
            seats_per_district,
        )


def _combined_distribution_for_run(summary_csv: Path) -> Optional[pd.DataFrame]:
    """
    Read one run's summary CSV and return its model-averaged ("Combined")
    focal-seat distribution: a frame with columns ``focal_seats`` and ``count``.

    Returns None for empty/unreadable summaries so callers can skip them.
    """
    df = pd.read_csv(summary_csv)
    if df.empty:
        return None

    df_plan = aggregate_to_plan_level(df)
    combined = _occurrence_counts(df_plan)
    combined = combined[combined["mode"] == COMBINED_MODE]
    if combined.empty:
        return None

    # A run may contain more than one election method or district config; collapse
    # them into a single per-run distribution over focal-seat counts.
    return combined.groupby("focal_seats", as_index=False)["count"].sum()


def plot_combined_bubbles_all_runs(
    config,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Compare every completed run in a single bubble figure.

    Scans ``outputs/*/summaries/*_summary.csv`` for finished runs and draws one
    row per run (y-axis), where bubble area encodes the model-averaged
    ("Combined") number of plans that produced each focal-seat count (x-axis). A
    dotted line marks the focal group's proportional-representation seat share.

    Args:
        config: Any run's parsed config; used only for the seat-count axis range
            and the population-share reference line, which are shared across runs.
        output_dir: Where to write the figure. Defaults to
            outputs/cross_run_summaries/figures.

    Returns:
        Path to the written figure, or None if no completed runs were found.
    """
    summary_paths = sorted(Path("outputs").glob("*/summaries/*_summary.csv"))

    # Each entry: (sort_key, display_label, combined_distribution_df).
    runs: List[Tuple[Tuple[int, int, str], str, pd.DataFrame]] = []
    for path in summary_paths:
        combined = _combined_distribution_for_run(path)
        if combined is None:
            continue
        # run_name doubles as the directory name; read it from the data so the
        # label matches the config even if the path layout changes.
        df_head = pd.read_csv(path, usecols=["run_name", "num_districts", "seats_per_district"])
        label = str(df_head["run_name"].iloc[0])
        num_dist = int(df_head["num_districts"].min())
        seats_per_district = int(df_head["seats_per_district"].min())
        runs.append(((num_dist, seats_per_district, label), label, combined))

    if not runs:
        print("[summarize_results] No completed runs found for cross-run bubble plot.")
        return None

    # Order rows by districting configuration so related systems sit together.
    runs.sort(key=lambda r: r[0])
    labels = [label for _, label, _ in runs]

    # Single bubble-area scale across every run so sizes are comparable.
    max_count = max(c["count"].max() for _, _, c in runs)
    size_scale = (BUBBLE_MAX_AREA - BUBBLE_MIN_AREA) / max_count if max_count > 0 else 0

    # Seat axis and proportional-representation line are run-independent.
    observed_max_seats = max(int(c["focal_seats"].max()) for _, _, c in runs)
    total_seats = max(int(config["total_seats"]), observed_max_seats)
    iprop = _focal_population_share(config)
    i_share = iprop * total_seats

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(labels) + 2))

    for y, (_, _, combined) in enumerate(runs):
        ax.scatter(
            combined["focal_seats"],
            [y] * len(combined),
            s=BUBBLE_MIN_AREA + combined["count"] * size_scale,
            color=BUBBLE_COLOR,
            alpha=0.7,
            edgecolor="gray",
            linewidth=0.5,
        )

    ax.axvline(i_share, color=PROP_LINE_COLOR, linestyle=":", linewidth=1.2)

    ax.set_xlim(0, total_seats + 1)
    ax.set_xticks(range(0, total_seats + 2, 1))
    ax.set_xticklabels([str(x) if x % 2 == 0 else "" for x in range(0, total_seats + 2)])
    ax.set_xlabel("City Council Seats")

    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([label.replace("_", " ") for label in labels])

    ax.tick_params(axis="both", which="major", labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.set_title("Combined outcomes across runs", fontsize=11)

    prop_handle = Line2D(
        [0], [0],
        color=PROP_LINE_COLOR,
        linestyle=":",
        linewidth=1.2,
        label=f"Proportional representation ({iprop * 100:.1f}%)",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(
        handles=[prop_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        fontsize=7,
        frameon=True,
    )

    if output_dir is None:
        output_dir = Path("outputs") / "cross_run_summaries" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "combined_bubbles_all_runs.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[summarize_results] Wrote cross-run figure: {fig_path}")
    return fig_path


def summarize_results(config) -> Path:
    """
    Aggregate election results into a summary csv and produce histogram figures.

    Args:
        config: Parsed config dict.

    Outputs:
        - outputs/<run_name>/summaries/<run_name>_summary.csv: one row per
          (replicate, plan, district, election_method) tuple, with columns for plan,
          mode, district_id, rep, focal_seats, the population columns from config, and
          combined_support.
        - outputs/<run_name>/summaries/figures/*.png: one histogram per
          (district_count, seats_per_district, election_method) showing the
          distribution of focal-group seats across modes.

    Returns:
        Path to the summary directory.
    """
    run_name = str(config["run_name"])
    focal_group = str(config["focal_group"])

    iprop, iprop_turnout, i_cs_turnout = _compute_representation_baselines(config)

    results_dir, summary_dir, figs_dir = _prepare_directories(run_name)

    df = build_summary_dataframe(config, results_dir, i_cs_turnout)

    # Persist the tidy, district-level table.
    csv_path = summary_dir / f"{run_name}_summary.csv"
    df.to_csv(csv_path, index=False)

    df_plan = aggregate_to_plan_level(df)

    plot_representation_histograms(
        df_plan, config, focal_group, iprop, i_cs_turnout, figs_dir, run_name
    )

    plot_representation_bubbles(df_plan, config, focal_group, iprop, figs_dir, run_name)

    # Per-district view: districts on the y-axis, voter models overlaid by color.
    plot_district_model_bubbles(df, config, focal_group, iprop, figs_dir, run_name)

    print(f"[summarize_results] Wrote CSV: {csv_path}")
    print(f"[summarize_results] Figures in: {figs_dir}")
    return summary_dir
