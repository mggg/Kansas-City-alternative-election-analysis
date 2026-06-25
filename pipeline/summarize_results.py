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
from typing import Any, Dict, List, Optional
import geopandas as gpd

import pandas as pd
import matplotlib.pyplot as plt

from pipeline.utils.helpers import (
    parse_district_configs,
    parse_plan_district_rep_from_path,
    count_focal_winners,
    load_json,
    find_settings_file,
    get_non_focal_group,
)


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

    # --- Read the high-level knobs from the config -----------------------------
    # Everything the pipeline needs is centralized in the config dict, so we pull
    # out the pieces this step cares about up front and keep the rest implicit.
    run_name = str(config["run_name"])
    # district_configs may use either the new {"num_districts", "winners"} schema or
    # the legacy {<n>: <winners>} schema; the helper normalizes both into objects.
    district_configs = parse_district_configs(config["district_configs"])
    focal_group = str(config["focal_group"])
    # slate_to_candidates maps a bloc label (e.g. "A") to the candidate ids it ran.
    # It is optional here: count_focal_winners can fall back to a prefix match.
    slate_to_candidates = config.get("slate_to_candidates", {}) or {}

    # --- Statewide focal-group share, straight from the geodata ----------------
    # We need a population baseline to draw the "proportional representation" lines
    # on each figure. This is the *overall* focal share before any districting.
    geodata_path = Path(config["geodata_path"])
    gdf = gpd.read_file(geodata_path)
    # vap  = total (voting-age) population across all precincts
    # ivap = population of the group of interest (the focal group) across all precincts
    vap = sum(gdf[config["population_column"]])
    ivap = sum(gdf[config["pop_of_interest_column"]])
    iprop = ivap / vap  # raw focal-group population proportion

    # --- Turnout adjustment ----------------------------------------------------
    # The raw population share is not what shows up at the ballot box: the two blocs
    # can turn out at different rates. We weight each bloc by its turnout to get the
    # effective share of *voters* that belong to the focal group.
    turnout = config["turnout"]
    cohesion_parameters = config["cohesion_parameters"]
    if len(turnout) != 2:
        raise ValueError("Turnout does not have exactly two keys")
    non_focal_group = get_non_focal_group(config)
    # Bayes-style reweighting: focal voters / (focal voters + non-focal voters).
    iprop_turnout = (
        iprop * turnout[focal_group]
        / (iprop * turnout[focal_group] + (1 - iprop) * turnout[non_focal_group])
    )

    # --- Combined support ------------------------------------------------------
    # "Combined support" is the share of the *vote* that flows to focal candidates.
    # It blends two sources: focal voters who back focal candidates (cohesion), plus
    # non-focal voters who cross over to focal candidates. cohesion_parameters[g][h]
    # is the share of bloc g's votes that go to bloc h's slate.
    focal_group_cohesion = cohesion_parameters[focal_group]
    non_focal_group_cohesion = cohesion_parameters[non_focal_group]
    i_cs_turnout = (
        iprop_turnout * focal_group_cohesion[focal_group]
        + (1 - iprop_turnout) * non_focal_group_cohesion[focal_group]
    )

    # The three voter models the pipeline simulates. These match the subdirectory
    # names created upstream by generate_profiles / simulate_elections.
    modes = ["slate_pl", "slate_bt", "cambridge"]

    # --- Input root ------------------------------------------------------------
    # simulate_elections writes one JSON per (mode, district config) under here.
    results_dir = Path("outputs") / f"{run_name}" / "election_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Could not find election results directory: {results_dir}")

    # --- Output roots ----------------------------------------------------------
    # Layout mirrors the rest of the pipeline: outputs/<run_name>/summaries/...
    # (run.py's has_valid_summaries() looks for exactly this CSV and figures dir.)
    summary_dir = Path("outputs") / f"{run_name}" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = summary_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # We accumulate one dict per row and build the DataFrame once at the end; this
    # is much faster than growing a DataFrame incrementally.
    rows: List[Dict[str, Any]] = []

    # Map the raw method keys emitted by simulate_elections to display names.
    method_name_map = {
        "stv": "STV",
        "plurality": "Plurality",
        "irv": "IRV",
    }

    # --- Walk every district config x voter model x results file ---------------
    for dc in district_configs:
        # Settings files are grouped by district count (one folder per num_districts),
        # matching settings_generator's outputs/<run>/settings/<n>/ layout.
        settings_dir = Path("outputs") / f"{run_name}" / "settings" / str(dc.num_districts)

        for mode in modes:
            mode_dir = results_dir / mode
            if not mode_dir.exists():
                continue

            for rf in sorted(mode_dir.glob("*.json")):
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
                    continue

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

                # --- One row per simulated profile (and per election method) ----
                for idx, result in enumerate(election_results):
                    # Recover (plan, district, replicate) by parsing the profile path,
                    # e.g. ..._district_plan_003_district_07_v1.csv -> (3, 7, 1).
                    plan, district, rep = parse_plan_district_rep_from_path(profile_files[idx])

                    # Join back to the settings file for this district to recover the
                    # population totals that profile was built from. The helper is
                    # tolerant of small naming differences via glob fallbacks.
                    settings_path = find_settings_file(
                        settings_dir, config["run_name"], plan=plan, district=district
                    )
                    settings_data = load_json(settings_path) if settings_path else {}
                    total_vap = settings_data.get(config["population_column"], None)
                    total_ivap = settings_data.get(config["pop_of_interest_column"], None)

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
                            "election_method": method_name_map.get(method_key, method_key.upper()),
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

    df = pd.DataFrame(rows)
    df = df.sort_values(["mode", "rep", "num_districts", "plan", "district_id"])

    # Persist the tidy, district-level table.
    csv_path = summary_dir / f"{run_name}_summary.csv"
    df.to_csv(csv_path, index=False)

    # --- Aggregate to the plan level ------------------------------------------
    # A "plan" is one sampled districting map; representation is naturally a
    # whole-map quantity, so we sum focal seats across that plan's districts. Each
    # (plan, mode, method, replicate) becomes one data point in the histograms.
    df_plan = (
        df.groupby(
            ["plan", "num_districts", "seats_per_district", "mode", "election_method", "rep"],
            as_index=False,
        )
        .agg({"focal_seats": "sum"})
    )

    # Fixed colors / labels so every figure reads the same way.
    mode_colors = {
        "cambridge": "#E32636",
        "slate_bt": "#FFBF00",
        "slate_pl": "#8DB600",
    }

    legend_mapping = {
        "slate_bt": "Deliberative",
        "slate_pl": "Impulsive",
        "cambridge": "Cambridge",
    }
    desired_order = ["slate_pl", "slate_bt", "cambridge"]

    # --- One histogram per (district count, seats, election method) ------------
    for (num_dist, seats_per_district, elm), group_distn in df_plan.groupby(
        ["num_districts", "seats_per_district", "election_method"]
    ):
        fig, ax = plt.subplots(figsize=(6, 4))

        # Overlay one translucent histogram per voter model; track the tallest bar
        # so we can scale the y-axis (and place text labels) consistently.
        max_bin_height = 0

        for mode, group_mode in group_distn.groupby("mode"):
            if group_mode["focal_seats"].empty:
                continue

            # One integer bin per possible focal-seat count in this group.
            counts, bins, patches = ax.hist(
                group_mode["focal_seats"],
                bins=range(
                    int(group_mode["focal_seats"].min()),
                    int(group_mode["focal_seats"].max()) + 2,
                ),
                align="left",
                edgecolor="gray",
                linewidth=0.5,
                color=mode_colors.get(mode, "xkcd:light gray"),
                alpha=0.5,
                label=mode,
            )

            if len(counts) > 0:
                max_bin_height = max(max_bin_height, counts.max())

        # Thin, uniform spines.
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # x-axis spans 0..total_seats; give the y-axis 20% headroom for the labels.
        total_seats = config["total_seats"]
        ylim = max_bin_height * 1.2 if max_bin_height > 0 else 1

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

        # --- Legend: modes only, renamed and in a fixed order ------------------
        handles, labels = ax.get_legend_handles_labels()
        handle_map = {label: handle for handle, label in zip(handles, labels) if label in legend_mapping}

        ordered_handles, ordered_labels = [], []
        for mode_key in desired_order:
            if mode_key in handle_map:
                ordered_handles.append(handle_map[mode_key])
                ordered_labels.append(legend_mapping[mode_key])

        ax.legend(ordered_handles, ordered_labels, title="Mode", fontsize=8)

        # --- Reference lines: two notions of "proportional" representation -----
        # i_cs_share : seats implied by combined *support* (votes for focal cands).
        # i_share    : seats implied by raw focal-group *population* share.
        # Comparing where the histogram mass falls against these lines is the whole
        # point of the figure.
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

        fig_path = figs_dir / f"{run_name}_{num_dist}x{seats_per_district}_{elm}_bymode.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[summarize_results] Wrote CSV: {csv_path}")
    print(f"[summarize_results] Figures in: {figs_dir}")
    return summary_dir
