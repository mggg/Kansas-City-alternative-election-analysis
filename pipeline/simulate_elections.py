"""
Run elections on generated voter profiles and record the winners.

Reads voter profile CSV files and runs the appropriate election 
rule (STV for multi-seat districts, plurality and IRV for single-seat districts), 
and writes aggregated election results to JSON files.
"""

import json
import inspect
from glob import glob
from pathlib import Path
from joblib import Parallel, delayed
from votekit import RankProfile, ScoreProfile, elections
from typing import List, Iterable, Any, get_args
from dataclasses import dataclass



# Optional progress bar for joblib.
try:
    from joblib_progress import joblib_progress 
except Exception: 
    joblib_progress = None 


@dataclass(frozen=True)
class DistrictConfig:
    """One district configuration: number of districts and seats won per district."""
    num_districts: int
    winners: int


def _required_profile(cls):
    annotation = inspect.signature(cls.__init__).parameters["profile"].annotation
    expected_types = get_args(annotation)
    return expected_types if expected_types else (annotation,)  # tuple of acceptable profile types

def _import_voting_rules_from_vote_kit(rules: str) -> dict:
    classes = {rule: getattr(elections, rule) for rule in rules}
    return classes


def _candidate_list_from_elected(elected: Iterable[set]) -> List[str]:
    """
    Flatten votekit election output (iterable of singleton sets) into a list of strings.

    Args:
        elected: Iterable of singleton sets, as returned by votekit election methods.

    Returns:
        List of candidate id strings in election order. Empty sets are skipped silently.
    """
    winners: List[str] = []
    for s in elected:
        if s:
            winners.append(str(next(iter(s))))
    return winners

def _process_profile(profile_file: str | Path, voting_configs: dict) -> List[str]:
    """
    Load a voter profile csv and run an election to determine winners.
    Dynamically load election classes based on voting_config settings

    Args:
        profile_file: Path to the voter profile csv.
        voting_configs: Election and voting settings specified in configuration files

    Returns:
        {[type]: [winner_ids]} e.g. { "stv": ["A2", "B1", "B3"] }

    TO-DO: Figure out how to use RankProfile OR ScoreProfile for BlockPlurality if desired. 
        Current default is RankProfile.
    """
    profile_path = Path(profile_file)

    # Dynamically obtain election type classes from VoteKit based on the voting
    # rule configuration provided.

    election_classes = _import_voting_rules_from_vote_kit(voting_configs.keys())
    results = {}

    # For each election class, run the election simulation

    for rule, election_class in election_classes.items():

        # We will use type annotations from the election class object to determine what type
        # of profile we need to create for object instantiation. If RankProfile is in that
        # accepted class list, we use it. If not, use ScoreProfile.

        profile_types = _required_profile(election_class)

        profile_class = RankProfile if RankProfile in profile_types else ScoreProfile

        profile = profile_class.from_csv(profile_path)

        # The parameters used in the class constructors are specified in the
        # configuration files, under voting_configs. We use keyword argument spreading
        # to give us flexibility in execution

        elected = election_class(profile, **voting_configs[rule]).get_elected()
        results[rule] = _candidate_list_from_elected(elected)

    return results

def _parse_district_configs(raw: Any) -> List[DistrictConfig]:
    """
    Parse the district_configs field from the config file into DistrictConfig objects.
    accepts two schemas:
      - newer: [{"num_districts": 5, "winners": 2}, ...]
      - older: [{<num_districts>: <winners>}, ...] e.g. [{80: 1}, {20: 4}]

    Args:
        raw: The raw district_configs value from the config (expected to be a list).

    Returns:
        List of DistrictConfig(num_districts, winners).

    Raises:
        ValueError: If raw is not a list or entries don't match either schema.
    """
    if not isinstance(raw, list):
        raise ValueError("district_configs must be a list")

    parsed: List[DistrictConfig] = []
    for item in raw:
        if isinstance(item, dict) and "num_districts" in item and "winners" in item:
            parsed.append(DistrictConfig(int(item["num_districts"]), int(item["winners"])))
        elif isinstance(item, dict) and len(item) == 1:
            (k, v), = item.items()
            parsed.append(DistrictConfig(int(k), int(v)))
        else:
            raise ValueError(
                "Each district_configs entry must be either "
                '{"num_districts": <int>, "winners": <int>} or {<int>: <int>}.'
            )
    return parsed


def simulate_elections(config) -> None:
    """
    Run elections in parallel over all voter profiles.

    Args:
        config: Parsed config dict.

    Outputs:
        One json file per (mode, district_count, winners) combination at
        outputs/election_results/<run_name>_election_results/<mode>/
        <run_name>_<n>_districts_<w>_winners_for_voter_mode_<mode>.json.
        Each file contains a "election_results" list where each entry corresponds
        to one profile file:
          - multi-seat: {"stv": [...]}
          - single-seat: {"plurality": [...], "irv": [...]}

    Returns:
        None.
    """
    run_name = str(config["run_name"])
    district_configs = _parse_district_configs(config["district_configs"])

    modes = ["slate_pl", "slate_bt", "cambridge"]
    # Use all available cores by default. Set SIMULATE_ELECTIONS_N_JOBS=1 to run
    # serially in the main process so breakpoints inside _process_profile are hit
    # under the debugger (joblib worker subprocesses are not debugged otherwise).
    n_jobs = -1

    out_root = Path("outputs") / f'{run_name}' / "election_results" 
    out_root.mkdir(parents=True, exist_ok=True)

    # run elections for each voter model
    for mode in modes:
        # profile path
        profile_folder = Path(f"./outputs/{run_name}/profiles/{mode}/")

        output_dir = out_root / mode
        output_dir.mkdir(parents=True, exist_ok=True)

        for dc in district_configs:
            all_profile_files = glob(f"{profile_folder}/{dc.num_districts}/*.csv")

            desc = f"Running elections for {dc.num_districts} districts, {dc.winners} winner(s), mode={mode}"
            if joblib_progress is not None:
                ctx = joblib_progress(description=desc, total=len(all_profile_files))
            else:
                ctx = None

            if ctx is not None:
                with ctx:
                    results_list = Parallel(n_jobs=n_jobs)(
                        delayed(_process_profile)(pf, config["voting_configs"]) for pf in all_profile_files
                    )
            else:
                print(f"[simulate_elections] {desc} (no joblib_progress installed)")
                results_list = Parallel(n_jobs=n_jobs)(
                    delayed(_process_profile)(pf, config["voting_configs"]) for pf in all_profile_files
                )


            # write all winners for this district/mode combo to one json file
            out_path = output_dir / (
                f"{run_name}_{dc.num_districts}_districts_{dc.winners}_winners_for_voter_mode_{mode}.json"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_name": run_name,
                        "voter_mode": mode,
                        "district_num": dc.num_districts,
                        "winners_per_district": dc.winners,
                        "profile_files": all_profile_files,
                        "election_results": results_list,
                    },
                    f,
                    indent=2,
                )

            print(f"[simulate_elections] Wrote: {out_path}")
