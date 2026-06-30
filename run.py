"""
Run pipeline step by step.
"""
from pathlib import Path
from glob import glob
import json
from pipeline.district_generator import generate_districts
from pipeline.settings_generator import generate_settings
from pipeline.profile_generator import generate_profiles
from pipeline.simulate_elections import simulate_elections
from pipeline.data_generator_blocks import generate_data
from pipeline.summarize_results import summarize_results

def load_all_config_files():
    all_config_files = [load_config(path) for path in glob(f"configs/*.json")]
    return all_config_files


def load_config(config_path: str) -> dict:
    """Load config from JSON file."""
    with open(config_path) as f:
        return json.load(f)

if __name__ == "__main__":
    # Load config
    configurations = load_all_config_files()
    # print(configurations[5)

    for config in configurations:
        print(f"Run name: {config['run_name']}")
        print(f"Districts: {config['district_configs']}")
        print(f"Chain length: {config['chain_length']}")
        
        # Step 0 - Data Generation
        print("\n=== Running generate_data ===")
        generate_data()
        print("=== generate_data complete ===")

        # Step 1 — Generate districts
        print("\n=== Running generate_districts ===")
        generate_districts(config)
        print("=== generate_districts complete ===")

        # Step 2 - Settings generator
        print("\n=== Running generate_settings ===")
        generate_settings(config)
        print("=== generate_settings complete ===")

        # Step 3 - Profile Generator
        print("\n=== Running Profile Generations ===")
        generate_profiles(config)

    #     # Step 4 - Simulate Elections
    #     print("\n=== Running Election Simulations ===")
    #     simulate_elections(config)

    #     # Step 5 - Summarize Results
    #     print("\n=== Summarizing Results ===")
    #     summarize_results(config)