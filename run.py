"""
Run pipeline step by step.
"""
from pathlib import Path
import json
from pipeline.district_generator import generate_districts
from pipeline.settings_generator import generate_settings

def load_config(config_path: str) -> dict:
    """Load config from JSON file."""
    with open(config_path) as f:
        return json.load(f)

if __name__ == "__main__":
    # Load config
    config = load_config("configs/baseline.json")
    print(f"Run name: {config['run_name']}")
    print(f"Districts: {config['district_configs']}")
    print(f"Chain length: {config['chain_length']}")
    
    # Step 1 — Generate districts
    print("\n=== Running generate_districts ===")
    generate_districts(config)
    print("=== generate_districts complete ===")

    # Step 2 - Settings generator
    print("\n=== Running generate_settings ===")
    generate_settings(config)
    print("=== generate_settings complete ===")