# pipeline-config — Config Builder UI

A small visual interface for creating new run config files for this project.
It renders every field found in the JSON files under `configs/` and can write
a new config straight into that directory.

## Run it

From the **project root**:

```bash
python pipeline-config/server.py
```

Then open <http://localhost:8000> in a browser.

The server uses only the Python standard library, so nothing extra needs to be
installed.

## Using it

0. To start from an existing config, pick it in **Load Existing Config** and
   click **Load into form** — every field (including voting rules, district
   configs, slates, turnout, and the cohesion/alpha matrices) is prepopulated
   from that file's JSON, and the filename is pre-filled so you can save back to
   it or under a new name.
1. Fill in the form fields (they are pre-populated with sensible defaults that
   match the existing configs). **`geodata_path`** is a dropdown of the geodata
   files found under `data/`; picking one reads that file's schema and turns
   **`population_column`**, **`population_vap_column`**, and
   **`pop_of_interest_column`** into dropdowns of that file's actual columns.
2. Add one or more **voting rules** (`voting_configs`) and **district configs**.
   Pick a rule and the form shows only the parameters that rule accepts
   (transcribed from `README.md` → *VoteKit Voting Rule Parameters*) — e.g.
   `n_seats`, `quota`, `transfer`, `simultaneous`, `tiebreak`,
   `fpv_tie_convention`, `budget`, etc. Any field left on **(default)** / blank
   is omitted so VoteKit uses its own default. Click a rule chip to load it back
   into the editor and adjust it.
3. Watch the **Live JSON Preview** on the right update as you type.
4. Click **Generate into configs/** to write `<filename>.json` into the
   project's `configs/` directory. If the file exists you'll be prompted to
   tick **Overwrite**.

### Without the server

If you just open `index.html` as a file (`file://`), the **Generate** button
can't reach the server and the geodata/column dropdowns fall back to their
default names (rather than reading `data/`). **Download JSON** and **Copy JSON**
still work — save the file and drop it into `configs/` yourself.

## Fields

The form covers the full config schema used by `run.py`:

| Field | Notes |
|-------|-------|
| `run_name`, `seed`, `num_reps` | run identity / reproducibility |
| `geodata_path`, `gerrychain_output_dir` | data & output paths |
| `population_column`, `population_vap_column`, `pop_of_interest_column` | geodata columns |
| `chain_length`, `num_subsamples`, `epsilon` | Markov chain sampling |
| `total_seats`, `num_voters` | election sizing |
| `voting_configs` | map of rule → its VoteKit parameters (rule-specific fields) |
| `district_configs` | list of `{ num_districts, winners }` |
| `slate_to_candidates`, `focal_group` | two blocs + candidate lists |
| `turnout` | per-slate turnout rate |
| `cohesion_parameters`, `alphas` | slate × slate matrices |

Valid voting-rule names come from `voting-rule-reference.md`.
