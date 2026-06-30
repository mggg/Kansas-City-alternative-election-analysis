# Kansas-City-alternative-election-analysis

## District Generator Parameters

- `run_name`: Name of the simulation run. Used to create the output directory and output file names
- `seed`: Random seed used to ensure reproducibility of the Markov Chain simulation.
- `chain_length`: Number of Markov Chain iterations (i.e., the number of districting plans generated).
- `n_district`: Number of districts to generate in the simulated districting plans.
- `epsilon` (`seed_epsilon`): Population tolerance used when generating the **initial random districting plan**. Each district is initialized within ±`epsilon` of the ideal district population. |
- `geodata_path`: Path to the GeoPackage (`.gpkg`) containing the geographic units (precincts or census blocks) and their demographic attributes.
- `population_column`: Column name of total population variable.

Note: This implementation uses two population tolerances. seed_epsilon controls the generation of the initial random partition, while chain_epsilon controls the maximum population deviation allowed throughout the Markov Chain. This allows the chain to begin from a valid initial plan while enforcing a consistent population balance (±5%) during sampling.

## VoteKit Voting Rule Parameters

All classes take `profile` as the first positional argument. The type required differs:
- **Ranking elections** require a `RankProfile`
- **Score elections** (Rating/Approval/Cumulative/Limited) require a `ScoreProfile`
- **`BlockPlurality`** accepts *either* and dispatches internally

A note on **deprecated aliases**: many classes still accept legacy kwargs via `_handle_deprecated_kwargs` — `m` → `n_seats`, and `k` → `per_candidate_limit` or `budget` (depending on class). New code should use the new names.

### Shared base parameters

From `Election` / `RankingElection` (usually set internally by each subclass, not passed by users):
- `score_function` — `Callable[[profile], dict[str, float]]`, default `None`
- `sort_high_low` — `bool`, default `True`
- `n_seats` — `int`, default `1` (must be positive)

### Ranking elections

| Class | Parameters (beyond `profile`) |
|---|---|
| **STV** | `n_seats=1`, `transfer=fractional_transfer` (a Callable), `quota="droop"` ‹"droop"/"hare"›, `simultaneous=True`, `tiebreak=None` ‹"borda"/"random"› |
| **FastSTV** | `n_seats=1`, `transfer="fractional"` ‹"fractional"/"fractional_random"/"cambridge_random"/"random"›, `quota="droop"`, `simultaneous=True`, `tiebreak=None` ‹"borda"/"random"/"cambridge_random"› |
| **IRV** | `quota="droop"`, `tiebreak=None` (n_seats fixed to 1) |
| **FastIRV** | `quota="droop"`, `tiebreak=None` |
| **SequentialRCV** / **FastSequentialRCV** | `n_seats=1`, `quota="droop"`, `simultaneous=True`, `tiebreak=None` |
| **AlbanySTV** | same as FastSTV (forces `dynamic_threshold=True`) |
| *(internal `NumpyInnerSTV`)* | adds `dynamic_threshold=False`, `block_rcv=False` |
| **Plurality** | `n_seats=1`, `tiebreak=None` ‹"random"/"borda"›, `fpv_tie_convention="average"` ‹"high"/"low"/"average"› |
| **SNTV** | `n_seats=1`, `tiebreak=None` (wrapper around Plurality) |
| **Borda** | `n_seats=1`, `score_vector=None` (defaults to `(n, n-1, …, 1)`), `tiebreak=None` ‹"random"/"first_place"›, `scoring_tie_convention="low"` |
| **Alaska** | `m_1=2` (first-round semifinalists), `m_2=1` (final seats), `transfer=fractional_transfer`, `quota="droop"`, `simultaneous=True`, `tiebreak=None`, `fpv_tie_convention="average"` |
| **TopTwo** | `tiebreak=None`, `fpv_tie_convention="average"` |
| **CondoBorda** | `n_seats=1` |
| **DominatingSets** | *(none beyond profile)* |
| **PluralityVeto** / **SerialVeto** | `n_seats=1`, `tiebreak="first_place"` ‹"first_place"/"borda"/"random"/"lex"›, `scoring_tie_convention="average"` |
| **SimultaneousVeto** | `n_seats=1`, `candidate_weights="first_place"` ‹"first_place"/"uniform"/"borda"/"harmonic"/ dict / int›, `tiebreak="first_place"` ‹+"remaining_score"/"veto_pressure"/"lex"›, `scoring_tie_convention="average"`, `return_all_tied_winners=False` |
| **RandomDictator** | `n_seats=1`, `fpv_tie_convention="average"` |
| **BoostedRandomDictator** | `n_seats=1`, `fpv_tie_convention="average"` |
| **RankedPairs** | `tiebreak="lexicographic"`, `n_seats=1` |
| **Schulze** | `tiebreak="lexicographic"`, `n_seats=1` |

### Score elections

| Class | Parameters (beyond `profile`) |
|---|---|
| **GeneralRating** | `n_seats=1`, `per_candidate_limit=1`, `budget=None` (total points/voter), `tiebreak=None` ‹"random"› |
| **Rating** | `n_seats=1`, `per_candidate_limit=1`, `tiebreak=None` |
| **Limited** | `n_seats=1`, `budget=1` (must be ≤ n_seats), `tiebreak=None` |
| **Cumulative** | `n_seats=1`, `tiebreak=None` (sets `budget=n_seats`, `per_candidate_limit` unbounded) |
| **Approval** | `n_seats=1`, `tiebreak=None` (sets `per_candidate_limit=1`, no budget) |
| **BlockPlurality** | `n_seats=1`, `budget=None` (→ n_seats), `tiebreak=None`, `scoring_tie_convention="low"` (only used for RankProfile input) |
| **BlocPlurality** | *(deprecated alias of BlockPlurality for ScoreProfile)* — `n_seats=1`, `budget=None`, `tiebreak=None` |

### Cross-cutting parameters

- **`n_seats`** — number of seats; nearly universal.
- **`tiebreak`** — present almost everywhere, but the *accepted values differ by class* (e.g., STV uses borda/random/cambridge_random; Condorcet methods use "lexicographic"; veto methods add "lex"/"veto_pressure"; rating methods only accept None/"random"). Default `None` means a tie *raises an error*.
- **`quota`** — STV family only ("droop"/"hare").
- **`transfer`** — STV/Alaska only.
- **`budget`** / **`per_candidate_limit`** — score elections only.
- **`*_tie_convention`** (`fpv_tie_convention` / `scoring_tie_convention`) — how tied scores split points ("high"/"average"/"low").