# SynDebug

Quantum program assertions for quantum computers using QEC codes.

The repo is organized as follows:

patch_register.py contains the code for QecProgram, PatchRegister, and CodePatch classes.

logical_gadgets.py contains code for transversal logical operators used in various demos.

qec_assertions.py contains the assertion method definitions.

test_init_assert.py and test_entangle_assert.py contain the code for generating the data in table 2 and table 3 respectively

test_init_assert_exhaustive.py and test_entangle_assert_exhaustive.py exhaustively iterate through combinations of code type, distance, and initializations.

analyze_exhaustive_results.py checks and summarizes the output of the exhaustive sweeps.

Saved versions of the paper results are stored in the results folder

## Reproducing the results

### Setup

Tested with Python 3.12. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run every script below from the repo root, since output paths are relative to the current directory.

Each script writes a new CSV with a timestamp (`YYYYMMDD_HHMMSS`) in its name, so the saved paper results are never overwritten.

### Table 2: classical initialization assertions

```bash
python test_init_assert.py
```

This writes:

- `results/initialization/classical_results_<timestamp>.csv`: circuit depth with and without the assertion
- `results/timings/classical_timing_<timestamp>.csv`: simulation time with and without the assertion

The saved paper versions are `results/initialization/table_results/classical_results_table.csv` and `results/timings/classical_timing.csv`.
### Table 3: entanglement assertions

```bash
python test_entangle_assert.py
```

This writes `results/entanglement/entanglement_results_<timestamp>.csv`. The saved paper version is `results/entanglement/table/entanglement_results_table.csv`
### Exhaustive sweeps

```bash
python test_init_assert_exhaustive.py
python test_entangle_assert_exhaustive.py
```

These write `results/initialization/exhaustive/exhaustive_classical_results<timestamp>.csv` (1008 cases) and `results/entanglement/exhaustive/exhaustive_entanglement_results<timestamp>.csv` (48 cases)


Then summarize them:

```bash
python analyze_exhaustive_results.py
```

By default this analyzes the newest CSV in each `exhaustive/` folder, judged by the timestamp in the filename. The saved paper CSVs have no timestamp and count as oldest, so on a fresh clone they are the ones analyzed. To analyze specific files, such as the paper results after a rerun:

```bash
python analyze_exhaustive_results.py \
    --classical results/initialization/exhaustive/exhaustive_classical_results.csv \
    --entanglement results/entanglement/exhaustive/exhaustive_entanglement_results.csv
```

