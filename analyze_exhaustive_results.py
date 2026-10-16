"""Analyze the exhaustive SynDebug sweep results.

For each of the two result sets — the classical-init assertions and the
entanglement assertions — this loads one CSV, verifies that every case passed
its assertion, and reports the average logical-circuit depth with and without
SynDebug along with the average percent change.

By default the newest CSV in each folder is used, judged by the timestamp the
sweep scripts put in the filename; a CSV without one (such as the saved paper
results) counts as oldest. Pass --classical or --entanglement to analyze a
specific file instead.

The two datasets are summarized separately. If any case failed its assertion,
the failing cases are listed and the script exits non-zero *before* reporting
averages (the assertion check is treated as a hard gate).

Run:  python analyze_exhaustive_results.py [--classical CSV] [--entanglement CSV]
"""

import argparse
import csv
import glob
import os
import re
import sys
from statistics import mean

# (label, command-line flag, directory) for each result set.
RESULT_SETS = [
    ("Classical init assertions", "--classical",    "results/initialization/exhaustive"),
    ("Entanglement assertions",   "--entanglement", "results/entanglement/exhaustive"),
]

# Sweep scripts name their output <prefix>YYYYMMDD_HHMMSS.csv; that format
# sorts chronologically as plain text.
TIMESTAMP = re.compile(r"(\d{8}_\d{6})\.csv$")


def load_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def pick_csv(directory: str, flag: str) -> tuple[str, str]:
    """Return the newest CSV in *directory* and a note on how it was chosen."""
    matches = glob.glob(os.path.join(directory, "*.csv"))
    if not matches:
        raise FileNotFoundError(f"no CSV files found in {directory!r}")
    if len(matches) == 1:
        return matches[0], "only CSV in folder"

    def stamp(path: str) -> str:
        m = TIMESTAMP.search(os.path.basename(path))
        return m.group(1) if m else ""  # no timestamp counts as oldest

    ranked = sorted(matches, key=stamp)
    if stamp(ranked[-1]) == stamp(ranked[-2]):
        raise ValueError(
            f"can't tell which CSV in {directory!r} is newest: "
            f"{sorted(os.path.basename(m) for m in matches)}; "
            f"pass one with {flag}"
        )
    return ranked[-1], f"newest of {len(matches)} in folder"


def analyze(label: str, path: str, note: str) -> bool:
    """Summarize one result set. Returns True if every case passed."""
    rows = load_rows(path)

    print(f"\n=== {label} ===")
    print(f"  file  : {path}  ({note})")
    print(f"  cases : {len(rows)}")

    if not rows:
        print("  !! no rows to analyze")
        return False

    failures = [r for r in rows if not parse_bool(r["assertion_passed"])]
    if failures:
        print(f"  !! {len(failures)} case(s) FAILED their assertion:")
        for r in failures:
            extra = f" n={r['n_patches']}" if "n_patches" in r else ""
            print(
                f"       {r['name']} d={r['dist']}{extra} "
                f"state={r['state_string']!r}"
            )
        return False

    print("  all cases passed their assertion ✓")

    no_syndebug = [float(r["qc_no_syndebug_depth"]) for r in rows]
    with_syndebug = [float(r["qc_with_syndebug_depth"]) for r in rows]
    percent = [float(r["percent_added"]) for r in rows]

    print(f"  avg depth without SynDebug : {mean(no_syndebug):.2f}")
    print(f"  avg depth with SynDebug    : {mean(with_syndebug):.2f}")
    print(f"  avg percent change         : {mean(percent):.2f}%")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the exhaustive SynDebug sweep results.")
    for _, flag, directory in RESULT_SETS:
        parser.add_argument(
            flag, metavar="CSV",
            help=f"CSV to analyze instead of the newest in {directory}",
        )
    args = parser.parse_args()

    all_passed = True
    load_failed = False
    for label, flag, directory in RESULT_SETS:
        try:
            path = getattr(args, flag.lstrip("-"))
            if path:
                note = f"given with {flag}"
            else:
                path, note = pick_csv(directory, flag)
            if not analyze(label, path, note):
                all_passed = False
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n=== {label} ===\n  !! {exc}")
            load_failed = True

    if load_failed:
        print("\nCould not load one or more result sets.")
    if not all_passed:
        print("\nOne or more result sets failed their assertion check.")
    return 1 if load_failed or not all_passed else 0


if __name__ == "__main__":
    sys.exit(main())
