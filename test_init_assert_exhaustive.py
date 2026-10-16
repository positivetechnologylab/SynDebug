"""Exhaustive QecProgram sweep over the k=1 codes available in mqt.qecc.

Rather than a hand-picked set of examples, this enumerates the full Cartesian
product of:

    code  ×  distance  ×  n_patches (1..MAX_PATCHES)  ×  state_string

where ``state_string`` ranges over *every* combination of the per-patch logical
states {'0', '1', 'p', 'm'} of length ``n_patches`` (i.e. 4**n_patches strings).

QecProgram requires single-logical-qubit (k=1) CSS codes, so codes such as
Carbon [[12,2,4]], Hamming [[15,7,3]], the iceberg [[2m,2,2]] family, and the
bivariate-bicycle LDPC codes (k>1) are not included here. The named codes
(Steane, Shor, Tetrahedral, Golay) are fixed-distance; the rotated surface code
and the two color-code families are swept over distances 3, 5 and 7.

Only logical-circuit depth (with and without the SynDebug assertion) is
recorded — the timing measurements from earlier revisions are dropped.
"""

import csv
import itertools
from datetime import datetime

from mqt.qecc.codes import (
    CSSCode,
    HexagonalColorCode,
    SquareOctagonColorCode,
)

from patch_register import QecProgram
from qec_assertions import assert_classical, logical_integer_from_measurements


# ── Sweep configuration ───────────────────────────────────────────────────────
MAX_PATCHES = 3                      # n_patches ranges over 1..MAX_PATCHES
STATE_CHARS = "01pm"                 # '0'->|0>, '1'->|1>, 'p'->|+>, 'm'->|->
PARAMETRIC_DISTANCES = [3, 5, 7]     # distances for the generated color codes
SURFACE_DISTANCES = [3, 5]           # mqt.qecc only ships surface_3 and surface_5

# Each spec is (display_name, builder(distance) -> CSSCode, distances).
# Named codes are fixed-distance and use a single [None] pass (the distance
# argument is ignored by from_code_name); the rest are swept over distances.
CODE_SPECS = [
    ("Steane",               lambda d: CSSCode.from_code_name("Steane", d),      [None]),
    ("Shor",                 lambda d: CSSCode.from_code_name("Shor", d),        [None]),
    ("Tetrahedral",          lambda d: CSSCode.from_code_name("Tetrahedral", d), [None]),
    ("Golay",                lambda d: CSSCode.from_code_name("Golay", d),       [None]),
    ("Surface",              lambda d: CSSCode.from_code_name("Surface", d),     SURFACE_DISTANCES),
    ("Hexagonal Color",      lambda d: HexagonalColorCode(d),                    PARAMETRIC_DISTANCES),
    ("Square Octagon Color", lambda d: SquareOctagonColorCode(d),               PARAMETRIC_DISTANCES),
]


def describe(label: str, program: QecProgram, state_string: str) -> None:
    p0 = program.register.patch(0)
    print(f"\n--- {label} ---")
    print(f"  state_string : {state_string!r}")
    print(f"  patches      : {len(program.register)}")
    print(f"  total qubits : {program.register.total_qubits()}")
    print(f"  per-patch [[n,k,d]] : [[{p0.n},1,{p0.code.distance}]]")
    print(f"  prep-circuit ops    : {len(program.circuit)}")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"results/initialization/exhaustive/exhaustive_classical_results{timestamp}.csv"
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "name", "dist", "n_patches", "state_string",
    "qc_no_syndebug_depth", "qc_with_syndebug_depth", "percent_added",
    "assertion_passed",
])

# ── Exhaustive sweep ──────────────────────────────────────────────────────────
for name, builder, distances in CODE_SPECS:
    for dist in distances:
        # Build the code once per (name, distance); QecProgram replicates it
        # across patches and the object is only read, never mutated.
        code = builder(dist)
        actual_dist = code.distance

        for n_patches in range(1, MAX_PATCHES + 1):
            for state_tuple in itertools.product(STATE_CHARS, repeat=n_patches):
                state_string = "".join(state_tuple)

                label = f"{name} d={actual_dist} n={n_patches} {state_string}"
                try:
                    program = QecProgram(
                        code=code,
                        n_patches=n_patches,
                        state_string=state_string,
                    )

                    describe(label, program, state_string)

                    # add stabilizer round
                    program.stabilizer_round()

                    # depth before the assertion
                    qc_no_syndebug_depth = program.depth()

                    basis = "".join("Z" if s in ("0", "1") else "X" for s in state_string)
                    bool_in_given_basis = [s in ("1", "m") for s in state_string]
                    value = logical_integer_from_measurements([bool_in_given_basis])[0]

                    # raise_on_fail=False so a single statistical failure records a
                    # result and the sweep keeps going instead of aborting.
                    assertion_bool = assert_classical(
                        program,
                        list(range(n_patches)),
                        value=value,
                        n_shots=10,
                        basis=basis,
                        raise_on_fail=False,
                    )

                    # depth again after the assertion
                    qc_with_syndebug_depth = program.depth()

                    percent_of_total_prog = (
                        (qc_with_syndebug_depth - qc_no_syndebug_depth)
                        / qc_with_syndebug_depth * 100
                    )

                    print("Basis: ", basis)
                    print("Value: ", value)
                    print("Depth without SynDebug: ", qc_no_syndebug_depth)
                    print("Depth with SynDebug   : ", qc_with_syndebug_depth)
                    print("Percentage of total: ", percent_of_total_prog, "%")
                    print("Assertion passed: ", assertion_bool)

                    csv_writer.writerow([
                        name, actual_dist, n_patches, state_string,
                        qc_no_syndebug_depth, qc_with_syndebug_depth,
                        percent_of_total_prog, assertion_bool,
                    ])
                    csv_file.flush()
                except Exception as exc:  # keep the sweep alive on a bad case
                    print(f"!! SKIPPED {label}: {type(exc).__name__}: {exc}")


csv_file.close()
