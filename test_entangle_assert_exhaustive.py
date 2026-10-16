"""Exhaustive entanglement-assertion sweep over the k=1 codes in mqt.qecc.

For each code and distance, two patches are prepared in one of the four
Bell-pair seed states, a transversal CNOT entangles them, and an entanglement
assertion is applied. The four seed strings

    'p0', 'p1', 'm0', 'm1'

are the control/target combinations whose transversal CNOT produces the four
Bell pairs, so this is the entangling analogue of the (single-patch) classical
init sweep: only the 2-patch case is meaningful here.

QecProgram requires single-logical-qubit (k=1) CSS codes, so codes such as
Carbon [[12,2,4]], Hamming [[15,7,3]], the iceberg [[2m,2,2]] family, and the
bivariate-bicycle LDPC codes (k>1) are not included here. The named codes
(Steane, Shor, Tetrahedral, Golay) are fixed-distance; the color-code families
are swept over distances 3, 5 and 7, and the rotated surface code over 3 and 5
(mqt.qecc only ships surface_3 and surface_5).

Only logical-circuit depth (with and without the SynDebug assertion) is recorded.
"""

import csv
from datetime import datetime

from mqt.qecc.codes import (
    CSSCode,
    HexagonalColorCode,
    SquareOctagonColorCode,
)

from patch_register import QecProgram
from qec_assertions import assert_entanglement

from logical_gadgets import transversal_cnot


# ── Sweep configuration ───────────────────────────────────────────────────────
N_PATCHES = 2                        # entanglement assertions act on a patch pair
# Control/target seed states whose transversal CNOT yields the four Bell pairs.
BELL_STATE_STRINGS = ["p0", "p1", "m0", "m1"]
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
csv_path = f"results/entanglement/exhaustive/exhaustive_entanglement_results{timestamp}.csv"
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

        for state_string in BELL_STATE_STRINGS:
            label = f"{name} d={actual_dist} {state_string}"
            try:
                program = QecProgram(
                    code=code,
                    n_patches=N_PATCHES,
                    state_string=state_string,
                )

                describe(label, program, state_string)

                program.stabilizer_round()

                # entangle the two patches
                transversal_cnot(program, 0, 1)

                program.stabilizer_round()

                # depth before the assertion
                qc_no_syndebug_depth = program.depth()

                # raise_on_fail=False so a single statistical failure records a
                # result and the sweep keeps going instead of aborting.
                assertion_bool = assert_entanglement(
                    program,
                    subset_a=[0],
                    subset_b=[1],
                    n_shots=10,
                    raise_on_fail=False,
                )

                # depth again after the assertion
                qc_with_syndebug_depth = program.depth()

                percent_of_total_prog = (
                    (qc_with_syndebug_depth - qc_no_syndebug_depth)
                    / qc_with_syndebug_depth * 100
                )

                print("Depth without SynDebug: ", qc_no_syndebug_depth)
                print("Depth with SynDebug   : ", qc_with_syndebug_depth)
                print("Percentage of total: ", percent_of_total_prog, "%")
                print("Assertion passed: ", assertion_bool)

                csv_writer.writerow([
                    name, actual_dist, 2, state_string,
                    qc_no_syndebug_depth, qc_with_syndebug_depth,
                    percent_of_total_prog, assertion_bool,
                ])
                csv_file.flush()
            except Exception as exc:  # keep the sweep alive on a bad case
                print(f"!! SKIPPED {label}: {type(exc).__name__}: {exc}")


csv_file.close()
