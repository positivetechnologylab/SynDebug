"""Examples of QecPrograms built from the k=1 codes available in mqt.qecc.

QecProgram requires single-logical-qubit (k=1) CSS codes, so codes such as
Carbon [[12,2,4]], Hamming [[15,7,3]], the iceberg [[2m,2,2]] family, and the
bivariate-bicycle LDPC codes (k>1) are skipped here.
"""

from mqt.qecc.codes import (
    CSSCode,
    HexagonalColorCode,
    SquareOctagonColorCode,
)

from patch_register import QecProgram
from qec_assertions import assert_entanglement, logical_integer_from_measurements

from logical_gadgets import transversal_cnot

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import transpile
from qiskit.transpiler import InstructionProperties
from qiskit.transpiler.passes import RemoveBarriers

import csv
from datetime import datetime

rb = RemoveBarriers()

def describe(label: str, program: QecProgram, state_string: str) -> None:
    p0 = program.register.patch(0)
    print(f"\n--- {label} ---")
    print(f"  state_string : {state_string!r}")
    print(f"  patches      : {len(program.register)}")
    print(f"  total qubits : {program.register.total_qubits()}")
    print(f"  per-patch [[n,k,d]] : [[{p0.n},1,{p0.code.distance}]]")
    print(f"  prep-circuit ops    : {len(program.circuit)}")


# ── Named CSS codes via CSSCode.from_code_name ────────────────────────────────
# (label, code-name, distance argument, n_patches, state_string)
named_examples = [
    ("Steane [[7,1,3]] — |0>_L",            "Steane",      None, 2, "p0"),
    ("Steane [[7,1,3]] — |1>_L|->_L pair",  "Steane",      None, 2, "m0"),
    ("Shor [[9,1,3]] — |1>_L",              "Shor",        None, 2, "p0"),
    ("Shor [[9,1,3]] — |+>|0>|0> GHZ seed", "Shor",        None, 2, "p1"),
    ("Tetrahedral [[15,1,3]] — |0>_L",      "Tetrahedral", None, 2, "p0"),
    ("Golay [[23,1,7]] — |->_L",            "Golay",       None, 2, "m1"),
]

# ── Rotated surface codes across distances and patch counts ───────────────────
surface_examples = [
    ("Surface", "Surface", 3, 2, "p0"),
    ("Surface", "Surface", 3, 2, "m0"),
    ("Surface", "Surface", 5, 2, "p0"),
    ("Surface", "Surface", 5, 2, "m1"),
]

# ── Color codes (hexagonal 6.6.6 and square-octagon 4.8.8) ────────────────────
color_examples = [
    (HexagonalColorCode,  "Hexagonal Color",   3, 2, "p0"),
    (HexagonalColorCode,  "Hexagonal Color",   5, 2, "p1"),
    (HexagonalColorCode,  "Hexagonal Color",   7, 2, "m0"),
    (HexagonalColorCode,  "Hexagonal Color",   3, 2, "p0"),
    (SquareOctagonColorCode, "Square Octagon Color", 3, 2, "p0"),
    (SquareOctagonColorCode, "Square Octagon Color", 5, 2, "p0"),
    (SquareOctagonColorCode, "Square Octagon Color", 3, 2, "p1"),
]

combined_examples = named_examples + surface_examples + color_examples

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"results/entanglement/entanglement_results_{timestamp}.csv"
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "name", "dist", "state_string",
    "qc_no_syndebug_depth", "qc_with_syndebug_depth", "percent_added"
])

# iterate through various codes
for label, name, dist, n_patches, state_string in combined_examples:

    if name == "Hexagonal Color" or name == "Square Octagon Color":
            program = QecProgram(
            code=label(dist),
            n_patches=n_patches,
            state_string=state_string,
        )
    else:
        program = QecProgram(
            code=CSSCode.from_code_name(name, dist),
            n_patches=n_patches,
            state_string=state_string,
        )
    
    describe(label, program, state_string)
    num_qubits = program.circuit.num_qubits

    program.stabilizer_round()

    # add entangling operation
    transversal_cnot(program, 0, 1)

    program.stabilizer_round()

    # first measure depth before assertion
    qc_no_syndebug_depth = program.depth()

    assertion_bool = assert_entanglement(program, subset_a=[0], subset_b=[1], n_shots=10)
    
    # measure depth again after assertion
    qc_with_syndebug_depth = program.depth()

    percent_of_total_prog = (qc_with_syndebug_depth - qc_no_syndebug_depth) / qc_with_syndebug_depth * 100

    print("Depth without SynDebug: ", qc_no_syndebug_depth)
    print("Depth with SynDebug   : ", qc_with_syndebug_depth)
    print("Percentage of total: ", percent_of_total_prog, "%")
    print("Assertion passed: ", assertion_bool)
    
    csv_writer.writerow([
    name, dist, state_string,
    qc_no_syndebug_depth, qc_with_syndebug_depth, percent_of_total_prog
])



csv_file.close()