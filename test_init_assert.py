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
from qec_assertions import assert_classical, logical_integer_from_measurements

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import transpile
from qiskit.transpiler import InstructionProperties
from qiskit.transpiler.passes import RemoveBarriers

import csv
from datetime import datetime
import time

import numpy as np

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
    ("Steane [[7,1,3]] — |0>_L",            "Steane",      None, 1, "0"),
    ("Steane [[7,1,3]] — |1>_L|->_L pair",  "Steane",      None, 2, "10"),
    ("Shor [[9,1,3]] — |1>_L",              "Shor",        None, 1, "1"),
    ("Shor [[9,1,3]] — |+>|0>|0> GHZ seed", "Shor",        None, 3, "100"),
    ("Tetrahedral [[15,1,3]] — |0>_L",      "Tetrahedral", None, 1, "0"),
    ("Golay [[23,1,7]] — |->_L",            "Golay",       None, 1, "m"),
]

# ── Rotated surface codes across distances and patch counts ───────────────────
surface_examples = [
    ("Surface", "Surface", 3, 1, "0"),
    ("Surface", "Surface", 3, 2, "p0"),  # mixed-basis pair (lattice-surgery seed)
    ("Surface", "Surface", 5, 1, "0"),
    ("Surface", "Surface", 5, 2, "pm"),
]

# ── Color codes (hexagonal 6.6.6 and square-octagon 4.8.8) ────────────────────
color_examples = [
    (HexagonalColorCode,  "Hexagonal Color",   3, 1, "0"),
    (HexagonalColorCode,  "Hexagonal Color",   5, 1, "1"),
    (HexagonalColorCode,  "Hexagonal Color",   7, 1, "0"),
    (HexagonalColorCode,  "Hexagonal Color",   3, 3, "0p1"),
    (SquareOctagonColorCode, "Square Octagon Color", 3, 1, "0"),
    (SquareOctagonColorCode, "Square Octagon Color", 5, 1, "p"),
    (SquareOctagonColorCode, "Square Octagon Color", 3, 2, "pm"),
]

combined_examples = named_examples + surface_examples + color_examples

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"results/initialization/classical_results_{timestamp}.csv"
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "name", "dist", "state_string",
    "qc_no_syndebug_depth", "qc_with_syndebug_depth", "percent_added"
])

csv_path_timings = f"results/timings/classical_timing_{timestamp}.csv"
csv_file_timings = open(csv_path_timings, "w", newline="")
csv_writer_timings = csv.writer(csv_file_timings)

csv_writer_timings.writerow([
    "name", "dist", "state_string",
    "qc_no_syndebug_time", "qc_no_syndebug_time_std", "qc_with_syndebug_time", "qc_with_syndebug_time_std"
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

    # add stabilizer round
    program.stabilizer_round()

    timings_list = []

    for _ in range(10):
    # measure execution time
        start_time = time.perf_counter()
        # Code block to measure
        program.run(shots=10000)
        # End the timer
        end_time = time.perf_counter()

        timings_list.append(end_time - start_time)
    
    qc_no_syndebug_time = np.mean(timings_list)
    qc_no_syndebug_time_std = np.std(timings_list)

    # first measure depth before assertion
    qc_no_syndebug_depth = program.depth()
    # print(program.circuit.diagram("timeline-text"))

    basis = "".join(["Z" if s in ["0", "1"] else "X" for s in state_string])
    bool_in_given_basis = [s in ["1", "m"] for s in state_string]
    value = logical_integer_from_measurements([bool_in_given_basis])[0]

    assertion_bool = assert_classical(program, [i for i in range(n_patches)], value=value, n_shots=10, basis=basis)
    
    timings_list = []
    for _ in range(10):
        # measure execution time
        start_time = time.perf_counter()
        # Code block to measure
        program.run(shots=10000)
        # End the timer
        end_time = time.perf_counter()

        timings_list.append(end_time - start_time)

    qc_with_syndebug_time = np.median(timings_list)
    qc_with_syndebug_time_std = np.std(timings_list)

    # measure depth again after assertion
    qc_with_syndebug_depth = program.depth()
    

    percent_of_total_prog = (qc_with_syndebug_depth - qc_no_syndebug_depth) / qc_with_syndebug_depth * 100

    print("Basis: ", basis)
    print("Value: ", value)
    print("Depth without SynDebug: ", qc_no_syndebug_depth)
    print("Depth with SynDebug   : ", qc_with_syndebug_depth)
    print("Sim time without SynDebug: ", qc_no_syndebug_time)
    print("Sim time with SynDebug   : ", qc_with_syndebug_time)

    print("Percentage of total: ", percent_of_total_prog, "%")
    print("Assertion passed: ", assertion_bool)
    # print(program.circuit.diagram("timeline-text"))

    csv_writer.writerow([
    name, dist, state_string,
    qc_no_syndebug_depth, qc_with_syndebug_depth, percent_of_total_prog
])
    
    csv_writer_timings.writerow([
    name, dist, state_string, qc_no_syndebug_time, qc_no_syndebug_time_std, qc_with_syndebug_time, qc_with_syndebug_time_std
])



csv_file.close()
csv_file_timings.close()