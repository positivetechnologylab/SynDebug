"""Logical patch register for multi-patch CSS code circuits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import InstructionProperties
from qiskit.transpiler.passes import RemoveBarriers

import stim
from stim import Circuit, target_rec

from mqt.qecc.circuit_synthesis import heuristic_prep_circuit
from mqt.qecc.codes import CSSCode

rb = RemoveBarriers()


# Inverse of the gate map used by qiskit_to_stim_circuit in patch_register.
_STIM_TO_QISKIT_1Q = {
    "H": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "S_DAG": "sdg",
    "SQRT_X": "sx",
    "R": "reset",
}
_STIM_MEASUREMENTS = frozenset({"M", "MR"})
_STIM_X_MEASUREMENTS = frozenset({"MX"})
_STIM_DROPPED = frozenset({
    "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "MPAD",
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR",
})


def stim_to_qiskit_circuit(circuit: Circuit) -> QuantumCircuit:
    """Convert a Stim circuit to a Qiskit circuit.

    Supports the inverse of ``qiskit_to_stim_circuit``: H, X, Y, Z, S, S_DAG,
    SQRT_X, CX, R, M, MR, MX, and TICK (mapped to a Qiskit barrier on all qubits).
    MX is an X-basis measurement, emitted as H, measure, H.
    A ClassicalRegister sized to the total number of measurements is allocated
    automatically; measurements are bound to classical bits in encounter order.
    REPEAT blocks are unrolled inline. Annotations and noise channels are
    silently dropped.
    """
    num_qubits = circuit.num_qubits
    num_measurements = circuit.num_measurements
    qc = QuantumCircuit(num_qubits)
    if num_measurements:
        qc.add_register(ClassicalRegister(num_measurements, "c"))

    cbit_cursor = 0

    def emit(inst: stim.CircuitInstruction) -> None:
        nonlocal cbit_cursor
        name = inst.name
        if name in _STIM_DROPPED:
            return
        if name == "TICK":
            qc.barrier()
            return
        targets = inst.targets_copy()
        if name in _STIM_TO_QISKIT_1Q:
            method = getattr(qc, _STIM_TO_QISKIT_1Q[name])
            for t in targets:
                method(t.qubit_value)
            return
        if name == "CX":
            for i in range(0, len(targets), 2):
                qc.cx(targets[i].qubit_value, targets[i + 1].qubit_value)
            return
        if name in _STIM_MEASUREMENTS:
            for t in targets:
                qc.measure(t.qubit_value, cbit_cursor)
                cbit_cursor += 1
                if name == "MR":
                    qc.reset(t.qubit_value)
            return
        if name in _STIM_X_MEASUREMENTS:
            # Qiskit has no native X-basis measurement: rotate the X basis into
            # the Z basis with H, measure, then rotate back so the (non-demolition)
            # measurement leaves the qubit in an X eigenstate as stim's MX does.
            for t in targets:
                qc.h(t.qubit_value)
                qc.measure(t.qubit_value, cbit_cursor)
                qc.h(t.qubit_value)
                cbit_cursor += 1
            return
        raise ValueError(f"Unsupported stim operation: {name}")

    def walk(c: stim.Circuit) -> None:
        for item in c:
            if isinstance(item, stim.CircuitRepeatBlock):
                body = item.body_copy()
                for _ in range(item.repeat_count):
                    walk(body)
            else:
                emit(item)

    walk(circuit)
    return qc




class QubitPair(NamedTuple):
    control: int
    target: int


def _check_single_logical(code: CSSCode, context: str) -> None:
    k = code.n - code.Hx.shape[0] - code.Hz.shape[0]
    if k != 1:
        raise ValueError(f"{context}: code encodes k={k} logical qubits, expected k=1")

# from MQT.qecc
def qiskit_to_stim_circuit(qc: QuantumCircuit) -> Circuit:
    """Convert a Qiskit circuit to a Stim circuit."""
    single_qubit_gate_map = {
        "h": "H",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "s": "S",
        "sdg": "S_DAG",
        "sx": "SQRT_X",
        "measure": "MR",
        "reset": "R",
    }
    stim_circuit = Circuit()
    for gate in qc:
        op = gate.operation.name
        qubit = qc.find_bit(gate.qubits[0])[0]
        if op in single_qubit_gate_map:
            stim_circuit.append_operation(single_qubit_gate_map[op], [qubit])
        elif op == "cx":
            target = qc.find_bit(gate.qubits[1])[0]
            stim_circuit.append_operation("CX", [qubit, target])
        elif op == "barrier":
            stim_circuit.append("TICK")
        else:
            msg = f"Unsupported gate: {op}"
            raise ValueError(msg)
    return stim_circuit

@dataclass
class CodePatch:
    """A single code block with a fixed offset into the global qubit register."""

    code: CSSCode
    offset: int  # index of first data qubit in the global register

    def __post_init__(self) -> None:
        _check_single_logical(self.code, "CodePatch")

    @property
    def n(self) -> int:
        return self.code.n

    @property
    def ancilla_offset(self) -> int:
        return self.offset + self.n

    @property
    def num_ancilla(self) -> int:
        return self.code.Hx.shape[0] + self.code.Hz.shape[0]

    def data_qubits(self) -> list[int]:
        return list(range(self.offset, self.offset + self.n))

    def x_ancilla_qubits(self) -> list[int]:
        """Physical qubit indices for the X-stabilizer ancilla."""
        start = self.ancilla_offset
        return list(range(start, start + self.code.Hx.shape[0]))

    def z_ancilla_qubits(self) -> list[int]:
        """Physical qubit indices for the Z-stabilizer ancilla."""
        start = self.ancilla_offset + self.code.Hx.shape[0]
        return list(range(start, start + self.code.Hz.shape[0]))

    def logical_x(self) -> list[int]:
        """Physical qubit indices where the logical X operator acts non-trivially."""
        op = self.code.x_logicals_as_pauli_strings()[0]
        return [self.offset + i for i, p in enumerate(op) if p == "X"]

    def logical_z(self) -> list[int]:
        """Physical qubit indices where the logical Z operator acts non-trivially."""
        op = self.code.z_logicals_as_pauli_strings()[0]
        return [self.offset + i for i, p in enumerate(op) if p == "Z"]


@dataclass
class PatchRegister:
    """Ordered collection of k=1 code patches with contiguous physical qubit indices."""

    _patches: list[CodePatch] = field(default_factory=list)
    _next_offset: int = 0

    def add_patch(self, code: CSSCode) -> int:
        """Add a patch and return its index in the register."""
        _check_single_logical(code, "add_patch")
        patch = CodePatch(code=code, offset=self._next_offset)
        self._patches.append(patch)
        self._next_offset += code.n + code.Hx.shape[0] + code.Hz.shape[0]
        return len(self._patches) - 1

    def patch(self, idx: int) -> CodePatch:
        return self._patches[idx]

    def __len__(self) -> int:
        return len(self._patches)

    def total_qubits(self) -> int:
        return self._next_offset

    # ── State preparation ──────────────────────────────────────────────────────
    def prepare_logical_state(self, state_string: str) -> Circuit:
        """Return a circuit that prepares a multi-patch logical state.

        Args:
            state_string: One character per patch.
                '0' → |0>_L,  '1' → |1>_L (= X_L|0>_L),
                'p' → |+>_L,  'm' → |->_L (= Z_L|+>_L).
        """
        if len(state_string) != len(self._patches):
            raise ValueError(
                f"state_string length {len(state_string)} does not match "
                f"{len(self._patches)} patches"
            )
        invalid = set(state_string) - {"0", "1", "p", "m"}
        if invalid:
            raise ValueError(
                f"state_string contains invalid characters {sorted(invalid)}; "
                "allowed characters are '0', '1', 'p', 'm'"
            )
        circ = QuantumCircuit(self.total_qubits())
        for i, patch in enumerate(self._patches):
            ch = state_string[i]
            if ch == "0":
                zero_state, flip = True, False
            elif ch == "1":
                zero_state, flip = True, True
            elif ch == "p":
                zero_state, flip = False, False
            else:  # ch == "m"
                zero_state, flip = False, True
            sp = heuristic_prep_circuit(patch.code, zero_state=zero_state)
            qubits = list(range(patch.offset, patch.offset + patch.n))
            circ.compose(sp.circ, qubits=qubits, inplace=True)
            if flip:
                flip_qubits = patch.logical_x() if zero_state else patch.logical_z()
                gate = "x" if zero_state else "z"
                for q in flip_qubits:
                    getattr(circ, gate)(q)
        return qiskit_to_stim_circuit(circ)

    # ── Syndrome extraction ────────────────────────────────────────────────────
    def syndrome_round(
        self,
        *,
        first_round: bool = False,
        init_basis: str | None = None,
    ) -> Circuit:
        """Return one bare-ancilla syndrome-extraction round as a Stim circuit.

        Per patch, in register order, X stabilizers are measured first (H – CX
        ladder anc→data – H – M), then Z stabilizers (CX ladder data→anc – M).
        All ancillas are reset at the start of the round, so the same call can
        be composed repeatedly to build a multi-round circuit.

        Detector convention:
            first_round=False (default) — each stabilizer outcome is XORed with
                the same stabilizer's measurement from the immediately preceding
                round (offset n_meas records back).
            first_round=True  — only stabilizers deterministic under init_basis
                get a single-target detector. init_basis may be:
                    'Z'  — uniform: every patch prepared in |0>_L / |1>_L
                    'X'  — uniform: every patch prepared in |+>_L / |->_L
                    str of length len(self) — per-patch basis, one of 'Z'/'X'
                           per character (any other character → no detectors
                           for that patch in this round).
        """
        if first_round:
            if init_basis is None:
                raise ValueError("first_round=True requires init_basis")
            if len(init_basis) == 1:
                init_basis = init_basis * len(self._patches)
            elif len(init_basis) != len(self._patches):
                raise ValueError(
                    f"init_basis length {len(init_basis)} does not match "
                    f"{len(self._patches)} patches"
                )

        circ = Circuit()

        all_ancillas: list[int] = []
        for p in self._patches:
            all_ancillas.extend(p.x_ancilla_qubits())
            all_ancillas.extend(p.z_ancilla_qubits())
        if not all_ancillas:
            return circ
        circ.append("R", all_ancillas)
        circ.append("TICK")

        measure_order: list[int] = []
        for p in self._patches:
            x_ancs = p.x_ancilla_qubits()
            z_ancs = p.z_ancilla_qubits()
            for row_idx, row in enumerate(p.code.Hx):
                anc = x_ancs[row_idx]
                data = [p.offset + c for c, b in enumerate(row) if b]
                circ.append("H", [anc])
                for d in data:
                    circ.append("CX", [anc, d])
                circ.append("H", [anc])
                circ.append("M", [anc])
                measure_order.append(anc)
            for row_idx, row in enumerate(p.code.Hz):
                anc = z_ancs[row_idx]
                data = [p.offset + c for c, b in enumerate(row) if b]
                for d in data:
                    circ.append("CX", [d, anc])
                circ.append("M", [anc])
                measure_order.append(anc)

        n_meas = len(measure_order)
        if first_round:
            idx = 0
            for patch_i, p in enumerate(self._patches):
                basis = init_basis[patch_i]
                n_x = p.code.Hx.shape[0]
                n_z = p.code.Hz.shape[0]
                for _ in range(n_x):
                    if basis == "X":
                        circ.append("DETECTOR", [target_rec(idx - n_meas)])
                    idx += 1
                for _ in range(n_z):
                    if basis == "Z":
                        circ.append("DETECTOR", [target_rec(idx - n_meas)])
                    idx += 1
        else:
            for i in range(n_meas):
                circ.append(
                    "DETECTOR",
                    [target_rec(i - n_meas), target_rec(i - 2 * n_meas)],
                )

        return circ

    # ── Parallel stabilizer checks ─────────────────────────────────────────────
    def x_check_round(self) -> Circuit:
        """Measure every X stabilizer across all patches in parallel layers.

        All X ancillas (across every patch) are reset together, H'd together,
        and the CX ladder is emitted one column at a time: at step ``t``, every
        stabilizer in every patch contributes its ``t``-th data qubit to a
        single batched ``CX`` instruction. Patches use disjoint physical qubits,
        so all cross-patch CXs in a given step commute and execute in parallel.
        """
        x_ancillas: list[int] = []
        stabilizers: list[tuple[int, list[int]]] = []
        for p in self._patches:
            x_ancs = p.x_ancilla_qubits()
            for row_idx, row in enumerate(p.code.Hx):
                anc = x_ancs[row_idx]
                data = [p.offset + c for c, b in enumerate(row) if b]
                x_ancillas.append(anc)
                stabilizers.append((anc, data))

        circ = Circuit()
        if not x_ancillas:
            return circ

        circ.append("R", x_ancillas)
        circ.append("TICK")
        circ.append("H", x_ancillas)
        circ.append("TICK")

        max_w = max(len(d) for _, d in stabilizers)
        for t in range(max_w):
            layer: list[int] = []
            for anc, data in stabilizers:
                if t < len(data):
                    layer.extend([anc, data[t]])
            if layer:
                circ.append("CX", layer)
                circ.append("TICK")

        circ.append("H", x_ancillas)
        circ.append("TICK")
        circ.append("M", x_ancillas)
        return circ

    def z_check_round(self) -> Circuit:
        """Measure every Z stabilizer across all patches in parallel layers.

        Mirrors :meth:`x_check_round` but with the CX direction reversed
        (data → ancilla) and no surrounding Hadamards.
        """
        z_ancillas: list[int] = []
        stabilizers: list[tuple[int, list[int]]] = []
        for p in self._patches:
            z_ancs = p.z_ancilla_qubits()
            for row_idx, row in enumerate(p.code.Hz):
                anc = z_ancs[row_idx]
                data = [p.offset + c for c, b in enumerate(row) if b]
                z_ancillas.append(anc)
                stabilizers.append((anc, data))

        circ = Circuit()
        if not z_ancillas:
            return circ

        circ.append("R", z_ancillas)
        circ.append("TICK")

        max_w = max(len(d) for _, d in stabilizers)
        for t in range(max_w):
            layer: list[int] = []
            for anc, data in stabilizers:
                if t < len(data):
                    layer.extend([data[t], anc])
            if layer:
                circ.append("CX", layer)
                circ.append("TICK")

        circ.append("M", z_ancillas)
        return circ

    # ── Introspection ──────────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = [f"PatchRegister: {len(self._patches)} patches, "
                 f"{self.total_qubits()} total physical qubits"]
        for i, p in enumerate(self._patches):
            k = p.n - p.code.Hx.shape[0] - p.code.Hz.shape[0]
            anc_end = p.ancilla_offset + p.num_ancilla - 1
            lines.append(
                f"  [{i}]  data {p.offset}–{p.offset + p.n - 1}"
                f"  ancilla {p.ancilla_offset}–{anc_end}"
                f"  [[{p.n},{k},{p.code.distance}]]"
                f"  self-dual={p.code.is_self_dual()}"
            )
        return "\n".join(lines)


class QecProgram:
    """A logical program: a patch register initialised from a single code type and a prepared stim circuit."""

    def __init__(self, code: CSSCode, n_patches: int, state_string: str) -> None:
        self.register = PatchRegister()
        for _ in range(n_patches):
            self.register.add_patch(code)
        self.circuit: Circuit = self.register.prepare_logical_state(state_string)
        self.code = code
        self.assertion_count = 0
        # self.assertion_layer_count = 0
        # self.assertion_op_count = 0
        self.qec_extract_count = 0
        self.observable_count = 0
        # (start, width) into the measurement record for each logical measurement,
        # in the order they were appended; used by run() to reconstruct outcomes.
        self.logical_measurements: list[tuple[int, int]] = []
        self.assertion_types = []

    def compose(self, other: Circuit) -> None:
        self.circuit += other

    def append(self, name: str, target, arg=None) -> None:
        self.circuit.append(name, target, arg)

    def stabilizer_round(self) -> None:
        """Append one parallel syndrome-extraction round to the program.

        X stabilizers are measured first across every patch in parallel,
        followed by Z stabilizers in the same fashion.
        """
        self.compose(self.register.x_check_round())
        self.compose(self.register.z_check_round())
        self.qec_extract_count += 1
    
    def measure_logical(
        self,
        indices: list[int],
        basis: str | list[str],
    ) -> list[int]:
        """Measure logical-qubit observables and record them as Stim observables.

        For each logical-qubit index, the corresponding patch in ``self.register``
        is read to find the physical-qubit support of its logical Z (or X)
        operator. Those data qubits are measured in the requested basis ('M' for
        Z, 'MX' for X) and the resulting measurement records are tied together
        into a single ``OBSERVABLE_INCLUDE`` so Stim tracks each logical outcome
        directly.

        Args:
            indices: Logical-qubit (patch) indices to measure, in measurement
                order.
            basis: Either a single character ('Z' or 'X') applied to every qubit
                in ``indices``, or a sequence of the same length giving a
                per-qubit basis.

        Returns:
            The column of each measured qubit in the ``observables`` and
            ``errors`` arrays returned by :meth:`run`, in ``indices`` order.
        """
        if isinstance(basis, str) and len(basis) == 1:
            bases = [basis] * len(indices)
        else:
            bases = list(basis)
        if len(bases) != len(indices):
            raise ValueError(
                f"basis length {len(bases)} does not match {len(indices)} indices"
            )

        cols: list[int] = []
        for idx, b in zip(indices, bases):
            b = b.upper()
            patch = self.register.patch(idx)
            start = self.circuit.num_measurements
            if b == "Z":
                support = patch.logical_z()
                self.circuit.append("M", support)
            elif b == "X":
                support = patch.logical_x()
                self.circuit.append("MX", support)
            else:
                raise ValueError(f"basis must be 'Z' or 'X', got {b!r}")

            w = len(support)
            cols.append(len(self.logical_measurements))
            self.logical_measurements.append((start, w))
            self.circuit.append(
                "OBSERVABLE_INCLUDE",
                [target_rec(j - w) for j in range(w)],
                self.observable_count,
            )
            self.observable_count += 1
        return cols

    def list_assertions(self) -> None:
        print("Assertion count: ", self.assertion_count)
        print("Assertion types: ", self.assertion_types)

    def run(self, shots):
        """Sample the circuit, returning raw measurements, logical outcomes and errors.

        All three returned arrays come from the same sampling pass, so they are
        aligned shot-by-shot.

        Returns:
            A ``(samples, observables, errors)`` tuple.

            * ``samples`` — raw measurement-record array,
              shape ``(shots, n_measurements)``.
            * ``observables`` — absolute logical outcome (0/1) of each
              :meth:`measure_logical` call, shape
              ``(shots, len(self.logical_measurements))``, computed as the parity
              of the raw records over that logical operator's support.
            * ``errors`` — per-shot logical error flips, same shape as
              ``observables``: each outcome XORed with its noiseless reference
              value, so a ``True`` marks a shot where the logical qubit ended up
              flipped from its intended value. The mean over shots
              (``errors.mean(axis=0)``) is the per-logical-qubit error rate.

        Note:
            ``observables`` are the actual measured values; ``errors`` are the
            error-frame flips that ``OBSERVABLE_INCLUDE`` would report through a
            detector sampler (always 0 in a noiseless circuit).
        """
        samples = self.circuit.compile_sampler().sample(shots=shots)
        if self.logical_measurements:
            observables = np.column_stack([
                samples[:, start:start + width].sum(axis=1) % 2
                for start, width in self.logical_measurements
            ]).astype(bool)
            reference = np.asarray(self.circuit.reference_sample(), dtype=bool)
            ref_outcomes = np.array([
                bool(reference[start:start + width].sum() % 2)
                for start, width in self.logical_measurements
            ])
            errors = observables ^ ref_outcomes
        else:
            observables = np.empty((shots, 0), dtype=bool)
            errors = np.empty((shots, 0), dtype=bool)
        return samples, observables, errors

    
    def depth(self) -> int:

        num_qubits = self.circuit.num_qubits
        
        qc = rb(stim_to_qiskit_circuit(self.circuit))

        transpiled_qc = transpile(
            qc,
            basis_gates=["h", "x", "s", "cx", "z", "t"],
            initial_layout=list(range(num_qubits)),
            optimization_level=0,
        )
        
        return transpiled_qc.depth()
