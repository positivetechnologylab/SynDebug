from patch_register import QecProgram
from stim import Circuit

def transversal_cnot(program: QecProgram, control: int, target: int) -> None:
    """Append a transversal CNOT between two patches to ``program``'s circuit.

    CX is applied from each data qubit of the control patch to the
    corresponding data qubit of the target patch.
    """
    ctrl_patch = program.register.patch(control)
    tgt_patch = program.register.patch(target)
    if ctrl_patch.n != tgt_patch.n:
        raise ValueError(
            f"Transversal CNOT requires equal-size patches, "
            f"got n={ctrl_patch.n} (control) and n={tgt_patch.n} (target)"
        )
    circ = Circuit()
    for c, t in zip(ctrl_patch.data_qubits(), tgt_patch.data_qubits()):
        circ.append("CX", [c, t])
    program.compose(circ)


def logical_swap(program: QecProgram, a: int, b: int) -> None:
    """Swap the logical states of patches ``a`` and ``b``.

    Implemented as the standard three-CNOT swap, each CNOT being transversal:
    CX(a,b) · CX(b,a) · CX(a,b).
    """
    transversal_cnot(program, a, b)
    transversal_cnot(program, b, a)
    transversal_cnot(program, a, b)


def logical_x(program: QecProgram, patch_index: int) -> None:
    """Apply a logical X to a patch by acting on its logical-X operator support."""
    patch = program.register.patch(patch_index)
    circ = Circuit()
    circ.append("X", patch.logical_x())
    program.compose(circ)