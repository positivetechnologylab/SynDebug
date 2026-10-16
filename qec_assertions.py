"""
QEC assertion utilities.
"""
import dataclasses
import numpy as np
import scipy.stats
from typing import List
from stim import Circuit

from patch_register import QecProgram, CodePatch

def logical_integer_from_measurements(
    logical_measurements: np.ndarray,
) -> np.ndarray:
    """
    Convert per-shot logical measurement bits into integers.

    Each row of *logical_measurements* is a binary string of logical
    outcomes; this returns the integer it encodes, using the MSB-first
    convention (column 0 is the most-significant bit)

    Args:
        logical_measurements: Boolean array of shape (n_shots, n_bits).

    Returns:
        Integer array of shape (n_shots,), one value per row.
    """
    logical_measurements = np.asarray(logical_measurements)
    powers = 1 << np.arange(logical_measurements.shape[1] - 1, -1, -1)
    return logical_measurements @ powers

@dataclasses.dataclass
class Chi2Result:
    statistic: float
    p_value: float
    dof: int
    expected: np.ndarray

    def __str__(self) -> str:
        verdict = "independent" if self.p_value > 0.05 else "NOT independent"
        return (
            f"Chi2={self.statistic:.4f}  dof={self.dof}  "
            f"p={self.p_value:.4f}  -> {verdict} (α=0.05)"
        )


def chi2_independence_test(table: np.ndarray) -> Chi2Result:
    """
    Test whether the row and column variables in a contingency table are
    independent using Pearson's chi-squared test.

    Works on any 2D contingency table, including the general m×n tables
    produced by subset_contingency_table().

    Parameters
    ----------
    table : int array of shape (n_rows, n_cols) from contingency_table()
            or subset_contingency_table()

    Returns
    -------
    Chi2Result with fields:
        statistic : chi-squared test statistic
        p_value   : p-value (small -> evidence against independence)
        dof       : degrees of freedom  (n_rows-1) * (n_cols-1)
        expected  : expected counts array under the independence hypothesis

    Notes
    -----
    Cells with zero expected count are excluded from the statistic to avoid
    division by zero; scipy.stats.chi2_contingency handles this automatically
    via the lambda_ parameter.  Rows or columns that are entirely zero (an
    outcome that never occurred) are dropped before the test, since they carry
    no information and reduce degrees of freedom artificially.
    """
    # Drop all-zero rows and columns — they collapse degrees of freedom
    # without contributing any information.
    mask_rows = table.sum(axis=1) > 0
    mask_cols = table.sum(axis=0) > 0
    trimmed = table[np.ix_(mask_rows, mask_cols)]

    if trimmed.shape[0] < 2 or trimmed.shape[1] < 2:
        raise ValueError(
            "After removing all-zero rows/columns fewer than 2 remain on at "
            "least one axis — the test cannot be performed (e.g. a deterministic "
            "state has no variation to test)."
        )

    raw = tuple(scipy.stats.chi2_contingency(trimmed, correction=False))
    return Chi2Result(
        statistic=float(raw[0]),
        p_value=float(raw[1]),
        dof=int(raw[2]),
        expected=np.asarray(raw[3]),
    )


def subset_contingency_table(
    samples: np.ndarray,
    qubits_a: list[int],
    qubits_b: list[int],
) -> np.ndarray:
    """
    Build a contingency table comparing two subsets of qubits.

    Each row corresponds to one of the 2^|qubits_a| joint outcomes for subset A,
    and each column corresponds to one of the 2^|qubits_b| joint outcomes for
    subset B.  Outcome strings follow qubit order as given (leftmost qubit is
    the most-significant bit).

    Parameters
    ----------
    samples  : boolean array of shape (n_shots, n_qubits) from stim's sampler
    qubits_a : qubit indices whose joint outcome labels the rows
    qubits_b : qubit indices whose joint outcome labels the columns

    Returns
    -------
    table : int array of shape (2**len(qubits_a), 2**len(qubits_b))
        table[i, j] = number of shots where subset A produced outcome i
                      and subset B produced outcome j, with i and j
                      interpreted as big-endian binary integers.
    """
    if not qubits_a or not qubits_b:
        raise ValueError("qubits_a and qubits_b must each contain at least one index")
    if set(qubits_a) & set(qubits_b):
        raise ValueError("qubits_a and qubits_b must be disjoint")

    n_rows = 2 ** len(qubits_a)
    n_cols = 2 ** len(qubits_b)

    # Convert each subset's per-shot bits to an integer index (little-endian:
    # the first qubit in the list is the least significant bit)
    powers_a = 1 << np.arange(len(qubits_a))  # [1, 2, 4, ...]
    powers_b = 1 << np.arange(len(qubits_b))

    row_idx = samples[:, qubits_a].astype(int) @ powers_a   # shape (n_shots,)
    col_idx = samples[:, qubits_b].astype(int) @ powers_b

    flat = row_idx * n_cols + col_idx
    counts = np.bincount(flat, minlength=n_rows * n_cols)
    return counts.reshape(n_rows, n_cols)



def assert_classical(
    prog: QecProgram,
    logic_qubit_indices: List[int],
    value: int,
    n_shots: int,
    *,
    basis: str = 'Z',
    p0: float = 1.0,
    alpha: float = 0.05,
    raise_on_fail: bool = True,
    verbose: bool = False,
) -> bool:
    
    cols = prog.measure_logical(logic_qubit_indices, basis)
    _, logical_measurements, _ = prog.run(n_shots)

    outcomes = logical_integer_from_measurements(logical_measurements[:, cols])

    count_match = int(np.sum(outcomes == value))

    p_value = float(scipy.stats.binomtest(count_match, n_shots, p=p0, alternative='less').pvalue)
    passed = p_value >= alpha

    if verbose:
        print(
            f"assert_logical_classical  logical_qubits={logic_qubit_indices}  "
            f"basis={basis}  d={prog.code.distance}:  "
            f"value={value}  count={count_match}/{n_shots}  "
            f"p_hat={count_match / n_shots:.4f}  p0={p0}  "
            f"binom_p={p_value:.4f}  alpha={alpha}  "
            f"-> {'PASS' if passed else 'FAIL'}"
        )
    if not passed and raise_on_fail:
        raise AssertionError(
            f"assert_logical_classical failed for logical qubit "
            f"{logic_qubit_indices} (d={prog.code.distance}, basis={basis}): "
            f"expected logical {value}, got {count_match}/{n_shots} shots "
            f"matching (p_hat={count_match / n_shots:.4f}), significantly "
            f"less than p0={p0} (binomial p={p_value:.4f}, alpha={alpha})"
        )
    
    # update assertion count for logging resource usage
    prog.assertion_count += 1
    prog.assertion_types.append("assert_classical")

    return passed


def assert_superposition(
    prog: QecProgram,
    logic_qubit_indices: List[int],
    n_shots: int,
    *,
    basis: str = 'Z',
    p0: float = 1.0,
    alpha: float = 0.05,
    raise_on_fail: bool = True,
    verbose: bool = False,
) -> bool:

    cols = prog.measure_logical(logic_qubit_indices, basis)
    _, logical_measurements, _ = prog.run(n_shots)

    outcomes = logical_integer_from_measurements(logical_measurements[:, cols])

    unique_outcomes, unique_counts = np.unique_counts(outcomes)
    print("Unique outcomes : ", unique_outcomes)
    print("Unique counts : ", unique_counts)
    
    o_max = unique_outcomes[np.argmax(unique_counts)]
    count_match = unique_counts[np.argmax(unique_counts)]

    p_value = float(scipy.stats.binomtest(count_match, n_shots, p=p0, alternative='less').pvalue)
    passed = p_value < alpha

    if verbose:
        print(
            f"assert_logical_superposition  logical_qubit={logic_qubit_indices}  "
            f"basis={basis}  d={prog.code.distance}:  "
            f"O_max={o_max}  count={count_match}/{n_shots}  "
            f"p_hat={count_match / n_shots:.4f}  p0={p0}  "
            f"binom_p={p_value:.4f}  alpha={alpha}  "
            f"-> {'PASS' if passed else 'FAIL'}"
        )
    if not passed and raise_on_fail:
        raise AssertionError(
            f"assert_logical_superposition failed for logical qubit "
            f"{logic_qubit_indices} (d={prog.code.distance}, basis={basis}): "
            f"most probable outcome {o_max} observed {count_match}/{n_shots} times "
            f"(p_hat={count_match / n_shots:.4f}), not significantly less than "
            f"p0={p0} (binomial p={p_value:.4f}, alpha={alpha})"
        )
    
    prog.assertion_count += 1
    prog.assertion_types.append("assert_superposition")

    return passed


def assert_entanglement(
    prog: QecProgram,
    subset_a: List[int],
    subset_b: List[int],
    n_shots: int,
    *,
    basis: str = 'Z',
    alpha: float = 0.05,
    raise_on_fail: bool = True,
    verbose: bool = False
) -> bool:
    """
    Assert that two subsets of logical qubits are entangled.

    Measures the logical qubits in *subset_a* and *subset_b*, builds a
    contingency table cross-tabulating the joint outcomes of the two subsets,
    then performs Pearson's chi-squared test of independence:

        H₀: subset_a and subset_b are independent   (not entangled)
        H₁: subset_a and subset_b are correlated     (entangled)

    Entanglement manifests as a correlation between the two subsets, so the
    assertion passes when H₀ is rejected (p_value < alpha).

    If the contingency table is degenerate — one subset shows no variation, so
    fewer than two rows or columns survive — the independence test cannot be
    performed (chi2_independence_test raises ValueError).  This is treated as a
    failure: no correlation can be detected.  Note this also occurs when an
    entangled state is measured in a basis where the correlation collapses, so
    a failure here may indicate the wrong measurement basis rather than a
    genuinely unentangled state.

    Args:
        prog:          The QecProgram under test.
        subset_a:      Logical qubit indices whose joint outcome labels the
                       contingency table rows.
        subset_b:      Logical qubit indices whose joint outcome labels the
                       contingency table columns; must be disjoint from
                       *subset_a*.
        n_shots:       Number of shots to sample.
        basis:         Logical observable basis: 'Z' (default) or 'X'.
        alpha:         Significance level (default 0.05).
        raise_on_fail: If True (default), raise AssertionError on failure;
                       if False, return False instead.
        verbose:       If True, print test statistics before returning.

    Returns:
        True if the assertion passes, False if it fails and raise_on_fail
        is False.
    """
    # Checked on patch indices before measuring: once mapped to measurement
    # columns the subsets never overlap, so subset_contingency_table can't
    # catch it.
    if set(subset_a) & set(subset_b):
        raise ValueError("subset_a and subset_b must be disjoint")

    logic_qubit_indices = subset_a + subset_b
    cols = prog.measure_logical(logic_qubit_indices, basis)
    _, logical_measurements, _ = prog.run(n_shots)

    n_a = len(subset_a)
    table = subset_contingency_table(logical_measurements, cols[:n_a], cols[n_a:])

    try:
        result = chi2_independence_test(table)
        passed = result.p_value < alpha
        detail = (
            f"Chi2={result.statistic:.4f}  dof={result.dof}  "
            f"p={result.p_value:.4f}"
        )
    except ValueError as exc:
        # Degenerate table: a subset has no variation, so independence cannot
        # be tested. No correlation is detectable -> treat as a failure.
        result = None
        passed = False
        detail = f"degenerate contingency table ({exc})"

    if verbose:
        print(
            f"assert_entanglement  subset_a={subset_a}  subset_b={subset_b}  "
            f"basis={basis}  d={prog.code.distance}:  "
            f"{detail}  alpha={alpha}  "
            f"-> {'PASS' if passed else 'FAIL'}"
        )
    if not passed and raise_on_fail:
        raise AssertionError(
            f"assert_entanglement failed for subset_a={subset_a}, "
            f"subset_b={subset_b} (d={prog.code.distance}, basis={basis}): "
            f"subsets are not significantly correlated "
            f"({detail}, alpha={alpha})"
        )

    prog.assertion_count += 1
    prog.assertion_types.append("assert_entanglement")

    return passed

