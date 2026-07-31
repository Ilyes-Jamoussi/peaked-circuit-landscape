"""Surjectivity of the fifteen-rotation gate word onto SU(4).

The first-rung theorem states delta*(tau_p = 1) <= the geometric measure of
entanglement of the scrambled state with respect to the brick partition, with
EQUALITY when the gate map theta -> G(theta) is onto SU(4). The inequality is
unconditional; the equality is a hypothesis, and the paper calls it "verified
numerically for the fixed word order". This script is that verification.

Two claims are tested, in increasing strength:

  (a) State reachability -- what the theorem actually needs. G(theta)^dagger |00>
      must cover the unit sphere of C^4 up to phase, so that the variational
      image at tau_p = 1 is the whole set of block-product states.
  (b) Full surjectivity onto SU(4) -- the claim as written in the text.

Both are decided by direct optimization against Haar-random targets: the gate
map is a product of fifteen one-parameter Pauli rotations in a FIXED word order,
and a product of one-parameter subgroups covering a 15-dimensional group is not
surjective by any general argument, so this has to be measured.

Self-tests: G(0) = identity, det G = 1, and the reported residuals are invariant
under the global phase that cancels in delta.

Usage:
    python analysis/gate_surjectivity.py        # ~1 min
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import unitary_group

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pennylane as qml  # noqa: E402
import pq_experiment  # noqa: E402,F401  (wires up the reproduction repo path)
from peaked_circuits import PARAMETERS_PER_SU4_GATE  # noqa: E402

NUM_STATE_TARGETS = 12
NUM_GATE_TARGETS = 8
RESTARTS_STATE = 4
RESTARTS_GATE = 6
BASE_SEED = 5150
TOLERANCE = 1e-8


def gate(theta: np.ndarray) -> np.ndarray:
    """The fixed fifteen-rotation two-qubit word, as a 4x4 matrix."""
    return np.array(
        qml.matrix(qml.ArbitraryUnitary(np.asarray(theta), wires=[0, 1]))
    )


def state_residual(theta: np.ndarray, target: np.ndarray) -> float:
    """1 - |<target | G(theta)^dagger 00>|^2; zero iff the state is reached."""
    reached = gate(theta).conj().T[:, 0]
    return 1.0 - abs(np.vdot(target, reached)) ** 2


def gate_residual(theta: np.ndarray, target: np.ndarray) -> float:
    """1 - |Tr(G(theta)^dagger U)/4|^2; zero iff G = U up to a global phase."""
    overlap = np.trace(gate(theta).conj().T @ target) / 4.0
    return 1.0 - abs(overlap) ** 2


def best_residual(objective, target, restarts: int, rng) -> float:
    best = 1.0
    for _ in range(restarts):
        start = rng.normal(0.0, 1.0, PARAMETERS_PER_SU4_GATE)
        result = minimize(objective, start, args=(target,), method="BFGS",
                          options={"maxiter": 4000})
        best = min(best, float(result.fun))
    return best


def self_tests() -> None:
    rng = np.random.default_rng(BASE_SEED)
    assert np.allclose(gate(np.zeros(PARAMETERS_PER_SU4_GATE)), np.eye(4)), (
        "the word must be the identity at theta = 0"
    )
    theta = rng.normal(size=PARAMETERS_PER_SU4_GATE)
    determinant = np.linalg.det(gate(theta))
    assert abs(determinant - 1.0) < 1e-9, f"det G = {determinant}, expected 1"

    # Both residuals must ignore the global phase, which cancels in delta.
    target = unitary_group.rvs(4, random_state=7)
    target /= np.linalg.det(target) ** 0.25
    phase = np.exp(1j * 0.7)
    assert abs(gate_residual(theta, target)
               - gate_residual(theta, phase * target)) < 1e-12
    vector = rng.normal(size=4) + 1j * rng.normal(size=4)
    vector /= np.linalg.norm(vector)
    assert abs(state_residual(theta, vector)
               - state_residual(theta, phase * vector)) < 1e-12
    print("self-tests passed (identity at zero, unit determinant, "
          "phase invariance of both residuals)")


def main() -> None:
    self_tests()
    rng = np.random.default_rng(BASE_SEED)

    print(f"\n(a) state reachability of G(theta)^dagger|00> "
          f"({NUM_STATE_TARGETS} Haar targets on the unit sphere of C^4)")
    state_worst = 0.0
    for index in range(NUM_STATE_TARGETS):
        target = rng.normal(size=4) + 1j * rng.normal(size=4)
        target /= np.linalg.norm(target)
        residual = best_residual(state_residual, target, RESTARTS_STATE, rng)
        state_worst = max(state_worst, residual)
        print(f"  target {index:>2}: 1 - fidelity = {residual:.3e}")

    print(f"\n(b) surjectivity onto SU(4) "
          f"({NUM_GATE_TARGETS} Haar targets in SU(4))")
    gate_worst = 0.0
    for index in range(NUM_GATE_TARGETS):
        target = unitary_group.rvs(4, random_state=int(rng.integers(1 << 31)))
        target = target / np.linalg.det(target) ** 0.25
        residual = best_residual(gate_residual, target, RESTARTS_GATE, rng)
        gate_worst = max(gate_worst, residual)
        print(f"  target {index:>2}: 1 - |Tr(G^dag U)/4|^2 = {residual:.3e}")

    print(f"\nworst-case state residual: {state_worst:.3e}")
    print(f"worst-case gate  residual: {gate_worst:.3e}")
    verdict = "SUPPORTED" if max(state_worst, gate_worst) < TOLERANCE else "FAILED"
    print(f"surjectivity hypothesis of the first-rung theorem: {verdict} "
          f"(tolerance {TOLERANCE:.0e})")
    print("This is numerical evidence for the equality branch of the theorem, "
          "not a proof; the inequality holds without it.")
    if verdict == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
