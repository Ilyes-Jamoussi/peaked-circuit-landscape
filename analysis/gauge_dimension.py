"""Derived gauge dimension of the brickwall parametrization vs measured corank.

Sec. II B counts the continuous gauge of the peaking parametrization: every
internal wire segment -- a qubit q together with two consecutive gates of V
acting on q -- carries an SU(2) of u u^dagger insertions absorbed into its
two endpoint gates, and no other continuous redundancy exists. The count is

    dim(gauge) = 3 S,   S = (n - 2)(tau_p - 1) + 2 floor((tau_p - 1) / 2)

for the even-n, even-start brickwall convention used throughout. At the
operating configuration (n, tau_p) = (8, 4): 3 S = 60.

Three claims are tested, in increasing strength:

  (a) Construction. Dressing the two endpoint gates of one segment by
      u^dagger and u and re-solving the fifteen-rotation word reproduces
      V(theta) exactly at a displaced theta': the gauge family is exhibited,
      not inferred. Tested on an interior segment and on a chain-edge
      segment, whose insertion commutes through an idle intermediate layer.
  (b) Saturation. The corank of the projective output-state Jacobian equals
      max(3S, P - (2D - 2)) at Haar-random points of every configuration of
      the campaign grid: the count exhausts the continuous redundancy, and
      the only excess over 3S is the trivial state-space cap (n = 4).
  (c) At the solutions. The corank at the atom sets of (8, 4), (10, 5) and
      (12, 6) equals 3S = 60, 108, 162; the (8, 4) selection is that of
      hessian_solutions.py, whose committed log this cross-checks.

Self-tests: the closed form for S against direct enumeration of segments;
the generic-rank floor min(P, 2D - 2) at the n = 4 random point of
hessian_solutions; unit determinant of the dressed gates.

Usage:
    python analysis/gauge_dimension.py     # ~1 min, dominated by n = 16
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401  (wires up the reproduction repo path)
from ceiling_bound import scrambled_state  # noqa: E402
from hessian_solutions import output_and_derivatives, projective_rank  # noqa: E402
from init_statistics import apply_two_qubit, gate_matrix  # noqa: E402
from peaked_circuits import CircuitConfig, brick_wall_pairs  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
BASE_SEED = 7130
PAULI = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]

GRID = [(4, 2), (6, 2), (6, 3), (8, 1), (8, 2), (8, 3), (8, 4),
        (10, 4), (10, 5), (12, 6), (14, 7), (16, 8)]


def segments(config: CircuitConfig):
    """(qubit, gate_a, gate_b) per internal wire segment, by enumeration."""
    gate_pairs = []
    for layer in range(config.num_peaking_layers):
        layer_index = config.num_random_layers + layer
        gate_pairs.extend(brick_wall_pairs(layer_index, config.num_qubits))
    found = []
    for q in range(config.num_qubits):
        acting = [g for g, pair in enumerate(gate_pairs) if q in pair]
        found.extend((q, a, b) for a, b in zip(acting, acting[1:]))
    return found, gate_pairs


def closed_form(n: int, tau_p: int) -> int:
    """S = (n - 2)(tau_p - 1) + 2 floor((tau_p - 1)/2), even-start brickwall."""
    return (n - 2) * (tau_p - 1) + 2 * ((tau_p - 1) // 2)


def su2(axis: np.ndarray, angle: float) -> np.ndarray:
    direction = axis / np.linalg.norm(axis)
    generator = sum(c * p for c, p in zip(direction, PAULI))
    return (np.cos(angle / 2) * np.eye(2)
            - 1j * np.sin(angle / 2) * generator)


def embed(u: np.ndarray, slot: int) -> np.ndarray:
    return np.kron(u, np.eye(2)) if slot == 0 else np.kron(np.eye(2), u)


def solve_word(target: np.ndarray, warm: np.ndarray):
    """Angles of the fifteen-rotation word reaching target, warm-started."""

    def residual(x):
        difference = (gate_matrix(x) - target).ravel()
        return np.concatenate([difference.real, difference.imag])

    result = least_squares(residual, warm, method="lm",
                           xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return result.x, float(np.sum(result.fun**2))


def circuit_matrix(config: CircuitConfig, gate_mats, gate_pairs) -> np.ndarray:
    n = config.num_qubits
    columns = np.eye(2**n, dtype=complex)
    out = np.empty_like(columns)
    for c in range(2**n):
        vec = columns[:, c]
        for g, mat in enumerate(gate_mats):
            vec = apply_two_qubit(vec, mat, gate_pairs[g][0], n)
        out[:, c] = vec
    return out


def corank(config: CircuitConfig, theta: np.ndarray, phi: np.ndarray) -> int:
    psi, dpsi = output_and_derivatives(config, theta, phi)
    rank, _ = projective_rank(psi, dpsi)
    return config.num_peaking_parameters - rank


def self_tests() -> None:
    for n, tau_p in GRID + [(6, 4), (10, 3), (12, 5)]:
        config = CircuitConfig(n, n, tau_p)
        found, _ = segments(config)
        assert len(found) == closed_form(n, tau_p), (n, tau_p, len(found))

    # generic-rank floor of hessian_solutions at the n = 4 random point:
    # P - (2D - 2) = 15 > 3S = 6, the one grid point where the cap binds
    rng = np.random.default_rng(23)
    config = CircuitConfig(4, 4, 2)
    theta = rng.normal(0.0, 0.3, config.num_peaking_parameters)
    phi = scrambled_state(
        config, np.random.default_rng(np.random.SeedSequence(42, spawn_key=(4, 0)))
    )
    assert corank(config, theta, phi) == 45 - 30

    u = su2(np.array([1.0, -0.4, 0.2]), 0.3)
    for slot in (0, 1):
        assert abs(np.linalg.det(embed(u, slot)) - 1.0) < 1e-12

    print("self-tests passed (segment closed form on the grid + odd depths, "
          "n = 4 cap corank 15, dressed gates in SU(4))")


def constructive(config: CircuitConfig, segment, label: str) -> None:
    """Dress one segment, re-solve the word, compare circuits exactly."""
    q, a, b = segment
    rng = np.random.default_rng(BASE_SEED + a + b)
    theta = rng.uniform(0.0, 2.0 * np.pi, config.num_peaking_parameters)
    per_gate = theta.reshape(-1, 15).copy()
    _, gate_pairs = segments(config)
    gate_mats = [gate_matrix(row) for row in per_gate]

    u = su2(rng.normal(size=3), 0.3)
    slot_a = 0 if gate_pairs[a][0] == q else 1
    slot_b = 0 if gate_pairs[b][0] == q else 1
    dressed_a = embed(u, slot_a).conj().T @ gate_mats[a]
    dressed_b = gate_mats[b] @ embed(u, slot_b)

    angles_a, res_a = solve_word(dressed_a, per_gate[a])
    angles_b, res_b = solve_word(dressed_b, per_gate[b])
    assert max(res_a, res_b) < 1e-20, (res_a, res_b)

    new_mats = list(gate_mats)
    new_mats[a], new_mats[b] = gate_matrix(angles_a), gate_matrix(angles_b)
    difference = np.linalg.norm(
        circuit_matrix(config, new_mats, gate_pairs)
        - circuit_matrix(config, gate_mats, gate_pairs)
    )
    new_theta = per_gate.copy()
    new_theta[a], new_theta[b] = angles_a, angles_b
    displacement = np.linalg.norm(new_theta.ravel() - theta)
    assert difference < 1e-9 and displacement > 1e-2, (difference, displacement)
    print(f"  {label}: wire {q}, gates {a} -> {b}, "
          f"|theta' - theta| = {displacement:.3f}, "
          f"||V' - V||_F = {difference:.1e}, word residuals "
          f"{res_a:.1e}/{res_b:.1e}")


def main() -> None:
    self_tests()

    print("\n(a) constructive gauge families at (8, 4)")
    config = CircuitConfig(8, 8, 4)
    found, _ = segments(config)
    interior = next(s for s in found if s[0] == 3)
    edge = next(s for s in found if s[0] == 0)
    constructive(config, interior, "interior segment")
    constructive(config, edge, "edge segment (through an idle layer)")

    print("\n(b) corank at Haar-random points vs max(3S, P - (2D - 2))")
    print(f"{'(n, tau_p)':>11} {'P':>5} {'3S':>5} {'P-(2D-2)':>9} "
          f"{'predicted':>10} {'corank':>7}")
    for n, tau_p in GRID:
        start = time.perf_counter()
        config = CircuitConfig(n, n, tau_p)
        big_p = config.num_peaking_parameters
        cap = big_p - (2 * 2**n - 2)
        predicted = max(3 * closed_form(n, tau_p), cap, 0)
        phi = scrambled_state(
            config,
            np.random.default_rng(np.random.SeedSequence(42, spawn_key=(n, 0))),
        )
        rng = np.random.default_rng(BASE_SEED + 100 * n + tau_p)
        points = 2 if (n, tau_p) == (8, 4) else 1
        measured = [
            corank(config, rng.uniform(0.0, 2.0 * np.pi, big_p), phi)
            for _ in range(points)
        ]
        assert all(m == predicted for m in measured), (n, tau_p, measured)
        shown = "/".join(str(m) for m in measured)
        print(f"{f'({n}, {tau_p})':>11} {big_p:>5} {3 * closed_form(n, tau_p):>5} "
              f"{cap:>9} {predicted:>10} {shown:>7} "
              f"[{time.perf_counter() - start:.0f} s]")

    print("\n(c) corank at the atom solutions")
    for n, picks in ((8, 5), (10, 2), (12, 2)):
        config = CircuitConfig(n, n, n // 2)
        expected = 3 * closed_form(n, n // 2)
        phi = scrambled_state(
            config,
            np.random.default_rng(np.random.SeedSequence(42, spawn_key=(n, 0))),
        )
        data = np.load(RESULTS / "pq" / f"pq_n{n}_i0_sigma0.1.npz")
        order = np.argsort(data["peak_weights"])[::-1]
        chosen = (list(order[:4]) + [order[100]]) if n == 8 else list(order[:picks])
        coranks = [corank(config, data["thetas_final"][i], phi) for i in chosen]
        assert all(c == expected for c in coranks), (n, coranks)
        shown = "/".join(str(c) for c in coranks)
        print(f"  (n, tau_p) = ({n}, {n // 2}): corank {shown} = 3S = {expected} "
              f"at solutions {'/'.join(str(i) for i in chosen)}")

    print("\nall checks passed: the counted gauge saturates the measured corank")


if __name__ == "__main__":
    main()
