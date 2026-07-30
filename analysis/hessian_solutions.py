"""Local geometry at atom solutions: gradient, Hessian, shelf dimension.

Gradient of delta from the exact derivative states of init_statistics;
Hessian by central differences of that analytic gradient (symmetry =
built-in error gauge); projective output-state Jacobian rank (its corank
= gauge dimension); shelf dimension = dim ker(Hessian) - gauge dimension.

Self-tests: analytic gradient vs 4th-order finite differences of delta;
output state matches pq_experiment's build_state_fn; generic projective
Jacobian rank min(P, 2D-2) at a random point (n = 4), where the Hessian
is indefinite.

Usage:
    python analysis/hessian_solutions.py     # ~25 min, n = 8 atom set
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401
from ceiling_bound import scrambled_state  # noqa: E402
from init_statistics import (  # noqa: E402
    apply_two_qubit,
    gate_matrix,
    peaking_gates,
    probe_and_derivatives,
)
from peaked_circuits import CircuitConfig  # noqa: E402
from pq_experiment import build_state_fn  # noqa: E402
from peaked_circuits import sample_haar_random_layers  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
FD_STEP = 1e-4


def gradient(config: CircuitConfig, theta: np.ndarray, phi: np.ndarray):
    """Exact analytic gradient of delta = |<w|phi>|^2."""
    w, derivatives = probe_and_derivatives(config, theta)
    amplitude = np.vdot(w, phi)  # <w|phi>
    partials = derivatives.conj() @ phi  # <d_a w|phi>
    return 2.0 * (np.conj(amplitude) * partials).real, float(
        (amplitude * np.conj(amplitude)).real
    )


def hessian(config: CircuitConfig, theta: np.ndarray, phi: np.ndarray):
    """Central differences of the analytic gradient; returns (H, asymmetry)."""
    num = len(theta)
    columns = np.empty((num, num))
    for b in range(num):
        plus, minus = theta.copy(), theta.copy()
        plus[b] += FD_STEP
        minus[b] -= FD_STEP
        g_plus, _ = gradient(config, plus, phi)
        g_minus, _ = gradient(config, minus, phi)
        columns[:, b] = (g_plus - g_minus) / (2 * FD_STEP)
    asymmetry = float(np.abs(columns - columns.T).max())
    return 0.5 * (columns + columns.T), asymmetry


def output_and_derivatives(config: CircuitConfig, theta: np.ndarray,
                           phi: np.ndarray):
    """psi = V(theta) phi and all d_a psi, by operator insertion."""
    n = config.num_qubits
    sites = peaking_gates(config)
    per_gate = theta.reshape(len(sites), 15)
    gate_mats = [gate_matrix(per_gate[g]) for g in range(len(sites))]

    prefixes = [phi]
    for g in range(len(sites)):
        prefixes.append(apply_two_qubit(prefixes[-1], gate_mats[g],
                                        sites[g], n))
    psi = prefixes[-1]

    derivatives = np.empty((len(sites) * 15, 2**n), dtype=complex)
    for g in range(len(sites)):
        for k in range(15):
            dgate = gate_matrix(per_gate[g], derivative=k)
            vec = apply_two_qubit(prefixes[g], dgate, sites[g], n)
            for gg in range(g + 1, len(sites)):
                vec = apply_two_qubit(vec, gate_mats[gg], sites[gg], n)
            derivatives[g * 15 + k] = vec
    return psi, derivatives


def projective_rank(psi: np.ndarray, derivatives: np.ndarray):
    """Real rank of (1 - psi psi^dag) d psi, and its singular values."""
    projected = derivatives - np.outer(derivatives @ psi.conj(), psi)
    real_map = np.concatenate([projected.real, projected.imag], axis=1)
    singular = np.linalg.svd(real_map.T, compute_uv=False)
    scale = singular.max()
    rank = int(np.sum(singular > 1e-8 * scale))
    return rank, singular


def self_tests() -> None:
    rng = np.random.default_rng(23)
    config = CircuitConfig(4, 4, 2)
    theta = rng.normal(0.0, 0.3, config.num_peaking_parameters)
    phi = scrambled_state(
        config, np.random.default_rng(np.random.SeedSequence(42, spawn_key=(4, 0)))
    )

    # (a) analytic gradient vs 4th-order finite differences of delta
    grad, _ = gradient(config, theta, phi)
    from init_statistics import probe_and_derivatives_cheap

    def delta_at(point):
        w = probe_and_derivatives_cheap(config, point)
        return abs(np.vdot(w, phi)) ** 2

    for a in (0, 11, 29, len(theta) - 1):
        h = 1e-3
        stencil = []
        for shift in (-2, -1, 1, 2):
            point = theta.copy()
            point[a] += shift * h
            stencil.append(delta_at(point))
        numeric = (stencil[0] - 8 * stencil[1] + 8 * stencil[2] - stencil[3]) / (
            12 * h
        )
        assert abs(numeric - grad[a]) < 1e-9, (a, numeric, grad[a])

    # (b) output state matches pq_experiment's build_state_fn
    layers = sample_haar_random_layers(
        config, np.random.default_rng(np.random.SeedSequence(42, spawn_key=(4, 0)))
    )
    psi, dpsi = output_and_derivatives(config, theta, phi)
    reference = np.asarray(build_state_fn(config, layers)(theta))
    assert np.allclose(psi, reference, atol=1e-10), "output state mismatch"

    # (c) generic projective rank = min(P, 2D-2); Hessian indefinite there
    rank, _ = projective_rank(psi, dpsi)
    expected = min(config.num_peaking_parameters, 2 * 2**4 - 2)
    assert rank == expected, (rank, expected)
    h_matrix, asym = hessian(config, theta, phi)
    eigenvalues = np.linalg.eigvalsh(h_matrix)
    assert asym < 1e-6, f"Hessian asymmetry {asym}"
    assert eigenvalues.min() < -1e-6 and eigenvalues.max() > 1e-6, "not indefinite"
    print("self-tests passed (gradient, output state, generic rank "
          f"{rank} = {expected}, indefinite random Hessian)")


def main() -> None:
    self_tests()
    num_qubits, instance = 8, 0
    config = CircuitConfig(num_qubits, num_qubits, num_qubits // 2)
    phi = scrambled_state(
        config,
        np.random.default_rng(
            np.random.SeedSequence(42, spawn_key=(num_qubits, instance))
        ),
    )
    data = np.load(RESULTS / "pq" / f"pq_n{num_qubits}_i{instance}_sigma0.1.npz")
    deltas, thetas = data["peak_weights"], data["thetas_final"]
    order = np.argsort(deltas)[::-1]
    chosen = list(order[:4]) + [order[100]]  # atom set + one mid-band

    print(f"\n{'sol':>5} {'delta':>8} {'|grad|':>9} {'lam_min':>9} "
          f"{'lam_max':>9} {'#<1e-6/-4/-3/-2/-1':>22} {'gauge':>6} "
          f"{'asym':>8}")
    for index in chosen:
        start = time.perf_counter()
        theta = thetas[index]
        grad, delta = gradient(config, theta, phi)
        h_matrix, asym = hessian(config, theta, phi)
        eigenvalues = np.linalg.eigvalsh(h_matrix)
        psi, dpsi = output_and_derivatives(config, theta, phi)
        rank, _ = projective_rank(psi, dpsi)
        gauge = config.num_peaking_parameters - rank
        # At a not-exactly-critical point (|grad| ~ 1e-3, learning-rate
        # decay stop), gauge directions carry |lambda| up to
        # |grad| x orbit curvature, NOT 0: read the spectrum through a
        # tolerance ladder and locate the flat band against the gauge
        # count.
        ladder = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
        counts = [int(np.sum(np.abs(eigenvalues) < tol)) for tol in ladder]
        print(f"{index:>5} {delta:>8.4f} {np.linalg.norm(grad):>9.2e} "
              f"{eigenvalues.min():>9.2e} {eigenvalues.max():>9.2e} "
              f"{'/'.join(str(c) for c in counts):>22} {gauge:>6} "
              f"{asym:>8.1e} [{time.perf_counter() - start:.0f} s]")
        small = np.sort(np.abs(eigenvalues))[:80]
        print("       |lam| percentiles 10/40/60/80: "
              + " ".join(f"{small[k]:.1e}" for k in (9, 39, 59, 79)))


if __name__ == "__main__":
    main()
