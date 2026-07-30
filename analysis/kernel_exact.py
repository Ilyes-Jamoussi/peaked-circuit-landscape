"""Exact finite-depth kernel of the field, and its Monte-Carlo falsification.

Implements the exact second-moment formula derived in 01-kernel §3:

    E[delta(theta) delta(theta')] = sum_c W_tau(c) * Tr[rho_R(c) rho'_R(c)]

where c runs over per-site spin configurations sigma in {e, s}^n, W_tau are
the exact weights of the two-copy spin transfer process (each Haar U(4)
gate projects the doubled space onto span{id, swap}: aligned inputs pass,
misaligned inputs go to 2/5 * (ee) + 2/5 * (ss)), R(c) is the set of
s-sites, and Tr[rho_R rho'_R] is the cross-purity of the two probe states
over R.

Self-tests: the transfer preserves the doubled trace exactly, and converges
at large depth to the Haar kernel (1 + F)/(2^n (2^n + 1)) of 01-kernel §2.
Falsification test: the exact correlation must match a fresh Monte-Carlo
estimate within statistical error at every probed depth.

Usage:
    python analysis/kernel_exact.py       # ~5 min, writes kernel_exact.png
"""

from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pennylane as qml  # noqa: E402
import pq_experiment  # noqa: E402,F401  (wires up the reproduction repo path)
from peaked_circuits import (  # noqa: E402
    CircuitConfig,
    _apply_peaking_section,
    brick_wall_pairs,
    build_peak_weight_fn,
    sample_haar_random_layers,
)

NUM_QUBITS = 8
PEAKING_LAYERS = 4
RANDOM_DEPTHS = (1, 2, 4, 8, 16)
SHIFTS = (0.02, 0.05, 0.1, 0.2, 0.4)
NUM_INSTANCES = 300
BASE_SEED = 2025
BASE_THETA_SCALE = 0.5
WALL_WEIGHT = 2.0 / 5.0  # q/(q^2+1) at local dimension q = 2
INIT_WEIGHT = 1.0 / 20.0  # (1 - 1/d)/(d^2 - 1) at d = 4, pure input


def spin_weights(num_qubits: int, depth: int) -> np.ndarray:
    """Exact weights W_tau(c) over configs c in {0,1}^n (bit = s-site)."""
    num_pairs = len(brick_wall_pairs(0, num_qubits))
    weights = np.zeros(2**num_qubits)
    for bits in product((0, 1), repeat=num_pairs):
        mask = 0
        for bit, (site_a, site_b) in zip(bits, brick_wall_pairs(0, num_qubits)):
            if bit:
                mask |= (1 << site_a) | (1 << site_b)
        weights[mask] = INIT_WEIGHT**num_pairs
    for layer in range(1, depth):
        for site_a, site_b in brick_wall_pairs(layer, num_qubits):
            updated = np.zeros_like(weights)
            mask_a, mask_b = 1 << site_a, 1 << site_b
            for config in np.flatnonzero(weights):
                bit_a, bit_b = bool(config & mask_a), bool(config & mask_b)
                if bit_a == bit_b:
                    updated[config] += weights[config]
                else:
                    both_e = config & ~(mask_a | mask_b)
                    both_s = config | mask_a | mask_b
                    updated[both_e] += WALL_WEIGHT * weights[config]
                    updated[both_s] += WALL_WEIGHT * weights[config]
            weights = updated
    return weights


def doubled_trace(weights: np.ndarray, num_qubits: int) -> float:
    """Tr of the averaged two-copy state; must be exactly 1."""
    configs = np.arange(len(weights))
    num_swaps = np.array([bin(c).count("1") for c in configs])
    return float(np.sum(weights * 4.0 ** (num_qubits - num_swaps) * 2.0**num_swaps))


def cross_purities(
    state_a: np.ndarray, state_b: np.ndarray, num_qubits: int
) -> np.ndarray:
    """Tr[rho_R rho'_R] for every config (R = s-sites of the bitmask)."""
    tensor_a = state_a.reshape([2] * num_qubits)
    tensor_b = state_b.reshape([2] * num_qubits)
    values = np.empty(2**num_qubits)
    for config in range(2**num_qubits):
        region = [q for q in range(num_qubits) if config & (1 << q)]
        complement = [q for q in range(num_qubits) if not config & (1 << q)]
        matrix_a = np.transpose(tensor_a, region + complement).reshape(
            2 ** len(region), -1
        )
        matrix_b = np.transpose(tensor_b, region + complement).reshape(
            2 ** len(region), -1
        )
        values[config] = float(
            np.sum(np.abs(matrix_a.conj().T @ matrix_b) ** 2)
        )
    return values


def exact_second_moment(
    weights: np.ndarray, state_a: np.ndarray, state_b: np.ndarray, num_qubits: int
) -> float:
    return float(np.dot(weights, cross_purities(state_a, state_b, num_qubits)))


def exact_correlation(
    weights: np.ndarray, state_a: np.ndarray, state_b: np.ndarray, num_qubits: int
) -> float:
    mean_sq = 4.0**-num_qubits
    cross = exact_second_moment(weights, state_a, state_b, num_qubits) - mean_sq
    var_a = exact_second_moment(weights, state_a, state_a, num_qubits) - mean_sq
    var_b = exact_second_moment(weights, state_b, state_b, num_qubits) - mean_sq
    return cross / np.sqrt(var_a * var_b)


def probe_state_fn(config: CircuitConfig):
    device = qml.device("default.qubit", wires=config.num_qubits)

    @qml.qnode(device)
    def probe(theta):
        qml.adjoint(_apply_peaking_section)(config, theta)
        return qml.state()

    return probe


def self_tests() -> None:
    for depth in (1, 2, 5, 12):
        weights = spin_weights(NUM_QUBITS, depth)
        trace = doubled_trace(weights, NUM_QUBITS)
        assert abs(trace - 1.0) < 1e-12, f"trace {trace} at depth {depth}"
    deep = spin_weights(NUM_QUBITS, 60)
    dim = 2.0**NUM_QUBITS
    rng = np.random.default_rng(7)
    vec_a = rng.normal(size=2**NUM_QUBITS) + 1j * rng.normal(size=2**NUM_QUBITS)
    vec_a /= np.linalg.norm(vec_a)
    vec_b = rng.normal(size=2**NUM_QUBITS) + 1j * rng.normal(size=2**NUM_QUBITS)
    vec_b /= np.linalg.norm(vec_b)
    fidelity = np.abs(np.vdot(vec_a, vec_b)) ** 2
    haar = (1.0 + fidelity) / (dim * (dim + 1.0))
    exact = exact_second_moment(deep, vec_a, vec_b, NUM_QUBITS)
    assert abs(exact - haar) / haar < 1e-9, f"deep limit off: {exact} vs {haar}"
    print("self-tests passed (trace preservation, Haar limit)")


def main() -> None:
    self_tests()
    rng = np.random.default_rng(np.random.SeedSequence(BASE_SEED))
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    all_ok = True
    for depth, color in zip(
        RANDOM_DEPTHS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"),
        strict=True,
    ):
        config = CircuitConfig(NUM_QUBITS, depth, PEAKING_LAYERS)
        num_parameters = config.num_peaking_parameters
        theta = rng.normal(0.0, BASE_THETA_SCALE, num_parameters)
        direction = rng.normal(0.0, 1.0, num_parameters)
        direction /= np.linalg.norm(direction)
        shifted = [theta + t * np.sqrt(num_parameters) * direction for t in SHIFTS]

        probe = probe_state_fn(config)
        base_state = np.asarray(probe(theta))
        shifted_states = [np.asarray(probe(point)) for point in shifted]
        overlaps = [
            float(np.abs(np.vdot(base_state, state)) ** 2)
            for state in shifted_states
        ]

        weights = spin_weights(NUM_QUBITS, depth)
        exact = [
            exact_correlation(weights, base_state, state, NUM_QUBITS)
            for state in shifted_states
        ]

        start = time.perf_counter()
        values = np.empty((NUM_INSTANCES, 1 + len(SHIFTS)))
        for index in range(NUM_INSTANCES):
            instance_rng = np.random.default_rng(
                np.random.SeedSequence(BASE_SEED, spawn_key=(depth, index, 999))
            )
            haar_layers = sample_haar_random_layers(config, instance_rng)
            field = build_peak_weight_fn(config, haar_layers)
            values[index, 0] = field(theta)
            for column, point in enumerate(shifted, start=1):
                values[index, column] = field(point)
        elapsed = time.perf_counter() - start

        print(f"tau_r = {depth}  ({elapsed:.0f} s)")
        bootstrap_rng = np.random.default_rng(11)
        for column, (t, overlap) in enumerate(zip(SHIFTS, overlaps, strict=True)):
            monte_carlo = float(
                np.corrcoef(values[:, 0], values[:, column + 1])[0, 1]
            )
            # Bootstrap standard error: delta is exponential-like, so the
            # Gaussian formula (1-r^2)/sqrt(M) badly underestimates it.
            resampled = np.empty(1000)
            for b in range(1000):
                rows = bootstrap_rng.integers(NUM_INSTANCES, size=NUM_INSTANCES)
                resampled[b] = np.corrcoef(
                    values[rows, 0], values[rows, column + 1]
                )[0, 1]
            stderr = float(np.std(resampled))
            gap = abs(exact[column] - monte_carlo)
            verdict = "ok" if gap < 2.0 * stderr + 1e-3 else "MISMATCH"
            all_ok &= verdict == "ok"
            print(
                f"  t = {t:4.2f}  F = {overlap:6.4f}  exact = {exact[column]:+7.4f}"
                f"  MC = {monte_carlo:+7.4f} +/- {stderr:5.3f}  {verdict}"
            )
        axis.plot(overlaps, exact, "-", color=color, linewidth=1.2)
        monte = [
            float(np.corrcoef(values[:, 0], values[:, k + 1])[0, 1])
            for k in range(len(SHIFTS))
        ]
        axis.plot(
            overlaps, monte, "o", color=color, markersize=4,
            label=rf"$\tau_r = {depth}$",
        )
    grid = np.linspace(0.0, 1.0, 100)
    dim = 2.0**NUM_QUBITS
    axis.plot(
        grid, (dim * grid - 1) / (dim - 1), ":", color="0.4", linewidth=1,
        label="deep limit",
    )
    axis.set_xlabel("probe overlap $F$")
    axis.set_ylabel("Corr$[\\delta(\\theta), \\delta(\\theta')]$")
    axis.set_title(
        "Exact finite-depth kernel (lines) vs Monte-Carlo (dots)", fontsize=10
    )
    axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "kernel_exact.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    print("FALSIFICATION TEST:", "PASSED" if all_ok else "FAILED")


if __name__ == "__main__":
    main()
