"""Numerical check of the deep-limit correlation prediction.

Measures the empirical correlation of the field delta between pairs of
parameter points against the deep-limit prediction
Corr = (2^n F - 1)/(2^n - 1), where F is the (exactly computed) probe
overlap. Fresh random instances; no optimization; independent of the P(q)
data and of its seed universe (base seed 2025 here, 42 there).

Usage:
    python analysis/check_kernel.py            # ~1-2 min, writes kernel_check.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pennylane as qml  # noqa: E402  (also wires up the repro path)
from pq_experiment import build_state_fn  # noqa: E402,F401  (import chains sys.path)
from peaked_circuits import (  # noqa: E402
    CircuitConfig,
    _apply_peaking_section,
    build_peak_weight_fn,
    sample_haar_random_layers,
)

NUM_QUBITS = 8
PEAKING_LAYERS = 4
RANDOM_DEPTHS = (1, 2, 4, 8)
SHIFTS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
NUM_INSTANCES = 300
BASE_SEED = 2025
BASE_THETA_SCALE = 0.5


def probe_state_fn(config: CircuitConfig):
    """theta -> V(theta)^dagger |0^n>, the probe state."""
    device = qml.device("default.qubit", wires=config.num_qubits)

    @qml.qnode(device)
    def probe(theta):
        qml.adjoint(_apply_peaking_section)(config, theta)
        return qml.state()

    return probe


def deep_limit_corr(overlap: np.ndarray, num_qubits: int) -> np.ndarray:
    dim = 2.0**num_qubits
    return (dim * overlap - 1.0) / (dim - 1.0)


def main() -> None:
    rng = np.random.default_rng(np.random.SeedSequence(BASE_SEED))
    results = {}
    for depth in RANDOM_DEPTHS:
        config = CircuitConfig(NUM_QUBITS, depth, PEAKING_LAYERS)
        num_parameters = config.num_peaking_parameters
        theta = rng.normal(0.0, BASE_THETA_SCALE, num_parameters)
        direction = rng.normal(0.0, 1.0, num_parameters)
        direction /= np.linalg.norm(direction)
        shifted = [theta + t * np.sqrt(num_parameters) * direction for t in SHIFTS]

        probe = probe_state_fn(config)
        base_state = np.asarray(probe(theta))
        overlaps = np.array(
            [np.abs(np.vdot(base_state, probe(point))) ** 2 for point in shifted]
        )

        start = time.perf_counter()
        values = np.empty((NUM_INSTANCES, 1 + len(SHIFTS)))
        for index in range(NUM_INSTANCES):
            instance_rng = np.random.default_rng(
                np.random.SeedSequence(BASE_SEED, spawn_key=(depth, index))
            )
            haar_layers = sample_haar_random_layers(config, instance_rng)
            field = build_peak_weight_fn(config, haar_layers)
            values[index, 0] = field(theta)
            for column, point in enumerate(shifted, start=1):
                values[index, column] = field(point)

        correlations = np.array(
            [
                np.corrcoef(values[:, 0], values[:, column])[0, 1]
                for column in range(1, 1 + len(SHIFTS))
            ]
        )
        results[depth] = (overlaps, correlations)
        elapsed = time.perf_counter() - start
        print(f"tau_r = {depth}  ({elapsed:.0f} s, {NUM_INSTANCES} instances)")
        for t, overlap, corr in zip(SHIFTS, overlaps, correlations, strict=True):
            print(
                f"  t = {t:4.2f}  F = {overlap:6.4f}  "
                f"Corr = {corr:+7.4f}  deep-limit = "
                f"{deep_limit_corr(overlap, NUM_QUBITS):+7.4f}"
            )

    fig, axis = plt.subplots(figsize=(4.8, 3.6))
    grid = np.linspace(0.0, 1.0, 200)
    axis.plot(
        grid,
        deep_limit_corr(grid, NUM_QUBITS),
        "-",
        color="0.3",
        label="deep limit $(2^nF-1)/(2^n-1)$",
    )
    for depth, marker in zip(RANDOM_DEPTHS, ("o", "s", "^", "D"), strict=True):
        overlaps, correlations = results[depth]
        axis.plot(
            overlaps, correlations, marker, linestyle="", markersize=5,
            label=rf"$\tau_r = {depth}$",
        )
    axis.set_xlabel("probe overlap $F(\\theta, \\theta')$")
    axis.set_ylabel("empirical Corr$[\\delta(\\theta), \\delta(\\theta')]$")
    axis.set_title(
        f"Field correlation vs probe overlap (n = {NUM_QUBITS}, "
        f"M = {NUM_INSTANCES})",
        fontsize=10,
    )
    axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "kernel_check.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
