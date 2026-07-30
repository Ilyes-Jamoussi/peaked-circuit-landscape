"""Robustness campaign for the finite-depth correlation excess (01-kernel §5).

Extends check_kernel.py along every axis a skeptic would probe: several
base points, three kinds of moves (global Gaussian, single-gate, single
layer), two system sizes, and depths up to 2n. Reports the excess
Delta = Corr_empirical - Corr_deep-limit with standard errors, and writes
kernel_robust.png.

Usage:
    python analysis/check_kernel_robust.py      # ~30 min, single process
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

import pennylane as qml  # noqa: E402
import pq_experiment  # noqa: E402,F401  (wires up the reproduction repo path)
from peaked_circuits import (  # noqa: E402
    CircuitConfig,
    PARAMETERS_PER_SU4_GATE,
    _apply_peaking_section,
    brick_wall_pairs,
    build_peak_weight_fn,
    sample_haar_random_layers,
)

QUBIT_COUNTS = (8, 10)
DEPTH_GRID = {8: (1, 2, 4, 8, 16), 10: (1, 2, 4, 10, 20)}
PEAKING_LAYERS = {8: 4, 10: 5}
NUM_BASES = 2
SHIFTS = (0.05, 0.12, 0.25, 0.5)
NUM_INSTANCES = 300
BASE_SEED = 2025
BASE_THETA_SCALE = 0.5
DIRECTION_KINDS = ("global", "gate", "layer")


def probe_state_fn(config: CircuitConfig):
    device = qml.device("default.qubit", wires=config.num_qubits)

    @qml.qnode(device)
    def probe(theta):
        qml.adjoint(_apply_peaking_section)(config, theta)
        return qml.state()

    return probe


def deep_limit_corr(overlap: float, num_qubits: int) -> float:
    dim = 2.0**num_qubits
    return (dim * overlap - 1.0) / (dim - 1.0)


def make_direction(
    kind: str, config: CircuitConfig, rng: np.random.Generator
):
    """Unit direction restricted to the move's support; returns (u, dims)."""
    num_parameters = config.num_peaking_parameters
    support = np.zeros(num_parameters, dtype=bool)
    if kind == "global":
        support[:] = True
    elif kind == "gate":
        gate = int(rng.integers(num_parameters // PARAMETERS_PER_SU4_GATE))
        start = gate * PARAMETERS_PER_SU4_GATE
        support[start : start + PARAMETERS_PER_SU4_GATE] = True
    elif kind == "layer":
        first_layer_gates = len(
            brick_wall_pairs(config.num_random_layers, config.num_qubits)
        )
        support[: PARAMETERS_PER_SU4_GATE * first_layer_gates] = True
    direction = np.where(support, rng.normal(0.0, 1.0, num_parameters), 0.0)
    return direction / np.linalg.norm(direction), int(support.sum())


def main() -> None:
    rows = []
    for num_qubits in QUBIT_COUNTS:
        for depth in DEPTH_GRID[num_qubits]:
            config = CircuitConfig(num_qubits, depth, PEAKING_LAYERS[num_qubits])
            num_parameters = config.num_peaking_parameters
            probe = probe_state_fn(config)
            rng = np.random.default_rng(
                np.random.SeedSequence(BASE_SEED, spawn_key=(num_qubits, depth, 77))
            )

            points = []  # (base_index, kind, t, theta_a, theta_b, F)
            for base_index in range(NUM_BASES):
                theta = rng.normal(0.0, BASE_THETA_SCALE, num_parameters)
                base_state = np.asarray(probe(theta))
                for kind in DIRECTION_KINDS:
                    direction, dims = make_direction(kind, config, rng)
                    for t in SHIFTS:
                        shifted = theta + t * np.sqrt(dims) * direction
                        overlap = float(
                            np.abs(np.vdot(base_state, probe(shifted))) ** 2
                        )
                        points.append((base_index, kind, t, theta, shifted, overlap))

            start = time.perf_counter()
            evaluations = np.empty((NUM_INSTANCES, NUM_BASES + len(points)))
            base_columns = {}
            for index in range(NUM_INSTANCES):
                instance_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        BASE_SEED, spawn_key=(num_qubits, depth, index)
                    )
                )
                haar_layers = sample_haar_random_layers(config, instance_rng)
                field = build_peak_weight_fn(config, haar_layers)
                seen = {}
                for column, (base_index, _, _, theta, shifted, _) in enumerate(
                    points
                ):
                    if base_index not in seen:
                        seen[base_index] = float(field(theta))
                        base_columns[base_index] = base_index
                        evaluations[index, base_index] = seen[base_index]
                    evaluations[index, NUM_BASES + column] = float(field(shifted))
            elapsed = time.perf_counter() - start

            for column, (base_index, kind, t, _, _, overlap) in enumerate(points):
                pair = np.corrcoef(
                    evaluations[:, base_index],
                    evaluations[:, NUM_BASES + column],
                )[0, 1]
                deep = deep_limit_corr(overlap, num_qubits)
                stderr = (1.0 - pair**2) / np.sqrt(NUM_INSTANCES)
                rows.append(
                    dict(
                        n=num_qubits,
                        depth=depth,
                        base=base_index,
                        kind=kind,
                        t=t,
                        overlap=overlap,
                        corr=pair,
                        deep=deep,
                        excess=pair - deep,
                        stderr=stderr,
                    )
                )
            print(
                f"n = {num_qubits}  tau_r = {depth:2d}  "
                f"({elapsed:.0f} s, {len(points)} pairs x {NUM_INSTANCES} instances)"
            )

    print(
        f"\n{'n':>3} {'tau':>4} {'base':>4} {'kind':>7} {'t':>5} "
        f"{'F':>7} {'Corr':>8} {'deep':>8} {'excess':>8} {'se':>6}"
    )
    for row in rows:
        print(
            f"{row['n']:>3} {row['depth']:>4} {row['base']:>4} {row['kind']:>7} "
            f"{row['t']:>5.2f} {row['overlap']:>7.4f} {row['corr']:>+8.4f} "
            f"{row['deep']:>+8.4f} {row['excess']:>+8.4f} {row['stderr']:>6.3f}"
        )

    fig, axes = plt.subplots(1, len(QUBIT_COUNTS), figsize=(9.6, 3.8), sharey=True)
    colors = {"global": "#1f77b4", "gate": "#d62728", "layer": "#2ca02c"}
    for axis, num_qubits in zip(axes, QUBIT_COUNTS, strict=True):
        depths = DEPTH_GRID[num_qubits]
        for row in rows:
            if row["n"] != num_qubits:
                continue
            size = 3 + 10 * depths.index(row["depth"]) / len(depths)
            axis.errorbar(
                row["overlap"],
                row["excess"],
                yerr=row["stderr"],
                marker="o",
                markersize=size,
                color=colors[row["kind"]],
                linestyle="",
                alpha=0.75,
            )
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.set_title(
            f"n = {num_qubits} (marker size grows with depth)", fontsize=10
        )
        axis.set_xlabel("probe overlap $F$")
    axes[0].set_ylabel("excess Corr $-$ deep limit")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=c, label=k)
        for k, c in colors.items()
    ]
    axes[0].legend(handles=handles, fontsize=8, frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "kernel_robust.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
