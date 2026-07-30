"""Recorded optimization trajectories.

Runs the protocol optimizer while snapshotting the state every few steps:
growth curve delta_t, basin commitment q_fin(t) = |<psi_t|psi_final>|^2,
step-to-step smoothness, and cumulative parameter-space path length in
units of the cell scale xi measured in 05-envelope.

Usage:
    python analysis/trajectories.py     # ~12 min, writes trajectories.png/.npz
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pennylane as qml  # noqa: E402
from pennylane import numpy as pnp  # noqa: E402

from pq_experiment import RestartSettings, build_state_fn  # noqa: E402
from peaked_circuits import (  # noqa: E402
    CircuitConfig,
    build_peak_weight_fn,
    sample_haar_random_layers,
)

RUNS = ((8, 12), (12, 8))  # (n, restarts)
# xi from envelope_exact.py (plateau read off the ray tail, F < 0.01; the
# first version used a non-decorrelated second draw as the plateau, which
# gave xi = 2.17 / 3.34 and overcounted cells by ~2x -- corrected 2026-07-16).
CELL_SCALE = {8: 4.35, 12: 4.45}
RECORD_EVERY = 5
BASE_SEED = 8088


def record_trajectory(field, state_fn, num_parameters, settings, rng):
    theta = pnp.array(
        rng.normal(0.0, settings.init_scale, num_parameters), requires_grad=True
    )
    optimizer = qml.AdamOptimizer(stepsize=settings.learning_rate)

    def cost(parameters):
        return -field(parameters)

    steps, deltas, states, positions = [], [], [], []

    def snapshot(step):
        steps.append(step)
        deltas.append(float(field(theta)))
        states.append(np.asarray(state_fn(theta)))
        positions.append(np.asarray(theta).copy())

    snapshot(0)
    for step in range(1, settings.max_steps + 1):
        theta, _ = optimizer.step_and_cost(cost, theta)
        if step % settings.decay_every == 0:
            optimizer.stepsize *= settings.decay_factor
        if step % RECORD_EVERY == 0:
            snapshot(step)

    states = np.stack(states)
    final = states[-1]
    q_final = np.abs(states @ final.conj()) ** 2
    q_step = np.abs(np.sum(states[1:] * states[:-1].conj(), axis=1)) ** 2
    positions = np.stack(positions)
    increments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return dict(
        steps=np.array(steps),
        deltas=np.array(deltas),
        q_final=q_final,
        q_step=q_step,
        path_length=np.concatenate([[0.0], np.cumsum(increments)]),
        displacement=np.linalg.norm(positions - positions[0], axis=1),
    )


def main() -> None:
    settings = RestartSettings(convergence_tol=0.0)  # no early stop: full traces
    fig, axes = plt.subplots(len(RUNS), 3, figsize=(11, 3.1 * len(RUNS)))
    stored = {}
    for row, (num_qubits, num_restarts) in enumerate(RUNS):
        config = CircuitConfig(num_qubits, num_qubits, num_qubits // 2)
        layers = sample_haar_random_layers(
            config,
            np.random.default_rng(
                np.random.SeedSequence(42, spawn_key=(num_qubits, 0))
            ),
        )
        field = build_peak_weight_fn(config, layers)
        state_fn = build_state_fn(config, layers)
        xi = CELL_SCALE[num_qubits]

        commit_fractions, cells_crossed = [], []
        start = time.perf_counter()
        for restart in range(num_restarts):
            rng = np.random.default_rng(
                np.random.SeedSequence(BASE_SEED, spawn_key=(num_qubits, restart))
            )
            trace = record_trajectory(
                field, state_fn, config.num_peaking_parameters, settings, rng
            )
            stored[f"n{num_qubits}_r{restart}_deltas"] = trace["deltas"]
            stored[f"n{num_qubits}_r{restart}_qfinal"] = trace["q_final"]
            stored[f"n{num_qubits}_r{restart}_path"] = trace["path_length"]

            fraction = trace["steps"] / trace["steps"][-1]
            committed = np.flatnonzero(trace["q_final"] > 0.5)
            commit_fractions.append(
                float(fraction[committed[0]]) if len(committed) else 1.0
            )
            cells_crossed.append(float(trace["path_length"][-1] / xi))

            axes[row][0].semilogy(fraction, trace["deltas"], alpha=0.6, lw=0.9)
            axes[row][1].plot(fraction, trace["q_final"], alpha=0.6, lw=0.9)
            axes[row][2].plot(
                fraction, trace["path_length"] / xi, alpha=0.6, lw=0.9
            )

        axes[row][0].axhline(2.0**-num_qubits, color="0.5", ls="--", lw=0.8)
        axes[row][0].set_ylabel(rf"$\delta_t$  ($n={num_qubits}$)")
        axes[row][1].axhline(0.5, color="0.5", ls="--", lw=0.8)
        axes[row][1].set_ylabel(r"$q_{\rm fin}(t)$")
        axes[row][2].set_ylabel(r"path / $\xi$")
        print(
            f"n = {num_qubits}: median commit fraction = "
            f"{np.median(commit_fractions):.2f}, "
            f"median cells crossed = {np.median(cells_crossed):.1f}, "
            f"ln N_visited per run ~ {np.log(max(np.median(cells_crossed), 1.0)):.1f} "
            f"[{time.perf_counter() - start:.0f} s]"
        )
    for axis in axes[-1]:
        axis.set_xlabel("step fraction")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "trajectories.png"
    fig.savefig(out, dpi=200)
    np.savez_compressed(
        Path(__file__).resolve().parent / "trajectories.npz", **stored
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
