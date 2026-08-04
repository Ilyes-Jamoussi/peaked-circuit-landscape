"""Paired Adam vs plain-SGD pacing traces on the robustness instance.

Reruns the frozen protocol (400 steps, lr 0.05 decayed x0.5 every 300,
init scale 0.1) with both optimizers from identical initializations on
the campaign's robustness instance (n = 10, instance 0), recording what
the restart archives discard: the per-step gradient norm and the
per-step parameter displacement. The two arms see the same gradients;
what separates them is the pacing of the update, which is the criterion
the stable-local class is defined by.

Usage:
    python analysis/optimizer_pacing.py     # ~25 min, writes optimizer_pacing.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pennylane as qml  # noqa: E402
from pennylane import numpy as pnp  # noqa: E402

from pq_experiment import RestartSettings  # noqa: E402
from peaked_circuits import (  # noqa: E402
    CircuitConfig,
    build_peak_weight_fn,
    sample_haar_random_layers,
)

NUM_QUBITS, INSTANCE = 10, 0  # the SGD-control instance of the campaign
NUM_RESTARTS = 12
BASE_SEED = 8088  # trajectories.py convention: decorrelated from campaign seeds
ARMS = ("adam", "sgd")


def make_optimizer(arm: str, learning_rate: float):
    if arm == "adam":
        return qml.AdamOptimizer(stepsize=learning_rate)
    return qml.GradientDescentOptimizer(stepsize=learning_rate)


def run_one(field, settings, arm, theta_init):
    theta = pnp.array(theta_init.copy(), requires_grad=True)
    optimizer = make_optimizer(arm, settings.learning_rate)

    def cost(parameters):
        return -field(parameters)

    deltas = np.empty(settings.max_steps)
    grad_norms = np.empty(settings.max_steps)
    update_norms = np.empty(settings.max_steps)
    for step in range(1, settings.max_steps + 1):
        gradient, forward = optimizer.compute_grad(cost, (theta,), {})
        new_args = optimizer.apply_grad(gradient, (theta,))
        new_theta = new_args[0] if isinstance(new_args, (list, tuple)) else new_args
        raw = gradient[0] if isinstance(gradient, (list, tuple)) else gradient
        deltas[step - 1] = -float(forward)
        grad_norms[step - 1] = float(np.linalg.norm(np.asarray(raw)))
        update_norms[step - 1] = float(
            np.linalg.norm(np.asarray(new_theta) - np.asarray(theta))
        )
        theta = new_theta
        if step % settings.decay_every == 0:
            optimizer.stepsize *= settings.decay_factor
    return deltas, grad_norms, update_norms, np.asarray(theta)


def self_test() -> None:
    """The manual compute_grad/apply_grad loop must replay step_and_cost."""
    settings = RestartSettings(convergence_tol=0.0, max_steps=30)
    config = CircuitConfig(4, 4, 2)
    layers = sample_haar_random_layers(
        config, np.random.default_rng(np.random.SeedSequence(7))
    )
    field = build_peak_weight_fn(config, layers)
    rng = np.random.default_rng(np.random.SeedSequence(11))
    theta_init = rng.normal(0.0, settings.init_scale, config.num_peaking_parameters)
    for arm in ARMS:
        theta_ref = pnp.array(theta_init.copy(), requires_grad=True)
        optimizer = make_optimizer(arm, settings.learning_rate)

        def cost(parameters):
            return -field(parameters)

        for _ in range(settings.max_steps):
            theta_ref, _ = optimizer.step_and_cost(cost, theta_ref)
        _, _, _, theta_end = run_one(field, settings, arm, theta_init)
        reference = float(field(theta_ref))
        replayed = float(field(pnp.array(theta_end, requires_grad=True)))
        assert abs(replayed - reference) < 1e-12, (arm, replayed, reference)
    print("self-test passed (instrumented loop replays step_and_cost, both arms)")


def main() -> None:
    self_test()
    settings = RestartSettings(convergence_tol=0.0)  # no early stop: full traces
    config = CircuitConfig(NUM_QUBITS, NUM_QUBITS, NUM_QUBITS // 2)
    layers = sample_haar_random_layers(
        config,
        np.random.default_rng(
            np.random.SeedSequence(42, spawn_key=(NUM_QUBITS, INSTANCE))
        ),
    )
    field = build_peak_weight_fn(config, layers)
    print(
        f"\nn = {NUM_QUBITS}, instance {INSTANCE}: {NUM_RESTARTS} paired restarts, "
        f"lr {settings.learning_rate}, {settings.max_steps} steps, "
        f"decay x{settings.decay_factor} every {settings.decay_every}"
    )
    stored: dict[str, np.ndarray] = {}
    summary = {arm: [] for arm in ARMS}
    for restart in range(NUM_RESTARTS):
        rng = np.random.default_rng(
            np.random.SeedSequence(BASE_SEED, spawn_key=(NUM_QUBITS, restart))
        )
        theta_init = rng.normal(
            0.0, settings.init_scale, config.num_peaking_parameters
        )
        for arm in ARMS:
            start = time.perf_counter()
            deltas, grad_norms, update_norms, _ = run_one(
                field, settings, arm, theta_init
            )
            stored[f"{arm}_r{restart}_deltas"] = deltas
            stored[f"{arm}_r{restart}_gradnorm"] = grad_norms
            stored[f"{arm}_r{restart}_updatenorm"] = update_norms
            summary[arm].append(
                [
                    deltas[-1],
                    float(np.median(grad_norms)),
                    float(np.median(update_norms)),
                    float(np.sum(update_norms)),
                ]
            )
            print(
                f"  restart {restart:2d} {arm:4s}: final delta {deltas[-1]:.4f}  "
                f"med|grad| {np.median(grad_norms):.2e}  "
                f"med|dtheta| {np.median(update_norms):.2e}  "
                f"path {np.sum(update_norms):.2f}  "
                f"[{time.perf_counter() - start:.0f} s]",
                flush=True,
            )

    out = Path(__file__).resolve().parent / "optimizer_pacing.npz"
    np.savez_compressed(out, **stored)
    print(f"\n{'arm':6s} {'final delta (med)':>18s} {'med|grad|':>11s} "
          f"{'med|dtheta|':>12s} {'path':>8s}")
    for arm in ARMS:
        table = np.array(summary[arm])
        print(
            f"{arm:6s} {np.median(table[:, 0]):18.4f} "
            f"{np.median(table[:, 1]):11.2e} "
            f"{np.median(table[:, 2]):12.2e} {np.median(table[:, 3]):8.2f}"
        )
    adam, sgd = np.array(summary["adam"]), np.array(summary["sgd"])
    print(
        "paired ratios (median over restarts): "
        f"|dtheta| adam/sgd = {np.median(adam[:, 2] / sgd[:, 2]):.1f}, "
        f"path adam/sgd = {np.median(adam[:, 3] / sgd[:, 3]):.1f}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
