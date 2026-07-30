"""Ceiling-vs-depth curve and instance-exact A&Z Conjecture 3.2 (09, 3c).

--run launches the missing ensembles (60 restarts each) via
pq_experiment; --analyze reads all points, computes the trivial-inversion
baselines b(tau_r - tau_p) from partial statevectors of the SAME
instances, and prints the alpha table
alpha = ln delta* / ln b. Existing points are reused: tau_p = 2 from
results/shallow_peaking, tau_p = n/2 from results/pq (its 200-restart
atom value).

Usage:
    python analysis/delta_star_curve.py --run --workers 8
    python analysis/delta_star_curve.py --analyze
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401
from ceiling_bound import apply_gate  # noqa: E402
from peaked_circuits import (  # noqa: E402
    CircuitConfig,
    brick_wall_pairs,
    sample_haar_random_layers,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
INSTANCES = (0, 1, 2)
#: (n, tau_p) -> ensemble directory; None = to be produced by --run
POINTS = {
    (8, 1): "results/ceiling_curve/n8_taup1",
    (8, 2): "results/shallow_peaking",
    (8, 3): "results/ceiling_curve/n8_taup3",
    (8, 4): "results/pq",
    (6, 1): "results/ceiling_curve/n6_taup1",
    (6, 2): "results/ceiling_curve/n6_taup2",
    (6, 3): "results/ceiling_curve/n6_taup3",
}
NEW_POINTS = {key: path for key, path in POINTS.items()
              if "ceiling_curve" in path}


def ensemble_file(directory: str, n: int, instance: int) -> Path:
    return ROOT / directory / f"pq_n{n}_i{instance}_sigma0.1.npz"


def partial_baseline(n: int, instance: int, num_layers: int) -> float:
    """b(tau) = |<0|U_(1..tau)|0>|^2 for the stored instance seeds."""
    config = CircuitConfig(n, n, n // 2)
    layers = sample_haar_random_layers(
        config,
        np.random.default_rng(np.random.SeedSequence(42, spawn_key=(n, instance))),
    )
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    for layer_index, gates in enumerate(layers[:num_layers]):
        for (site, _), gate in zip(
            brick_wall_pairs(layer_index, n), gates, strict=True
        ):
            state = apply_gate(state, gate, site, n)
    return float(np.abs(state[0]) ** 2)


def run_missing(workers: int) -> None:
    for (n, tau_p), directory in NEW_POINTS.items():
        for instance in INSTANCES:
            target = ensemble_file(directory, n, instance)
            if target.exists():
                print(f"skip (exists): {target.relative_to(ROOT)}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "pq_experiment.py",
                   "--num-qubits", str(n), "--instance", str(instance),
                   "--peaking-layers", str(tau_p), "--restarts", "60",
                   "--workers", str(workers), "--output", directory]
            print("run:", " ".join(cmd[1:]), flush=True)
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                raise SystemExit(f"run failed: {cmd}")


def analyze() -> None:
    print(f"{'n':>3} {'tau_p':>6} {'inst':>4} {'delta*':>8} {'atom':>9} "
          f"{'b(tau_r-tau_p)':>14} {'alpha':>7}")
    for (n, tau_p), directory in sorted(POINTS.items()):
        for instance in INSTANCES:
            path = ensemble_file(directory, n, instance)
            if not path.exists():
                print(f"{n:>3} {tau_p:>6} {instance:>4}   (missing)")
                continue
            deltas = np.load(path)["peak_weights"]
            best = float(deltas.max())
            top10 = np.sort(deltas)[::-1][: min(10, len(deltas))]
            atom = float(top10[0] - top10[-1])
            baseline = partial_baseline(n, instance, n - tau_p)
            alpha = float(np.log(best) / np.log(baseline))
            print(f"{n:>3} {tau_p:>6} {instance:>4} {best:>8.4f} {atom:>9.1e} "
                  f"{baseline:>14.3e} {alpha:>7.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.run:
        run_missing(args.workers)
    if args.analyze or not args.run:
        analyze()


if __name__ == "__main__":
    main()
