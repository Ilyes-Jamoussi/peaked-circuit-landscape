"""Residuals of the robustness scan against the EXACT kernel, Eq. (5).

check_kernel_robust.py measures the excess of the empirical correlation over
the DEEP-LIMIT line and reports Gaussian standard errors (1 - r^2)/sqrt(M).
Neither is what the paper quotes: the text reports the residual against the
exact finite-depth kernel, with bootstrap errors, over the same 240 points.
This script computes exactly that, so the quoted number has a committed log.

Same point set as check_kernel_robust.py by construction -- the base points,
directions and instance seeds are regenerated from its own helpers -- so the
two logs are row-for-row comparable.

Why the error model matters: the appendix records that the Gaussian formula
badly underestimates the sampling error of a correlation between
exponential-tailed variables. The self-tests below demonstrate that on
synthetic data, and every error here is bootstrap.

Self-tests: the exact kernel gives correlation 1 at coincident points and
reproduces the deep-limit formula at large depth; the bootstrap standard
error of a correlation between exponential variables exceeds the Gaussian
formula.

Usage:
    python analysis/check_kernel_exact_residuals.py            # ~50 min
    python analysis/check_kernel_exact_residuals.py --sizes 8  # one size
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401  (wires up the reproduction repo path)
from check_kernel_robust import (  # noqa: E402  (identical point set)
    BASE_SEED,
    BASE_THETA_SCALE,
    DEPTH_GRID,
    DIRECTION_KINDS,
    NUM_BASES,
    NUM_INSTANCES,
    PEAKING_LAYERS,
    QUBIT_COUNTS,
    SHIFTS,
    deep_limit_corr,
    make_direction,
)
from kernel_exact import exact_correlation, probe_state_fn, spin_weights  # noqa: E402
from peaked_circuits import (  # noqa: E402
    CircuitConfig,
    build_peak_weight_fn,
    sample_haar_random_layers,
)

BOOTSTRAP = 1000


def bootstrap_stderr(a: np.ndarray, b: np.ndarray, index: np.ndarray) -> float:
    """Bootstrap standard error of corr(a, b) over precomputed resamples."""
    return float(np.std([np.corrcoef(a[row], b[row])[0, 1] for row in index]))


def self_tests() -> None:
    num_qubits = 6
    config = CircuitConfig(num_qubits, 4, 3)
    probe = probe_state_fn(config)
    rng = np.random.default_rng(11)
    theta = rng.normal(0.0, BASE_THETA_SCALE, config.num_peaking_parameters)
    state_a = np.asarray(probe(theta))
    state_b = np.asarray(probe(theta + 0.3 * rng.normal(
        size=config.num_peaking_parameters)))

    coincident = exact_correlation(spin_weights(num_qubits, 4), state_a,
                                   state_a, num_qubits)
    assert abs(coincident - 1.0) < 1e-10, f"corr at theta = theta' is {coincident}"

    deep = exact_correlation(spin_weights(num_qubits, 80), state_a, state_b,
                             num_qubits)
    overlap = float(abs(np.vdot(state_a, state_b)) ** 2)
    assert abs(deep - deep_limit_corr(overlap, num_qubits)) < 1e-9, (
        f"deep limit {deep} vs closed form {deep_limit_corr(overlap, num_qubits)}"
    )

    # The point of the bootstrap: on exponential-tailed variables the
    # Gaussian formula understates the spread of a sample correlation.
    draws = rng.exponential(size=(NUM_INSTANCES, 2))
    x = draws[:, 0]
    y = 0.7 * draws[:, 0] + 0.3 * draws[:, 1]
    index = rng.integers(NUM_INSTANCES, size=(2000, NUM_INSTANCES))
    boot = bootstrap_stderr(x, y, index)
    gaussian = (1.0 - np.corrcoef(x, y)[0, 1] ** 2) / np.sqrt(NUM_INSTANCES)
    assert boot > gaussian, (boot, gaussian)
    print("self-tests passed (coincident correlation 1, deep limit, "
          f"bootstrap {boot:.4f} > Gaussian {gaussian:.4f} on exponential data)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=str, default=None,
                        help="comma-separated n filter (shards the scan)")
    args = parser.parse_args()

    self_tests()
    sizes = ({int(s) for s in args.sizes.split(",")} if args.sizes
             else set(QUBIT_COUNTS))

    rows = []
    for num_qubits in QUBIT_COUNTS:
        if num_qubits not in sizes:
            continue
        for depth in DEPTH_GRID[num_qubits]:
            config = CircuitConfig(num_qubits, depth, PEAKING_LAYERS[num_qubits])
            num_parameters = config.num_peaking_parameters
            probe = probe_state_fn(config)
            weights = spin_weights(num_qubits, depth)
            rng = np.random.default_rng(
                np.random.SeedSequence(BASE_SEED,
                                       spawn_key=(num_qubits, depth, 77))
            )

            points = []
            for base_index in range(NUM_BASES):
                theta = rng.normal(0.0, BASE_THETA_SCALE, num_parameters)
                base_state = np.asarray(probe(theta))
                for kind in DIRECTION_KINDS:
                    direction, dims = make_direction(kind, config, rng)
                    for shift in SHIFTS:
                        shifted = theta + shift * np.sqrt(dims) * direction
                        state = np.asarray(probe(shifted))
                        points.append((
                            base_index, kind, shift, theta, shifted,
                            float(abs(np.vdot(base_state, state)) ** 2),
                            float(exact_correlation(weights, base_state, state,
                                                    num_qubits)),
                        ))

            start = time.perf_counter()
            values = np.empty((NUM_INSTANCES, NUM_BASES + len(points)))
            for index in range(NUM_INSTANCES):
                instance_rng = np.random.default_rng(
                    np.random.SeedSequence(BASE_SEED,
                                           spawn_key=(num_qubits, depth, index))
                )
                layers = sample_haar_random_layers(config, instance_rng)
                field = build_peak_weight_fn(config, layers)
                seen = {}
                for column, (base_index, _, _, theta, shifted, _, _) in enumerate(
                    points
                ):
                    if base_index not in seen:
                        seen[base_index] = float(field(theta))
                        values[index, base_index] = seen[base_index]
                    values[index, NUM_BASES + column] = float(field(shifted))

            resamples = np.random.default_rng(11).integers(
                NUM_INSTANCES, size=(BOOTSTRAP, NUM_INSTANCES)
            )
            for column, (base_index, kind, shift, _, _, overlap,
                         exact) in enumerate(points):
                a = values[:, base_index]
                b = values[:, NUM_BASES + column]
                measured = float(np.corrcoef(a, b)[0, 1])
                rows.append(dict(
                    n=num_qubits, depth=depth, base=base_index, kind=kind,
                    t=shift, overlap=overlap, corr=measured, exact=exact,
                    deep=deep_limit_corr(overlap, num_qubits),
                    se=bootstrap_stderr(a, b, resamples),
                ))
            print(f"n = {num_qubits}  tau_r = {depth:2d}  "
                  f"({time.perf_counter() - start:.0f} s, {len(points)} pairs "
                  f"x {NUM_INSTANCES} instances)", flush=True)

    print(f"\n{'n':>3} {'tau':>4} {'base':>4} {'kind':>7} {'t':>5} {'F':>7} "
          f"{'Corr':>8} {'exact':>8} {'resid':>8} {'se_boot':>8}")
    for row in rows:
        print(f"{row['n']:>3} {row['depth']:>4} {row['base']:>4} "
              f"{row['kind']:>7} {row['t']:>5.2f} {row['overlap']:>7.4f} "
              f"{row['corr']:>+8.4f} {row['exact']:>+8.4f} "
              f"{row['corr'] - row['exact']:>+8.4f} {row['se']:>8.4f}")

    residual = np.array([r["corr"] - r["exact"] for r in rows])
    stderr = np.array([r["se"] for r in rows])
    mean = residual.mean()
    sem = residual.std(ddof=1) / np.sqrt(len(residual))
    print(f"\n{len(rows)} points")
    print(f"residual vs exact kernel : {mean:+.4f} +/- {sem:.4f} "
          f"(median {np.median(residual):+.4f})")
    deep_residual = np.array([r["corr"] - r["deep"] for r in rows])
    print(f"residual vs deep limit   : {deep_residual.mean():+.4f} +/- "
          f"{deep_residual.std(ddof=1) / np.sqrt(len(deep_residual)):.4f} "
          "(the quantity check_kernel_robust.py reports)")
    for kind in DIRECTION_KINDS:
        subset = np.array([r["corr"] - r["exact"] for r in rows
                           if r["kind"] == kind])
        print(f"  kind {kind:>6}: {subset.mean():+.4f} +/- "
              f"{subset.std(ddof=1) / np.sqrt(len(subset)):.4f}")
    beyond_two = int((np.abs(residual) > 2 * stderr).sum())
    beyond_three = int((np.abs(residual) > 3 * stderr).sum())
    print(f"|residual| > 2 se_boot: {beyond_two}/{len(rows)} "
          f"(chance ~{0.046 * len(rows):.0f});  > 3 se_boot: {beyond_three}/"
          f"{len(rows)} (chance ~{0.0027 * len(rows):.0f})")


if __name__ == "__main__":
    main()
