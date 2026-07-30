"""The surf test (V2; revised estimator, null corrected 2026-07-16).

Around found solutions theta* (from the P(q) ensembles) and around random
control points, measures the normalized local mean

    E_hat(t) = (D - 1) * < [delta(theta0 + t u) - F * delta(theta0)]
                           / ((1 - F) * (1 - delta0)) >_u

with F the exact probe overlap per perturbation. The (1 - delta0) factor
is mandatory: under the deep-limit kernel the conditional mean around a
point of peak delta0 is E[delta'|delta0] = (1-F)/(D-1) + delta0 (DF-1)/(D-1),
so the raw residual (delta' - F delta0)/(1-F) has expectation
(1 - delta0)/(D - 1) -- the residual state mass, not 1/D. The estimator
above therefore reads exactly 1 around ANY point of a typical field,
whatever its peak. (The first version omitted the (1 - delta0) division;
its "3-5x depletion around solutions" was entirely that factor.)

Registered prediction (E3-revised): NO excess of solutions over controls
at F ~ 0, since the exact anatomy found no scale separation; a fat-cell
envelope at the probed separations would push solution curves above 1.

Usage:
    python analysis/surf_test.py     # ~6 min
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401
from kernel_exact import probe_state_fn  # noqa: E402
from peaked_circuits import (  # noqa: E402
    CircuitConfig,
    build_peak_weight_fn,
    sample_haar_random_layers,
)

DATASETS = ((8, 0), (8, 1), (10, 0))
POINTS_PER_KIND = 8
DIRECTIONS = 16
SHIFTS = (0.1, 0.2, 0.3, 0.45, 0.7, 1.0)
EPSILON = 0.2
BASE_SEED = 7077
RESULTS = Path(__file__).resolve().parent.parent / "results" / "pq"


def main() -> None:
    rng = np.random.default_rng(np.random.SeedSequence(BASE_SEED))
    for num_qubits, instance in DATASETS:
        dim = 2.0**num_qubits
        data = np.load(RESULTS / f"pq_n{num_qubits}_i{instance}_sigma0.1.npz")
        thetas, deltas = data["thetas_final"], data["peak_weights"]
        near = np.flatnonzero(deltas >= (1 - EPSILON) * deltas.max())
        chosen = rng.choice(near, size=POINTS_PER_KIND, replace=False)

        config = CircuitConfig(num_qubits, num_qubits, num_qubits // 2)
        layers = sample_haar_random_layers(
            config,
            np.random.default_rng(
                np.random.SeedSequence(42, spawn_key=(num_qubits, instance))
            ),
        )
        field = build_peak_weight_fn(config, layers)
        probe = probe_state_fn(config)
        num_parameters = config.num_peaking_parameters

        points = [("solution", thetas[i], float(deltas[i])) for i in chosen]
        for _ in range(POINTS_PER_KIND):
            theta0 = rng.normal(0.0, 0.1, num_parameters)
            points.append(("control", theta0, float(field(theta0))))

        curves = {"solution": [], "control": []}
        for kind, theta0, delta0 in points:
            base_probe = np.asarray(probe(theta0))
            row = []
            for t in SHIFTS:
                estimates = []
                for _ in range(DIRECTIONS):
                    u = rng.normal(0.0, 1.0, num_parameters)
                    u /= np.linalg.norm(u)
                    shifted = theta0 + t * np.sqrt(num_parameters) * u
                    overlap = float(
                        np.abs(np.vdot(base_probe, np.asarray(probe(shifted)))) ** 2
                    )
                    value = float(field(shifted))
                    residual = (value - overlap * delta0) / (1.0 - overlap)
                    estimates.append(
                        residual * (dim - 1.0) / (1.0 - delta0)
                    )
                row.append(float(np.mean(estimates)))
            curves[kind].append(row)

        print(f"\nn = {num_qubits}, instance {instance} "
              f"({POINTS_PER_KIND} solutions vs {POINTS_PER_KIND} controls)")
        print(f"{'t':>6} {'solutions':>10} {'controls':>10} {'excess':>8}")
        for column, t in enumerate(SHIFTS):
            sol = np.mean([row[column] for row in curves["solution"]])
            ctl = np.mean([row[column] for row in curves["control"]])
            print(f"{t:>6.2f} {sol:>10.3f} {ctl:>10.3f} {sol - ctl:>+8.3f}")


if __name__ == "__main__":
    main()
