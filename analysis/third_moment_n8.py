"""m3 at n = 8 by certified truncation (registered prediction E4).

Reuses the S3 machinery of third_moment.py; configurations are processed
in decreasing |W| order and the truncation error is certified by
|X_c| <= 1: |error| <= sum of dropped |W|. Non-regression: the same policy
at n = 6 must reproduce the full exact value.

Usage:
    python analysis/third_moment_n8.py     # ~30-50 min
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401
from kernel_exact import probe_state_fn, spin_weights  # noqa: E402
from kernel_exact import exact_second_moment  # noqa: E402
from third_moment import (  # noqa: E402
    boundary_values,
    exact_third_moment,
    s3_weights,
)
from peaked_circuits import CircuitConfig  # noqa: E402

BASE_SEED = 4041
BUDGET = 6000


def truncated_third(
    weights: np.ndarray, state: np.ndarray, num_qubits: int, budget: int
) -> tuple[float, float]:
    """Returns (value, certified absolute error bound)."""
    order = np.argsort(-np.abs(weights))
    magnitudes = np.abs(weights[order])
    total = magnitudes.sum()
    kept = min(budget, int(np.count_nonzero(magnitudes)))
    configs = order[:kept]
    dropped = float(total - magnitudes[:kept].sum())
    values = boundary_values(state, num_qubits, configs)
    result = complex(np.dot(weights[configs], values))
    assert abs(result.imag) < 1e-9
    return result.real, dropped


def main() -> None:
    # Non-regression at n = 6: truncation must reproduce the full value.
    config6 = CircuitConfig(6, 6, 3)
    rng = np.random.default_rng(np.random.SeedSequence(BASE_SEED))
    theta6 = rng.normal(0.0, 0.1, config6.num_peaking_parameters)
    state6 = np.asarray(probe_state_fn(config6)(theta6))
    weights6 = s3_weights(6, 6, (6,))[6]
    full = exact_third_moment(weights6, state6, 6)
    trunc, bound = truncated_third(weights6, state6, 6, BUDGET)
    assert abs(trunc - full) <= bound + 1e-15, "certified bound violated"
    print(
        f"n = 6 non-regression: full {full:.6e}, truncated {trunc:.6e}, "
        f"bound {bound:.1e} -> ok"
    )

    num_qubits, depth = 8, 8
    dim = 2.0**num_qubits
    config = CircuitConfig(num_qubits, depth, num_qubits // 2)
    theta = rng.normal(0.0, 0.1, config.num_peaking_parameters)
    state = np.asarray(probe_state_fn(config)(theta))

    start = time.perf_counter()
    weights = s3_weights(num_qubits, depth, (depth,))[depth]
    print(
        f"S3 transfer n = 8: {np.count_nonzero(weights)} nonzero configs "
        f"[{time.perf_counter() - start:.0f} s]"
    )
    start = time.perf_counter()
    third, bound = truncated_third(weights, state, num_qubits, BUDGET)
    m3 = third / (6.0 / dim**3)
    m3_bound = bound / (6.0 / dim**3)
    second = exact_second_moment(
        spin_weights(num_qubits, depth), state, state, num_qubits
    )
    m2 = second / (2.0 / dim**2)
    print(
        f"n = 8, tau_r = 8: m2 = {m2:.3f} (exact), "
        f"m3 = {m3:.3f} +/- {m3_bound:.3f} (certified), "
        f"exponent ln m3/ln m2 = {np.log(m3) / np.log(m2):.2f} "
        f"[{time.perf_counter() - start:.0f} s]"
    )
    print("registered prediction E4: m3 in [2.0, 2.35]")


if __name__ == "__main__":
    main()
