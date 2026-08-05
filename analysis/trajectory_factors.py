"""Commitment-time factors of the recorded trajectories.

The trajectory archive (trajectories.npz, written by trajectories.py)
stores, for every recorded restart, the objective trace delta_t and the
endpoint overlap q_fin(t) = |<psi_t|psi_final>|^2. This script derives
the three commitment factors the manuscript quotes: with
t_commit the first step at which q_fin exceeds 0.5, t90 and t50 the
first steps at which delta reaches 90% and 50% of its final value, it
reports the medians of t_commit/t90 and t_commit/t50 over the restarts
of each size. trajectories.log carries only the commit fraction, the
cells crossed, and ln N_visited; none of those is a time ratio, so the
factors need their own committed log.

Self-tests: a synthetic monotone trace with a known commitment step
recovers its ratios exactly, and a trace whose value starts at its
final level (t90 = 0) is excluded rather than divided by zero.

Usage:
    python analysis/trajectory_factors.py    # seconds, reads trajectories.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def factors(deltas: np.ndarray, q_final: np.ndarray):
    """Return (t_commit/t90, t_commit/t50) for one restart, or None."""
    final = deltas[-1]
    committed = np.flatnonzero(q_final > 0.5)
    above90 = np.flatnonzero(deltas >= 0.9 * final)
    above50 = np.flatnonzero(deltas >= 0.5 * final)
    if not (len(committed) and len(above90) and len(above50)):
        return None
    t_commit, t90, t50 = committed[0], above90[0], above50[0]
    if t90 == 0 or t50 == 0:
        return None
    return t_commit / t90, t_commit / t50


def self_tests() -> None:
    ramp = np.linspace(0.0, 1.0, 101)
    q = (np.arange(101) >= 90).astype(float)
    pair = factors(ramp, q)
    assert pair is not None and abs(pair[0] - 1.0) < 1e-12, pair
    assert abs(pair[1] - 90.0 / 50.0) < 1e-12, pair
    flat = np.ones(101)
    assert factors(flat, q) is None  # value at final level from step 0


def main() -> None:
    self_tests()
    print("self-tests passed (synthetic ratios exact, flat trace excluded)")
    data = np.load(HERE / "trajectories.npz")
    for num_qubits, num_restarts in ((8, 12), (12, 8)):
        r90, r50 = [], []
        for restart in range(num_restarts):
            pair = factors(
                data[f"n{num_qubits}_r{restart}_deltas"],
                data[f"n{num_qubits}_r{restart}_qfinal"],
            )
            if pair is not None:
                r90.append(pair[0])
                r50.append(pair[1])
        print(
            f"n = {num_qubits}: {len(r90)} restarts, "
            f"median t_commit/t90 = {np.median(r90):.2f}, "
            f"median t_commit/t50 = {np.median(r50):.2f}"
        )


if __name__ == "__main__":
    main()
