"""Restart-count invariance of the block-ALS geometric measure at (8, 0).

geometric_measure.py runs twelve ALS restarts per point, and on the one
point where the first-rung prediction falls short — n = 8, instance 0,
where the optimizer misses the ALS value by 7.2e-4 — the shortfall is
attributed to the optimizer arm. That attribution needs the ALS arm to
be stable: this script reproduces the committed twelve-restart value by
replaying the rng stream of geometric_measure.main() up to that point,
then reruns block-ALS at 96 restarts from a fresh stream, and prints
both, so the invariance claim has a committed log.

Self-test: the replayed 12-restart value reproduces the committed
geometric_measure.log entry for (8, 0) to 1e-6.

Usage:
    python analysis/geometric_measure_restarts.py    # ~1 min
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometric_measure as gm  # noqa: E402
from peaked_circuits import CircuitConfig  # noqa: E402
from ceiling_bound import scrambled_state  # noqa: E402

COMMITTED_8_0 = 0.162661


def replay_committed_value() -> tuple[np.ndarray, float]:
    """Replay geometric_measure.main()'s rng stream through (8, 0)."""
    rng = np.random.default_rng(77)
    value, target_phi = float("nan"), None
    for n in (6, 8):
        for instance in (0, 1, 2):
            config = CircuitConfig(n, n, n // 2)
            phi = scrambled_state(
                config,
                np.random.default_rng(
                    np.random.SeedSequence(42, spawn_key=(n, instance))
                ),
            )
            prediction = gm.block_als(phi, n // 2, rng)
            if (n, instance) == (8, 0):
                return phi, prediction
    raise RuntimeError("point (8, 0) not reached")


def main() -> None:
    phi, twelve = replay_committed_value()
    assert abs(twelve - COMMITTED_8_0) < 1e-6, twelve
    print(f"self-test passed (replayed 12-restart value {twelve:.6f} "
          f"matches the committed {COMMITTED_8_0:.6f})")

    saved = gm.RESTARTS
    gm.RESTARTS = 96
    try:
        ninety_six = gm.block_als(phi, 4, np.random.default_rng(7))
    finally:
        gm.RESTARTS = saved
    print(f"n = 8, instance 0: 12 ALS restarts -> {twelve:.6f}")
    print(f"n = 8, instance 0: 96 ALS restarts -> {ninety_six:.6f}")
    print(f"96-restart shift: {ninety_six - twelve:+.1e}")


if __name__ == "__main__":
    main()
