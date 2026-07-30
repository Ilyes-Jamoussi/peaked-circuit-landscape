"""Geometric measure of entanglement over two-qubit blocks (09, 3c).

delta*(tau_p = 1) is exactly the maximal product-state fidelity of the
scrambled state over the n/2 brick blocks (registered theorem, 09 3c).
Computed by rank-one block ALS: iteratively set each 4-dimensional block
vector to its normalized environment; restarts from the top Schmidt
vectors and random starts.

Self-tests: n = 4 equals the top Schmidt probability of the middle cut
(closed form); product-state input gives fidelity 1; ALS never exceeds
the min-cut top-1... (upper bound: top Schmidt probability of every
block bipartition).

Usage:
    python analysis/geometric_measure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401
from ceiling_bound import scrambled_state  # noqa: E402
from peaked_circuits import CircuitConfig  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
SWEEPS = 400
RESTARTS = 12
TOL = 1e-14


def block_als(phi: np.ndarray, num_blocks: int, rng) -> float:
    """max |<a_1...a_k|phi>| over product states of 4-dim blocks."""
    tensor = phi.reshape([4] * num_blocks)
    best = 0.0
    for restart in range(RESTARTS):
        if restart == 0:
            # top-Schmidt-style init: truncate sequentially to rank 1
            blocks = []
            remainder = tensor.reshape(4, -1)
            for _ in range(num_blocks - 1):
                u, s, vh = np.linalg.svd(remainder, full_matrices=False)
                blocks.append(u[:, 0])
                remainder = (s[0] * vh[0]).reshape(4, -1) if _ < num_blocks - 2 \
                    else s[0] * vh[0]
            blocks.append(remainder / np.linalg.norm(remainder))
        else:
            blocks = [rng.normal(size=4) + 1j * rng.normal(size=4)
                      for _ in range(num_blocks)]
            blocks = [b / np.linalg.norm(b) for b in blocks]
        value = 0.0
        for _ in range(SWEEPS):
            previous = value
            for k in range(num_blocks):
                env = tensor
                # Contract every block but k in DESCENDING axis order:
                # axes below the current one are untouched, so block j
                # always sits at axis j when its turn comes. (Bugfix
                # 2026-07-17: the first version contracted axis 0 for
                # j < k, silently permuting blocks for >= 3 blocks; the
                # 2-block self-test could not see it — caught because
                # the measured tau_p = 1 atom EXCEEDED the "bound".)
                for j in range(num_blocks - 1, -1, -1):
                    if j == k:
                        continue
                    env = np.tensordot(env, blocks[j].conj(), axes=([j], [0]))
                env = env.reshape(4)
                norm = np.linalg.norm(env)
                blocks[k] = env / norm
                value = norm
            if abs(value - previous) < TOL:
                break
        best = max(best, value)
    return best**2


def self_tests() -> None:
    rng = np.random.default_rng(31)
    # closed form at two blocks: top Schmidt probability
    phi = rng.normal(size=16) + 1j * rng.normal(size=16)
    phi /= np.linalg.norm(phi)
    top_schmidt = np.linalg.svd(phi.reshape(4, 4), compute_uv=False)[0] ** 2
    als = block_als(phi, 2, rng)
    assert abs(als - top_schmidt) < 1e-10, (als, top_schmidt)
    # product state -> fidelity 1
    a = rng.normal(size=4) + 1j * rng.normal(size=4)
    b = rng.normal(size=4) + 1j * rng.normal(size=4)
    product = np.kron(a / np.linalg.norm(a), b / np.linalg.norm(b))
    assert abs(block_als(product, 2, rng) - 1.0) < 1e-10
    # three blocks: bounded by every bipartition's top Schmidt probability
    phi3 = rng.normal(size=64) + 1j * rng.normal(size=64)
    phi3 /= np.linalg.norm(phi3)
    value = block_als(phi3, 3, rng)
    for cut in (4, 16):
        top = np.linalg.svd(phi3.reshape(cut, -1), compute_uv=False)[0] ** 2
        assert value <= top + 1e-9, (value, top)
    # three blocks, exact recovery: a KNOWN product state embedded in noise
    a, b, c = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(3))
    product3 = np.einsum("a,b,c->abc", a, b, c).reshape(-1)
    product3 /= np.linalg.norm(product3)
    mixture = 0.9 * product3 + 0.1 * phi3
    mixture /= np.linalg.norm(mixture)
    value = block_als(mixture, 3, rng)
    lower = abs(np.vdot(product3, mixture)) ** 2
    assert value >= lower - 1e-9, (value, lower)
    print("self-tests passed (closed form, product state, cut bounds, "
          "3-block recovery)")


def main() -> None:
    self_tests()
    rng = np.random.default_rng(77)
    print(f"\n{'n':>3} {'inst':>4} {'G-measure pred':>15} {'measured tau_p=1':>17}")
    for n in (6, 8):
        for instance in (0, 1, 2):
            config = CircuitConfig(n, n, n // 2)
            phi = scrambled_state(
                config,
                np.random.default_rng(
                    np.random.SeedSequence(42, spawn_key=(n, instance))
                ),
            )
            prediction = block_als(phi, n // 2, rng)
            path = (RESULTS / "ceiling_curve" / f"n{n}_taup1"
                    / f"pq_n{n}_i{instance}_sigma0.1.npz")
            measured = (float(np.load(path)["peak_weights"].max())
                        if path.exists() else float("nan"))
            print(f"{n:>3} {instance:>4} {prediction:>15.6f} {measured:>17.6f}")


if __name__ == "__main__":
    main()
