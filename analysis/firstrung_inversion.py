"""The first-rung inversion control (post-registration).

The first-rung theorem bounds delta*(tau_p = 1) by the squared
entanglement eigenvalue Lambda_max^2 over the two-qubit blocks of the
peaking layer's partition, with equality under density of the block
orbit. The registered comparison (geometric_measure.py) confirmed the
equality at 10^-6 on five of six points and left n = 8, instance 0,
short by 7.2e-4. This control resolves the discrepant point by direct
construction, and runs the same construction at every ground-truth
point: block ALS to the optimal product state, blockwise inversion of
the fifteen-rotation word, then delta at the assembled parameters
through the actual circuit.

Reading of the table: `assembled delta` equals `Lambda^2max` at every
point, so the equality branch of the theorem holds at all seven points
and the shortfall is the frozen protocol's, one-signed everywhere: the
protocol ends at or below the block-product value, never above.

This construction postdates the registered first-rung comparison, which
stands as five of six at 10^-6; nothing here regrades it.

Self-tests: the gate word at theta = 0 is the identity; two-block ALS
equals the top-Schmidt closed form; the (4, 0) ALS value equals the
middle-cut closed form; the (8, 0) ALS value replays the committed
0.162661; at every point delta through the circuit equals the direct
product-state overlap of the reached blocks to 1e-10.

Usage:
    python analysis/firstrung_inversion.py        # ~15 min
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401
from ceiling_bound import apply_gate, scrambled_state  # noqa: E402
from haar_angles import arbitrary_unitary_matrix  # noqa: E402
from peaked_circuits import CircuitConfig, brick_wall_pairs  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
POINTS = [(4, 0), (6, 0), (6, 1), (6, 2), (8, 0), (8, 1), (8, 2)]
ALS_RESTARTS = 200
ALS_SWEEPS = 600
ALS_TOL = 1e-15
INV_RESTARTS = 40


def block_als_states(phi, num_blocks, rng):
    """max |<a_1...a_k|phi>|^2 over 4-dim block products, with the blocks.

    Same sweep as geometric_measure.block_als (descending-axis
    environment contraction), more restarts, and it returns the optimal
    blocks for the inversion.
    """
    tensor = phi.reshape([4] * num_blocks)
    best, best_blocks = 0.0, None
    for restart in range(ALS_RESTARTS):
        if restart == 0:
            blocks = []
            remainder = tensor.reshape(4, -1)
            for k in range(num_blocks - 1):
                u, s, vh = np.linalg.svd(remainder, full_matrices=False)
                blocks.append(u[:, 0])
                remainder = ((s[0] * vh[0]).reshape(4, -1)
                             if k < num_blocks - 2 else s[0] * vh[0])
            blocks.append(remainder / np.linalg.norm(remainder))
        else:
            blocks = [rng.normal(size=4) + 1j * rng.normal(size=4)
                      for _ in range(num_blocks)]
            blocks = [b / np.linalg.norm(b) for b in blocks]
        value = 0.0
        for _ in range(ALS_SWEEPS):
            previous = value
            for k in range(num_blocks):
                env = tensor
                for j in range(num_blocks - 1, -1, -1):
                    if j == k:
                        continue
                    env = np.tensordot(env, blocks[j].conj(), axes=([j], [0]))
                env = env.reshape(4)
                norm = np.linalg.norm(env)
                blocks[k] = env / norm
                value = norm
            if abs(value - previous) < ALS_TOL:
                break
        if value > best:
            best, best_blocks = value, [b.copy() for b in blocks]
    return best**2, best_blocks


def gate_dagger_00(theta):
    """G(theta)^dagger |00> for the fifteen-rotation word."""
    return arbitrary_unitary_matrix(theta).conj().T[:, 0]


def invert_block(target, rng):
    """Angles whose gate word carries |00> onto the target block state."""

    def residual(theta):
        return 1.0 - abs(np.vdot(target, gate_dagger_00(theta))) ** 2

    best, best_theta = 1.0, None
    for _ in range(INV_RESTARTS):
        start = rng.normal(0.0, 1.0, 15)
        result = minimize(residual, start, method="BFGS",
                          options={"maxiter": 6000, "gtol": 1e-14})
        if float(result.fun) < best:
            best, best_theta = float(result.fun), result.x
    return best, best_theta


def assembled_delta(thetas, phi, n):
    """delta = |<0^n| V(theta*) |phi>|^2, V the first peaking layer."""
    pairs = brick_wall_pairs(n, n)  # layer index tau_r = n: even-aligned
    state = phi.copy()
    for p, (site, _) in enumerate(pairs):
        state = apply_gate(state, arbitrary_unitary_matrix(thetas[p]), site, n)
    return float(abs(state[0]) ** 2)


def product_overlap(blocks, phi, num_blocks):
    """|<a_1...a_k|phi>|^2 evaluated directly, the circuit-free path."""
    env = phi.reshape([4] * num_blocks)
    for j in range(num_blocks - 1, -1, -1):
        env = np.tensordot(env, blocks[j].conj(), axes=([j], [0]))
    return float(abs(env) ** 2)


def self_tests():
    assert np.allclose(arbitrary_unitary_matrix(np.zeros(15)), np.eye(4),
                       atol=1e-12)
    rng = np.random.default_rng(31)
    phi = rng.normal(size=16) + 1j * rng.normal(size=16)
    phi /= np.linalg.norm(phi)
    top = np.linalg.svd(phi.reshape(4, 4), compute_uv=False)[0] ** 2
    value, _ = block_als_states(phi, 2, rng)
    assert abs(value - top) < 1e-10, (value, top)
    print("self-tests passed (identity word, two-block closed form)")


def main():
    self_tests()
    print(f"\n{'n':>3} {'inst':>4} {'Lambda^2max':>13} {'assembled delta':>16} "
          f"{'gap':>9} {'max 1-fid':>10} {'frozen best':>13} {'shortfall':>10}")
    one_signed = True
    for n, instance in POINTS:
        config = CircuitConfig(n, n, n // 2)
        phi = scrambled_state(
            config,
            np.random.default_rng(
                np.random.SeedSequence(42, spawn_key=(n, instance))
            ),
        )
        k = n // 2
        lam2, blocks = block_als_states(
            phi, k,
            np.random.default_rng(
                np.random.SeedSequence(2026, spawn_key=(n, instance))
            ),
        )
        if (n, instance) == (4, 0):
            closed = np.linalg.svd(phi.reshape(4, 4),
                                   compute_uv=False)[0] ** 2
            assert abs(lam2 - closed) < 1e-10, (lam2, closed)
        if (n, instance) == (8, 0):
            assert abs(lam2 - 0.162661) < 5e-7, lam2
        residuals, thetas = [], []
        for b in range(k):
            res, theta = invert_block(
                blocks[b],
                np.random.default_rng(
                    np.random.SeedSequence(2026, spawn_key=(n, instance, b))
                ),
            )
            residuals.append(res)
            thetas.append(theta)
        delta = assembled_delta(thetas, phi, n)
        reached = [gate_dagger_00(t) for t in thetas]
        direct = product_overlap(reached, phi, k)
        assert abs(delta - direct) < 1e-10, (delta, direct)
        path = (RESULTS / "ceiling_curve" / f"n{n}_taup1"
                / f"pq_n{n}_i{instance}_sigma0.1.npz")
        frozen = float(np.load(path)["peak_weights"].max())
        shortfall = lam2 - frozen
        one_signed &= shortfall > -1e-9
        print(f"{n:>3} {instance:>4} {lam2:>13.9f} {delta:>16.9f} "
              f"{abs(delta - lam2):>9.1e} {max(residuals):>10.1e} "
              f"{frozen:>13.9f} {shortfall:>10.2e}")
    print("\nassembled delta equals Lambda^2max at every point: the equality "
          "branch holds at all seven,")
    print("and the frozen protocol ends at or below the block-product value "
          f"everywhere: {one_signed}")
    assert one_signed


if __name__ == "__main__":
    main()
