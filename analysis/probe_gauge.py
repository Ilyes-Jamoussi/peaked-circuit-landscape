"""Derived gauge of the objective: probe-ray corank 3S + 9B + 4E, and the band.

The objective delta = |<w|phi>|^2 depends on theta only through the ray of
the probe w = V(theta)^dag |0^n>: every direction that fixes that ray is
exactly flat, whatever it does to the output state V U_r |0^n>. Sec. II B
counts this gauge in closed form. Call a qubit *closed* at a gate of V if
no later gate of V acts on it, B the number of gates with two closed legs
(the bra-boundary layer) and E the number with one (the edge gates of the
layer before, when the boundary layer misses the chain ends). Each closed
pair contributes 9 = 15 - dim CP^3 flat directions (only the ray of the
gate's <00| row survives in w), each closed leg 4 = 15 - 11 (only the
<0| x I block survives, a point of the Stiefel manifold V_2(C^4) modulo
its block phase), on top of the 3S directions that fix V itself:

    corank(probe-ray Jacobian) >= 3S + 9B + 4E,
    B = n/2, E = 0 (tau_p odd);  B = n/2 - 1, E = 2 (tau_p even)

for the even-n, even-start brickwall convention. At the operating
configuration (n, tau_p) = (8, 4): 60 + 27 + 8 = 95. At tau_p = 1 the
count leaves rank 3n = (n/2) dim CP^3, the brick-product manifold of the
first-rung theorem.

Five claims are tested, in increasing strength:

  (a) Saturation. The measured probe-ray corank equals the count at
      Haar-random points of every configuration of the campaign grid
      (the same points as gauge_dimension.py, whose output-state coranks
      are re-asserted alongside), instance-free by construction. The
      kernel is counted at the machine-zero threshold 1e-12 x sigma_max
      and the bracketing singular pair is printed: the kernel floor sits
      at ~1e-15 relative, while the smallest nonzero singular value dips
      to ~1e-9 at one shallow depth-scan point.
  (b) At the solutions. The corank at the archived best restarts of
      (8, 4), (10, 5), (12, 6) equals 95, 153, 215, before and after an
      L-BFGS-B polish of the Adam stop (a stationarity diagnostic,
      outside the frozen protocol; no reach level quoted anywhere rests
      on it).
  (c) The flat band is the count. At the polished stops the Hessian
      carries exactly 3S + 9B + 4E eigenvalues at the numerical floor,
      separated from the rest by a relative gap above 10^3.
  (d) Direction by direction. Along the band the probe ray is stationary
      (first-order speed at the alignment floor); along every other
      eigenvector it moves faster by two to six orders of magnitude; the
      R_z dressing directions are inside the boundary count and move the
      output state at rank n.
  (e) The stall artifact. At the Adam stop the band's residual scale is
      a size-independent fraction of the stopping gradient across the
      four (8, 4) atoms; polishing removes it without moving delta at
      the atom.

Self-tests: closed form for (B, E) against direct enumeration on the grid
and odd depths; the local kernel dimensions 9 and 4 in su(4) by SVD;
probe derivative states against finite differences of w; the tau_p = 1
brick-product rank 3n; exact invariance of delta and of the probe ray
under a finite boundary R_z dressing re-solved through the word, with the
output state moving; the ray-space cap at (4, 2).

Usage:
    python analysis/probe_gauge.py     # ~25 min, dominated by (12, 6)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401  (wires up the reproduction repo path)
from ceiling_bound import scrambled_state  # noqa: E402
from gauge_dimension import (  # noqa: E402
    BASE_SEED,
    GRID,
    closed_form,
    corank,
    embed,
    solve_word,
)
from hessian_solutions import (  # noqa: E402
    gradient,
    hessian,
    output_and_derivatives,
)
from init_statistics import (  # noqa: E402
    PAULIS,
    apply_two_qubit,
    gate_matrix,
    peaking_gates,
    probe_and_derivatives,
    probe_and_derivatives_cheap,
)
from peaked_circuits import CircuitConfig  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def closing_gates(config: CircuitConfig):
    """(gate, closed qubits) per gate after which a leg is never touched."""
    sites = peaking_gates(config)
    found = []
    for g, site in enumerate(sites):
        legs = tuple(
            q for q in (site, site + 1)
            if not any(q in (s, s + 1) for s in sites[g + 1:])
        )
        if legs:
            found.append((g, legs))
    return found


def boundary_counts(config: CircuitConfig):
    """(B, E) by enumeration: gates with two / one closed legs."""
    closing = closing_gates(config)
    return (sum(1 for _, legs in closing if len(legs) == 2),
            sum(1 for _, legs in closing if len(legs) == 1))


def closed_form_boundary(n: int, tau_p: int):
    """B = n/2, E = 0 (tau_p odd); B = n/2 - 1, E = 2 (tau_p even)."""
    if tau_p % 2 == 1:
        return n // 2, 0
    return n // 2 - 1, 2


def probe_count(n: int, tau_p: int) -> int:
    b, e = closed_form_boundary(n, tau_p)
    return 3 * closed_form(n, tau_p) + 9 * b + 4 * e


def probe_corank(config: CircuitConfig, theta: np.ndarray,
                 with_gap: bool = False):
    """Corank of the probe-ray Jacobian at the machine-zero threshold.

    The kernel sits at the numerical floor (~1e-15 of the top singular
    value), while the smallest genuinely nonzero singular value dips to
    ~1e-9 at isolated random points of the shallow depth scans; the
    threshold 1e-12 x max separates the two by orders of magnitude
    either way, and the bracketing pair is returned to exhibit it."""
    w, dw = probe_and_derivatives(config, theta)
    projected = dw - np.outer(dw @ w.conj(), w)
    real_map = np.concatenate([projected.real, projected.imag], axis=1)
    singular = np.sort(np.linalg.svd(real_map.T, compute_uv=False))
    rank = int(np.sum(singular > 1e-12 * singular[-1]))
    corank = config.num_peaking_parameters - rank
    if with_gap:
        return corank, float(singular[-rank - 1]), float(singular[-rank])
    return corank


def output_state(config: CircuitConfig, theta: np.ndarray, phi: np.ndarray):
    sites = peaking_gates(config)
    per_gate = theta.reshape(len(sites), 15)
    vec = phi
    for g, site in enumerate(sites):
        vec = apply_two_qubit(vec, gate_matrix(per_gate[g]), site,
                              config.num_qubits)
    return vec


def polish(config: CircuitConfig, theta: np.ndarray, phi: np.ndarray,
           gtol: float = 1e-7):
    """L-BFGS-B on -delta from the archived stop, to ||grad|| < gtol."""

    def objective(point):
        grad, delta = gradient(config, point, phi)
        return -delta, -grad

    point = theta
    for _ in range(3):
        point = minimize(objective, point, jac=True, method="L-BFGS-B",
                         options={"maxiter": 5000, "ftol": 1e-18,
                                  "gtol": 1e-14}).x
        grad, delta = gradient(config, point, phi)
        if np.linalg.norm(grad) < gtol:
            break
    norm = float(np.linalg.norm(grad))
    assert norm < gtol, norm
    return point, float(delta), norm


def ray_speeds(config: CircuitConfig, theta: np.ndarray, vectors: np.ndarray):
    """First-order probe-ray speed ||(1 - w w^dag) dw . v|| per column."""
    w, dw = probe_and_derivatives(config, theta)
    projected = dw - np.outer(dw @ w.conj(), w)
    real_map = np.concatenate([projected.real, projected.imag], axis=1)
    return np.linalg.norm(vectors.T @ real_map, axis=1)


def probe_kernel_basis(config: CircuitConfig, theta: np.ndarray):
    """Orthonormal basis of the probe-ray Jacobian's kernel in theta
    space (its left null space), machine-exact by SVD."""
    w, dw = probe_and_derivatives(config, theta)
    projected = dw - np.outer(dw @ w.conj(), w)
    real_map = np.concatenate([projected.real, projected.imag], axis=1)
    u, singular, _ = np.linalg.svd(real_map, compute_uv=True)
    rank = int(np.sum(singular > 1e-12 * singular.max()))
    return u[:, rank:]  # (P, corank)


def state_motion_rank(config: CircuitConfig, theta: np.ndarray,
                      phi: np.ndarray, band: np.ndarray):
    """Rank of the projective output-state motion restricted to the band."""
    psi, dpsi = output_and_derivatives(config, theta, phi)
    projected = dpsi - np.outer(dpsi @ psi.conj(), psi)
    real_map = np.concatenate([projected.real, projected.imag], axis=1)
    singular = np.linalg.svd(band.T @ real_map, compute_uv=False)
    rank = int(np.sum(singular > 1e-8 * singular.max()))
    return rank, singular


def rz_state_rank(config: CircuitConfig, psi: np.ndarray):
    """Rank of the projected Z_j psi: independent state motions of the
    boundary R_z dressing."""
    n = config.num_qubits
    rows = []
    for j in range(n):
        signs = 1.0 - 2.0 * ((np.arange(2**n) >> (n - 1 - j)) & 1)
        vec = signs * psi
        vec = vec - np.vdot(psi, vec) * psi
        rows.append(np.concatenate([vec.real, vec.imag]))
    singular = np.linalg.svd(np.array(rows), compute_uv=False)
    rank = int(np.sum(singular > 1e-10 * singular.max()))
    return rank, singular


def dressed_theta(config: CircuitConfig, theta: np.ndarray,
                  alphas: np.ndarray) -> np.ndarray:
    """Absorb the bra-boundary dressing (x)_j R_z(alpha_j) into the closing
    gates and re-solve each fifteen-rotation word exactly."""
    sites = peaking_gates(config)
    per_gate = theta.reshape(len(sites), 15).copy()
    for g, legs in closing_gates(config):
        left = np.eye(4, dtype=complex)
        for q in legs:
            rz = np.diag([np.exp(-0.5j * alphas[q]),
                          np.exp(0.5j * alphas[q])])
            left = embed(rz, q - sites[g]) @ left
        angles, residual = solve_word(left @ gate_matrix(per_gate[g]),
                                      per_gate[g])
        assert residual < 1e-20, residual
        per_gate[g] = angles
    return per_gate.ravel()


def kernel_dimension(constraint_rows) -> int:
    """Nullity of a real linear system on su(4), basis -i/2 P_a."""
    matrix = []
    for a in range(15):
        generator = -0.5j * PAULIS[a]
        entries = constraint_rows(generator)
        matrix.append(np.concatenate([entries.real, entries.imag]))
    singular = np.linalg.svd(np.array(matrix).T, compute_uv=False)
    return 15 - int(np.sum(singular > 1e-12 * singular.max()))


def self_tests() -> None:
    # (a) closed form for (B, E) against enumeration, grid + odd depths
    for n, tau_p in GRID + [(6, 4), (10, 3), (12, 5)]:
        config = CircuitConfig(n, n, tau_p)
        assert boundary_counts(config) == closed_form_boundary(n, tau_p), \
            (n, tau_p)

    # (b) the two local kernel dimensions in su(4): a closed pair keeps
    # only the ray of the <00| row (9 = 15 - 6), a closed leg only the
    # <0| x I block up to phase (4 = 15 - 11)
    bra = np.zeros(4, dtype=complex)
    bra[0] = 1.0

    def pair_rows(generator):
        row = bra @ generator
        return row[1:]  # components off the <00| ray

    block = np.zeros((2, 4), dtype=complex)
    block[0, 0] = block[1, 1] = 1.0  # <0| x I in the (closed, open) slots
    flat = block.ravel() / np.sqrt(2.0)

    def leg_rows(generator):
        image = (block @ generator).ravel()
        return image - (flat.conj() @ image) * flat  # off the block ray

    assert kernel_dimension(pair_rows) == 9
    assert kernel_dimension(leg_rows) == 4

    # (c) probe derivative states against 4th-order finite differences
    rng = np.random.default_rng(23)
    config = CircuitConfig(4, 4, 2)
    theta = rng.normal(0.0, 0.3, config.num_peaking_parameters)
    w, dw = probe_and_derivatives(config, theta)
    for a in (0, 11, 29, config.num_peaking_parameters - 1):
        h = 1e-3
        stencil = []
        for shift in (-2, -1, 1, 2):
            point = theta.copy()
            point[a] += shift * h
            stencil.append(probe_and_derivatives_cheap(config, point))
        numeric = (stencil[0] - 8 * stencil[1] + 8 * stencil[2]
                   - stencil[3]) / (12 * h)
        assert np.linalg.norm(numeric - dw[a]) < 1e-9, a

    # (d) tau_p = 1: the probe ray ranges over the brick-product manifold
    # of the first-rung theorem, rank 3n
    config = CircuitConfig(8, 8, 1)
    point = rng.uniform(0.0, 2.0 * np.pi, config.num_peaking_parameters)
    assert probe_count(8, 1) == 36
    assert probe_corank(config, point) == 36  # rank 60 - 36 = 24 = 3n

    # (e) finite boundary R_z dressing: delta and the probe ray exactly
    # invariant, the output state not
    config = CircuitConfig(8, 8, 4)
    theta = rng.uniform(0.0, 2.0 * np.pi, config.num_peaking_parameters)
    phi = scrambled_state(
        config, np.random.default_rng(np.random.SeedSequence(42,
                                                             spawn_key=(8, 0)))
    )
    alphas = rng.uniform(0.5, 2.0, 8)
    dressed = dressed_theta(config, theta, alphas)
    w = probe_and_derivatives_cheap(config, theta)
    w_dressed = probe_and_derivatives_cheap(config, dressed)
    assert abs(abs(np.vdot(w, w_dressed)) - 1.0) < 1e-12  # same ray
    delta = abs(np.vdot(w, phi)) ** 2
    delta_dressed = abs(np.vdot(w_dressed, phi)) ** 2
    assert abs(delta - delta_dressed) < 1e-12
    psi = output_state(config, theta, phi)
    psi_dressed = output_state(config, dressed, phi)
    moved = 1.0 - abs(np.vdot(psi, psi_dressed)) ** 2
    assert moved > 1e-3, moved

    # (f) the ray-space cap never binds for the probe on the grid
    for n, tau_p in GRID:
        config = CircuitConfig(n, n, tau_p)
        image = config.num_peaking_parameters - probe_count(n, tau_p)
        assert image <= 2 * 2**n - 2, (n, tau_p)

    print("self-tests passed (boundary closed form on the grid + odd "
          "depths, su(4) kernels 9 and 4, probe derivatives, tau_p = 1 "
          f"rank 24 = 3n, R_z dressing exact with state moved {moved:.3f}, "
          "ray-space cap)")


def band_measurements(config, theta, phi, expected, label):
    """Polished Hessian: threshold ladder, gap, ray speeds."""
    start = time.perf_counter()
    h_matrix, asym = hessian(config, theta, phi)
    eigval, eigvec = np.linalg.eigh(h_matrix)
    order = np.argsort(np.abs(eigval))
    absv = np.abs(eigval)[order]
    ladder = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    counts = "/".join(str(int(np.sum(absv < tol))) for tol in ladder)
    below, above = absv[expected - 1], absv[expected]
    assert below < 1e-7 and above / below > 1e3, (below, above)

    speeds = ray_speeds(config, theta, eigvec)
    flat = speeds[order[:expected]].max()
    moving = speeds[order[expected:]].min()
    assert flat < 1e-5 and moving / flat > 1e2, (flat, moving)

    print(f"    {label}: band {counts} at thresholds 1e-8..1e-2, "
          f"|mu| gap {below:.1e} -> {above:.1e}, asym {asym:.1e}")
    print(f"      probe-ray speed: flat <= {flat:.2e}, "
          f"moving >= {moving:.2e}  [{time.perf_counter() - start:.0f} s]")


def main() -> None:
    self_tests()

    print("\n(a) probe-ray corank at Haar-random points vs 3S + 9B + 4E")
    print(f"{'(n, tau_p)':>11} {'P':>5} {'3S':>5} {'B':>3} {'E':>3} "
          f"{'predicted':>10} {'corank_w':>9} {'kernel gap':>18} "
          f"{'corank_psi':>11}")
    for n, tau_p in GRID:
        start = time.perf_counter()
        config = CircuitConfig(n, n, tau_p)
        big_p = config.num_peaking_parameters
        b, e = closed_form_boundary(n, tau_p)
        predicted = probe_count(n, tau_p)
        state_predicted = max(3 * closed_form(n, tau_p),
                              big_p - (2 * 2**n - 2), 0)
        phi = scrambled_state(
            config,
            np.random.default_rng(np.random.SeedSequence(42, spawn_key=(n, 0))),
        )
        rng = np.random.default_rng(BASE_SEED + 100 * n + tau_p)
        points = 2 if (n, tau_p) == (8, 4) else 1
        measured_w, measured_psi, gaps = [], [], []
        for _ in range(points):
            point = rng.uniform(0.0, 2.0 * np.pi, big_p)
            kernel, below, above = probe_corank(config, point, with_gap=True)
            assert above / below > 1e3, (n, tau_p, below, above)
            measured_w.append(kernel)
            gaps.append((below, above))
            measured_psi.append(corank(config, point, phi))
        assert all(m == predicted for m in measured_w), (n, tau_p, measured_w)
        assert all(m == state_predicted for m in measured_psi), \
            (n, tau_p, measured_psi)
        shown_w = "/".join(str(m) for m in measured_w)
        shown_psi = "/".join(str(m) for m in measured_psi)
        below, above = gaps[0]
        print(f"{f'({n}, {tau_p})':>11} {big_p:>5} "
              f"{3 * closed_form(n, tau_p):>5} {b:>3} {e:>3} "
              f"{predicted:>10} {shown_w:>9} "
              f"{f'{below:.0e} -> {above:.0e}':>18} {shown_psi:>11} "
              f"[{time.perf_counter() - start:.0f} s]")
    print("  the (4, 2) probe rays: a compact image of dimension "
          f"{45 - probe_count(4, 2)} in a ray space of dimension 30")

    print("\n(b) at the archived best restarts, before and after polishing")
    print("  polishing is a stationarity diagnostic, outside the frozen "
          "protocol; no reach level rests on it")
    stall_ratios = []
    polished_deltas = {}
    contexts = {}
    polished_points = {8: [], 10: [], 12: []}
    for n in (8, 10, 12):
        tau_p = n // 2
        config = CircuitConfig(n, n, tau_p)
        expected = probe_count(n, tau_p)
        phi = scrambled_state(
            config,
            np.random.default_rng(np.random.SeedSequence(42, spawn_key=(n, 0))),
        )
        contexts[n] = (config, phi, expected)
        data = np.load(RESULTS / "pq" / f"pq_n{n}_i0_sigma0.1.npz")
        order = np.argsort(data["peak_weights"])[::-1]
        chosen = (list(order[:4]) + [order[100]]) if n == 8 else list(order[:2])
        kind = "atom set + one mid-band" if n == 8 else "best restarts"
        print(f"  (n, tau_p) = ({n}, {tau_p}), {kind}, "
              f"target corank {expected}:")
        for rank_pos, index in enumerate(chosen):
            start = time.perf_counter()
            theta = data["thetas_final"][index]
            grad, delta = gradient(config, theta, phi)
            before = probe_corank(config, theta)
            if n == 8 and rank_pos < 4:
                h_matrix, _ = hessian(config, theta, phi)
                absv = np.sort(np.abs(np.linalg.eigvalsh(h_matrix)))
                ratio = float(np.median(absv[:expected])
                              / np.linalg.norm(grad))
                stall_ratios.append(ratio)
            point, polished_delta, norm = polish(config, theta, phi)
            after = probe_corank(config, point)
            assert before == after == expected, (index, before, after)
            assert polished_delta >= delta - 1e-12
            print(f"    sol {index:>3}: delta {delta:.6f} -> "
                  f"{polished_delta:.6f}, |grad| "
                  f"{np.linalg.norm(grad):.2e} -> {norm:.2e}, "
                  f"|dtheta| {np.linalg.norm(point - theta):.2e}, "
                  f"corank_w {before}/{after} "
                  f"[{time.perf_counter() - start:.0f} s]")
            if n != 8 or rank_pos < 4:
                polished_points[n].append((index, point))
            if rank_pos == 0:
                polished_deltas[n] = (float(delta), polished_delta)

    print("\n(c, d) the flat band at the polished stops is the count, "
          "direction by direction")
    for n in (8, 10, 12):
        config, phi, expected = contexts[n]
        print(f"  (n, tau_p) = ({n}, {n // 2}), expected {expected}:")
        for index, point in polished_points[n]:
            band_measurements(config, point, phi, expected, f"sol {index}")

    print("\n(e) what the band moves: the boundary count, and its R_z floor")
    for n in (8, 10, 12):
        config, phi, expected = contexts[n]
        point = polished_points[n][0][1]
        gauge = 3 * closed_form(n, n // 2)
        kernel = probe_kernel_basis(config, point)
        assert kernel.shape[1] == expected, kernel.shape
        rank, singular = state_motion_rank(config, point, phi, kernel)
        assert rank == expected - gauge, (n, rank)
        psi = output_state(config, point, phi)
        psi = psi / np.linalg.norm(psi)
        rz_rank, rz_singular = rz_state_rank(config, psi)
        assert rz_rank == n, (n, rz_rank)
        print(f"  ({n}, {n // 2}): probe-kernel state-motion rank {rank} "
              f"= 9B + 4E = {expected - gauge} (gap "
              f"{singular[rank - 1]:.1e} -> {singular[rank]:.1e}), "
              f"R_z motions rank {rz_rank} = n, sigma_min "
              f"{rz_singular.min():.2f}")

    print("\n(f) the stall artifact at the Adam stop, four (8, 4) atoms")
    shown = "/".join(f"{r:.3f}" for r in stall_ratios)
    spread = max(stall_ratios) / min(stall_ratios)
    assert spread < 3.0, stall_ratios
    print(f"  median band |mu| over ||grad||: {shown} -- a constant "
          f"fraction (spread x{spread:.2f}); polishing removes it")

    print("\n(g) provenance of the peakedness values near the polished stops")
    for n, budget_line in ((10, "0.4632"), (12, "0.4081")):
        archived, polished_value = polished_deltas[n]
        print(f"  n = {n}, instance 0 -- protocols, not readings of one "
              "number:")
        print(f"    Adam 400 steps, frozen protocol, best of 200   "
              f"{archived:.6f}  results/pq/pq_n{n}_i0_sigma0.1.npz")
        print(f"    L-BFGS-B polish of that point (this script)    "
              f"{polished_value:.6f}  stationarity diagnostic only")
        print(f"    Adam 1600 steps, 16 matched seeds              "
              f"{budget_line}    analysis/step_budget_control.log")
        if n == 10:
            print("    converged ladder protocol                      "
                  "0.4632566 REGISTRATION-CONVERGED.md")
        print("    none of these is delta*.")

    print("\nall checks passed: the flat band is the objective's gauge, "
          "3S + 9B + 4E")


if __name__ == "__main__":
    main()
