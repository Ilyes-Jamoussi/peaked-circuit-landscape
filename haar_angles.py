"""Haar-distributed gate initialization for the ArbitraryUnitary parameterization.

The paper's verdict -- within the stable-local class and at the frozen budget,
nothing we measure reopens the first branch -- is quantified on a class defined
by a cadence: the per-step displacement is set by the schedule. The only
published competitor sits outside that definition in a way its own text hides.
Aaronson & Zhang (arXiv:2404.14493) announce Adam, but their code runs
``TNOptimizer(..., optimizer='L-BFGS-B')`` on Haar-initialized gates, so their
position with respect to the class is undetermined. The frozen protocol starts
every gate at theta ~ N(0, 0.1), i.e. at the identity; this module supplies the
other initialization, inside this repository's parameterization, so that the
two arms differ in one thing only.

The gate is ``qml.ArbitraryUnitary(angles, wires)`` and its convention is *not*
a single exponential of a Pauli sum. PennyLane 0.45.1 decomposes it into an
ordered product of fifteen Pauli rotations,

    U(theta) = E_15 ... E_2 E_1,     E_j = exp(-i theta_j P_j / 2),

over the two-qubit Pauli words in its own Gray-code order (see
``ARBITRARY_UNITARY_WORDS``). The angles are therefore *not* the Pauli
coefficients of one logarithm, and taking a matrix logarithm inverts nothing.
No closed form is known for this ordered product, so ``su4_to_arbitrary_angles``
inverts it numerically and exactly: fix the global phase, take the shortest
traceless logarithm, then follow the geodesic from the identity to the target
while integrating

    d theta / ds = J(theta)^-1 c_L ,   J[k, j] = tr(P_k S_j P_j S_j^*) / 4 ,

with S_j the suffix product E_15 ... E_{j+1}, correcting each predictor step by
Newton iterations on the exact equation. J is the analytic Jacobian of the
ordered product, so the corrector converges quadratically and the residual
reaches machine precision.

Two facts about that inversion belong in the paper rather than in a comment.
First, the map is not globally well conditioned: J degenerates on a measure-zero
set that the geodesic occasionally crosses, and about one draw in ten needs a
different SU(4) lift of the same physical gate, a few per thousand a restart
from a random point, and about one in three thousand a finer continuation on
top of both. Second, the recovered angles are canonical only
modulo winding; ``inversion_statistics`` reports both rates, and a campaign that
finds them elevated should not trust the arm.

Angles are returned wrapped to [-pi, pi). The wrap is exact, not an
approximation: E_j(t + 2 pi) = -E_j(t), so it changes the gate by a global
phase and leaves delta_{0^n} untouched.

Usage:
    python haar_angles.py                                  # self-tests, ~10 s
    python pq_experiment.py --num-qubits 12 --instance 0 --init-mode haar

Cost: about 16 ms per gate, so about one second per restart at n = 16, which is
negligible against the restart itself.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.linalg import schur
from scipy.stats import unitary_group

#: Pauli words of ``qml.ArbitraryUnitary``'s decomposition, in the Gray-code
#: order PennyLane emits them. Everything below is wrong if this order drifts,
#: so the self-tests re-derive it from the public matrix rather than trust it.
ARBITRARY_UNITARY_WORDS = (
    "XI", "YI", "ZI", "ZX", "IX", "XX", "YX", "YY",
    "ZY", "IY", "XY", "XZ", "YZ", "ZZ", "IZ",
)

PARAMETERS_PER_SU4_GATE = len(ARBITRARY_UNITARY_WORDS)

_ONE_QUBIT = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]]),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}
_PAULIS = np.stack(
    [np.kron(_ONE_QUBIT[word[0]], _ONE_QUBIT[word[1]])
     for word in ARBITRARY_UNITARY_WORDS]
)
#: Row k holds vec(P_k^T), so ``_TRACE_ROWS @ A.reshape(16)`` is (tr(P_k A))_k.
_TRACE_ROWS = _PAULIS.transpose(0, 2, 1).reshape(PARAMETERS_PER_SU4_GATE, 16)
_IDENTITY = np.eye(4, dtype=complex)

#: Continuation schedule. Thirty-two predictor steps with three Newton
#: corrections each leave nine draws in ten converging on the first path;
#: raising it buys little because the failures are conditioning, not step size.
_PATH_STEPS = 32
_NEWTON_STEPS = 3
_NEWTON_TOL = 1e-14
_ACCEPT_TOL = 1e-13
_RANDOM_PATHS = 8

#: Resolution of the retry, tried on every lift before the random paths.
#: Measured over 3000 draws, one target fails all four lifts at _PATH_STEPS and
#: clears at four times that. It is worth a retry rather than an exception: a
#: Haar restart at n = 16 draws 64 gates, so a per-gate failure rate of one in
#: three thousand aborts about one restart in fifty.
_REFINED_PATH_STEPS = 4 * _PATH_STEPS

_STATISTICS = {"draws": 0, "relifted": 0, "refined": 0, "restarted": 0, "wound": 0}


def reset_inversion_statistics() -> None:
    """Zero the inversion counters (they are per process)."""
    for key in _STATISTICS:
        _STATISTICS[key] = 0


def inversion_statistics() -> dict[str, float]:
    """Counts and rates of the conditioning failures of the inversion.

    ``relifted`` counts targets the shortest SU(4) lift could not reach, so a
    lift differing by a global phase was used instead; ``refined`` counts those
    that needed the finer continuation as well; ``restarted`` counts those that
    needed a path from a random point on top of that; ``wound`` counts
    solutions that left [-pi, pi) before wrapping, i.e. that the continuation
    reached the long way round. All three are properties of the ensemble, not
    of any one run, and belong in the arm's provenance.
    """
    draws = _STATISTICS["draws"]
    rates = {
        f"{name}_fraction": (_STATISTICS[name] / draws if draws else 0.0)
        for name in ("relifted", "refined", "restarted", "wound")
    }
    return {**_STATISTICS, **rates}


def arbitrary_unitary_matrix(angles: np.ndarray) -> np.ndarray:
    """The 4x4 matrix ``qml.ArbitraryUnitary(angles, wires)`` applies."""
    unitary = _IDENTITY
    for factor in _rotation_factors(np.asarray(angles, dtype=float)):
        unitary = factor @ unitary
    return unitary


def su4_to_arbitrary_angles(unitary: np.ndarray) -> np.ndarray:
    """Fifteen angles whose ``ArbitraryUnitary`` equals ``unitary`` up to phase.

    ``unitary`` is any 4x4 unitary. ``ArbitraryUnitary`` spans SU(4) only, so
    the global phase is not representable and is discarded; it is unobservable
    in delta_{0^n} anyway. Deterministic: the random paths of the fallback are
    seeded from the target itself, so repeated calls return the same angles.

    Raises ``RuntimeError`` if no representation is found, which no Haar draw
    has yet produced.
    """
    unitary = np.ascontiguousarray(unitary, dtype=complex)
    if unitary.shape != (4, 4):
        raise ValueError("su4_to_arbitrary_angles expects a 4x4 unitary.")
    if not np.allclose(unitary @ unitary.conj().T, _IDENTITY, atol=1e-10):
        # Otherwise the continuation just fails on every path and reports it as
        # a conditioning failure, which is a lie about the caller's input.
        raise ValueError("su4_to_arbitrary_angles expects a unitary matrix.")
    phases = np.linalg.det(unitary) ** 0.25 * 1j ** np.arange(4)
    lifts = unitary / phases[:, None, None]
    order = np.argsort([_shortest_logarithm(lift)[1] for lift in lifts])

    _STATISTICS["draws"] += 1
    for rank, index in enumerate(order):
        angles = _follow_path(
            np.zeros(PARAMETERS_PER_SU4_GATE), _IDENTITY, lifts[index]
        )
        if angles is not None:
            _STATISTICS["relifted"] += int(rank > 0)
            return _wrap(angles)

    for index in order:
        angles = _follow_path(
            np.zeros(PARAMETERS_PER_SU4_GATE),
            _IDENTITY,
            lifts[index],
            _REFINED_PATH_STEPS,
        )
        if angles is not None:
            _STATISTICS["relifted"] += 1
            _STATISTICS["refined"] += 1
            return _wrap(angles)

    _STATISTICS["relifted"] += 1
    _STATISTICS["refined"] += 1
    _STATISTICS["restarted"] += 1
    target = lifts[order[0]]
    rng = np.random.default_rng(
        int.from_bytes(
            hashlib.blake2b(unitary.tobytes(), digest_size=8).digest(), "big"
        )
    )
    for attempt in range(_RANDOM_PATHS):
        start = rng.normal(0.0, 1.0 + attempt, PARAMETERS_PER_SU4_GATE)
        angles = _follow_path(start, arbitrary_unitary_matrix(start), target)
        if angles is not None:
            return _wrap(angles)
    raise RuntimeError(
        "no ArbitraryUnitary angles found for this target after "
        f"{8 + _RANDOM_PATHS} continuation paths."
    )


def sample_haar_angles(num_gates: int, rng: np.random.Generator) -> np.ndarray:
    """Angles for ``num_gates`` independent Haar gates, flattened gate by gate.

    The layout matches ``_apply_peaking_section``: fifteen consecutive angles
    per gate, gates in brick-wall order.
    """
    if num_gates < 1:
        raise ValueError("num_gates must be >= 1.")
    angles = np.empty(num_gates * PARAMETERS_PER_SU4_GATE)
    for gate in range(num_gates):
        block = slice(
            gate * PARAMETERS_PER_SU4_GATE, (gate + 1) * PARAMETERS_PER_SU4_GATE
        )
        target = unitary_group.rvs(4, random_state=rng)
        angles[block] = su4_to_arbitrary_angles(target)
    return angles


def _wrap(angles: np.ndarray) -> np.ndarray:
    """Fold the angles into [-pi, pi); exact, since E_j(t + 2 pi) = -E_j(t)."""
    wrapped = (angles + np.pi) % (2.0 * np.pi) - np.pi
    _STATISTICS["wound"] += int(not np.allclose(wrapped, angles, rtol=0.0, atol=1e-9))
    return wrapped


def _rotation_factors(angles: np.ndarray) -> np.ndarray:
    """The fifteen factors E_j = cos(t/2) I - i sin(t/2) P_j, stacked."""
    half = angles / 2.0
    cosine, sine = np.cos(half)[:, None, None], np.sin(half)[:, None, None]
    return cosine * _IDENTITY - 1j * sine * _PAULIS


def _pauli_coefficients(generator: np.ndarray) -> np.ndarray:
    """Real c with ``generator = -i/2 sum_k c_k P_k`` for traceless anti-Hermitian."""
    return np.real(1j * (_TRACE_ROWS @ generator.reshape(16)) / 2.0)


def _product_jacobian(angles: np.ndarray) -> np.ndarray:
    """Jacobian of the ordered product, in Pauli coordinates of dU U^*.

    Differentiating U = E_15 ... E_1 in the j-th angle gives
    dU/dt_j = (-i/2) S_j P_j S_j^* U with S_j the suffix product, so the
    columns are the Pauli expansions of the rotated generators.
    """
    factors = _rotation_factors(angles)
    suffix = np.empty((PARAMETERS_PER_SU4_GATE, 4, 4), dtype=complex)
    running = _IDENTITY
    for index in range(PARAMETERS_PER_SU4_GATE - 1, -1, -1):
        suffix[index] = running
        running = running @ factors[index]
    rotated = suffix @ _PAULIS @ suffix.conj().transpose(0, 2, 1)
    return np.real(_TRACE_ROWS @ rotated.reshape(PARAMETERS_PER_SU4_GATE, 16).T) / 4.0


def _shortest_logarithm(unitary: np.ndarray) -> tuple[np.ndarray, float]:
    """Traceless anti-Hermitian L with exp(L) = unitary, of near-minimal norm.

    The principal phases sum to a multiple of 2 pi rather than to zero, and a
    logarithm with a trace is not in su(4) and cannot start the continuation.
    Unwinding the extreme phases restores tracelessness at the smallest cost.
    """
    triangular, basis = schur(unitary, output="complex")
    phases = np.angle(np.diag(triangular))
    winding = int(round(phases.sum() / (2.0 * np.pi)))
    if winding > 0:
        phases[np.argsort(phases)[-winding:]] -= 2.0 * np.pi
    elif winding < 0:
        phases[np.argsort(phases)[:-winding]] += 2.0 * np.pi
    logarithm = (basis * (1j * phases)) @ basis.conj().T
    return 0.5 * (logarithm - logarithm.conj().T), float(np.sum(phases ** 2))


def _geodesic(generator: np.ndarray, fraction: float) -> np.ndarray:
    """exp(fraction * generator) for anti-Hermitian ``generator``."""
    eigenvalues, basis = np.linalg.eigh(1j * generator)
    return (basis * np.exp(-1j * fraction * eigenvalues)) @ basis.conj().T


def _follow_path(
    start_angles: np.ndarray,
    start_unitary: np.ndarray,
    target: np.ndarray,
    path_steps: int = _PATH_STEPS,
) -> np.ndarray | None:
    """Continue from a known (angles, unitary) pair to ``target``, or give up.

    Returns None as soon as the Jacobian degenerates or the final residual
    misses ``_ACCEPT_TOL``; the caller then tries another path. Never returns
    an approximate answer, because a silently inexact initialization would
    make the arm measure a different ensemble than the one it reports.
    """
    generator, _ = _shortest_logarithm(target @ start_unitary.conj().T)
    direction = _pauli_coefficients(generator)
    angles = np.array(start_angles, dtype=float)
    try:
        for index in range(1, path_steps + 1):
            step = np.linalg.solve(_product_jacobian(angles), direction)
            angles += step / path_steps
            goal = _geodesic(generator, index / path_steps) @ start_unitary
            for _ in range(_NEWTON_STEPS):
                residual = _residual(goal, angles)
                if np.max(np.abs(residual)) < _NEWTON_TOL:
                    break
                angles += np.linalg.solve(_product_jacobian(angles), residual)
            if not np.all(np.isfinite(angles)):
                return None
        for _ in range(_NEWTON_STEPS):
            residual = _residual(target, angles)
            if np.max(np.abs(residual)) < _ACCEPT_TOL:
                return angles
            angles += np.linalg.solve(_product_jacobian(angles), residual)
    except np.linalg.LinAlgError:
        return None
    return angles if np.max(np.abs(_residual(target, angles))) < _ACCEPT_TOL else None


def _residual(target: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Pauli coordinates of log(target U(angles)^*): zero exactly on a solution."""
    gap, _ = _shortest_logarithm(target @ arbitrary_unitary_matrix(angles).conj().T)
    return _pauli_coefficients(gap)


def _projective_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Max entrywise gap after removing the unobservable global phase."""
    overlap = np.trace(left.conj().T @ right)
    aligned = right * np.conj(overlap / abs(overlap))
    return float(np.max(np.abs(left - aligned)))


def _self_tests() -> None:
    """Convention, exactness of the inversion, and the induced Haar moments."""
    import pennylane as qml

    probe = np.linspace(-2.0, 2.0, PARAMETERS_PER_SU4_GATE)
    reference = qml.matrix(qml.ArbitraryUnitary(probe, wires=[0, 1]))
    convention = float(np.max(np.abs(arbitrary_unitary_matrix(probe) - reference)))
    assert convention < 1e-13, (
        f"ArbitraryUnitary is not the ordered Pauli-rotation product assumed "
        f"here (gap {convention:.2e}); ARBITRARY_UNITARY_WORDS has drifted."
    )

    reset_inversion_statistics()
    rng = np.random.default_rng(20260806)
    worst = 0.0
    for _ in range(128):
        target = unitary_group.rvs(4, random_state=rng)
        angles = su4_to_arbitrary_angles(target)
        rebuilt = arbitrary_unitary_matrix(angles)
        worst = max(worst, _projective_distance(rebuilt, target))
    assert worst < 1e-12, f"reconstruction error {worst:.2e} exceeds 1e-12"

    sample = 384
    blocks = sample_haar_angles(sample, rng).reshape(-1, PARAMETERS_PER_SU4_GATE)
    gates = np.stack([arbitrary_unitary_matrix(block) for block in blocks])
    weight = float(np.mean(np.abs(gates[:, 0, 0]) ** 2))
    assert abs(weight - 0.25) < 0.04, f"mean |<00|U|00>|^2 = {weight:.4f}, expected 1/4"

    traces = np.einsum("iab,jba->ij", gates.conj().transpose(0, 2, 1), gates)
    potential = float(np.mean(np.abs(traces[~np.eye(sample, dtype=bool)]) ** 4))
    assert abs(potential - 2.0) < 0.06, (
        f"frame potential {potential:.4f}, expected 2 for a 2-design on SU(4)"
    )

    statistics = inversion_statistics()
    print(
        f"self-tests passed (convention gap {convention:.1e}; reconstruction "
        f"{worst:.1e} over {128} Haar draws; mean |<00|U|00>|^2 = {weight:.4f} "
        f"vs 1/4; frame potential {potential:.4f} vs 2)"
    )
    print(
        f"inversion over {statistics['draws']} draws: "
        f"{100 * statistics['relifted_fraction']:.1f}% needed another SU(4) lift, "
        f"{100 * statistics['refined_fraction']:.1f}% a finer continuation, "
        f"{100 * statistics['restarted_fraction']:.1f}% a random path, "
        f"{100 * statistics['wound_fraction']:.1f}% wound past [-pi, pi)"
    )


if __name__ == "__main__":
    _self_tests()
