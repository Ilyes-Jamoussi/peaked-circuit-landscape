"""Sanity tests for pq_experiment.py, and the guard on the frozen protocol.

The first seven tests are the original sanity checks: circuit geometry,
determinism, worker invariance, the two overlap identities, the agreement
between the state and projector qnodes, and settings validation.

The four added with the converged protocol exist to protect one claim. The
converged campaign re-measures reach at a longer budget and corrects the
archived ensembles for what the 400-step cap cost them; that correction only
means something if the frozen protocol did not move when the instrumentation
was added to its loop.

  * ``test_frozen_protocol_is_bit_identical`` compares a default run against
    an Adam loop written out inside this file. That loop, not the code under
    test, is the definition the archives were produced under. Both the cap and
    the early-stop path are compared, with ``np.array_equal``: an agreement to
    a few ulps would already mean the two grids no longer nest.
  * ``test_converged_nests_the_frozen_run`` checks the within-trajectory
    readout. Replaying the frozen stopping rule along a converged run must
    return exactly what a frozen run from the same seed returns; the whole
    truncation correction is that identity, and it is what frees the
    correction from matching seeds across archives.
  * ``test_converged_outcome_invariants`` checks the order relations the
    archive format promises, on both stopping paths.
  * ``test_restart_cache_roundtrip`` checks that a resumed run reloads its
    restarts instead of recomputing them, and that a cache written under one
    set of settings cannot be read back under another.

Everything runs at n = 4 with short budgets.

Usage:
    python test_pq.py        # ~30 s
    pytest test_pq.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from pq_experiment import (
    RestartOutcome,
    RestartSettings,
    az_scaling_config,
    build_state_fn,
    load_restart,
    optimize_one_restart,
    overlap_matrix,
    run_instance,
    settings_fingerprint,
    store_restart,
)
from peaked_circuits import (
    CircuitConfig,
    build_peak_weight_fn,
    sample_haar_random_layers,
)

TINY_SETTINGS = RestartSettings(max_steps=30, min_steps=5)
STOP_REASONS = {"cap", "rel_tol", "legacy_tol"}


def tiny_instance(num_workers: int = 1, base_seed: int = 7):
    """A cheap ensemble: n = 4, tau_r = 4, tau_p = 2, three restarts."""
    return run_instance(
        az_scaling_config(4),
        TINY_SETTINGS,
        num_restarts=3,
        base_seed=base_seed,
        instance_index=0,
        num_workers=num_workers,
    )


def tiny_landscape(base_seed: int = 3):
    """The n = 4 instance the protocol tests optimize on: config and delta."""
    config = az_scaling_config(4)
    haar_layers = sample_haar_random_layers(config, np.random.default_rng(base_seed))
    return config, build_peak_weight_fn(config, haar_layers)


def frozen_reference_run(peak_weight_fn, theta_init, settings):
    """The reproduction's Adam loop, written out instead of imported.

    This is the protocol the archived ensembles were produced under: Adam at a
    fixed stepsize, multiplied by ``decay_factor`` every ``decay_every`` steps,
    stopped once the cost change falls below the tolerance after ``min_steps``.
    Calling into pq_experiment for the reference would make the comparison
    circular, which is the one thing this test must not be.
    """
    theta = pnp.array(theta_init, requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=settings.learning_rate)

    def cost(parameters):
        return -peak_weight_fn(parameters)

    previous_cost = np.inf
    steps_taken = settings.max_steps
    for step in range(1, settings.max_steps + 1):
        theta, current_cost = optimizer.step_and_cost(cost, theta)
        current_cost = float(current_cost)
        if (
            step > settings.min_steps
            and abs(previous_cost - current_cost) < settings.convergence_tol
        ):
            steps_taken = step
            break
        previous_cost = current_cost
        if step % settings.decay_every == 0:
            optimizer.stepsize *= settings.decay_factor
    return np.asarray(theta), float(peak_weight_fn(theta)), steps_taken


def assert_converged_invariants(readout, peak_weights) -> None:
    """The order relations the .npz format guarantees by construction."""
    reasons = readout["stop_reasons"]
    running_max = readout["running_max"]
    weights = readout["ladder_weights"]
    assert set(reasons.tolist()) <= STOP_REASONS, set(reasons.tolist())
    assert np.all(readout["legacy_weight"] <= running_max)
    assert np.all(running_max <= peak_weights)  # polishing never loses ground
    assert np.all(readout["polish_gain"] >= 0.0)
    assert np.all(readout["legacy_stop_step"] >= 1)
    for row in weights:
        recorded = row[~np.isnan(row)]
        assert np.all(np.diff(recorded) >= 0.0), row  # it is a running maximum
        assert np.all(np.isnan(row[len(recorded):])), row  # NaN only pads


def test_az_scaling_config() -> None:
    config = az_scaling_config(12)
    assert (config.num_qubits, config.num_random_layers, config.num_peaking_layers) == (
        12,
        12,
        6,
    )
    try:
        az_scaling_config(9)
    except ValueError:
        pass
    else:
        raise AssertionError("odd n must be rejected when tau_p defaults to n/2")
    override = az_scaling_config(9, num_peaking_layers=4)
    assert override.num_peaking_layers == 4  # explicit tau_p bypasses the check


def test_overlap_matrix_properties() -> None:
    data = tiny_instance()
    overlaps = data.overlaps
    assert overlaps.shape == (3, 3)
    assert np.allclose(overlaps, overlaps.T)
    assert np.allclose(np.diag(overlaps), 1.0)
    assert np.all(overlaps >= 0.0) and np.all(overlaps <= 1.0 + 1e-12)
    assert np.all(data.peak_weights > data.baseline_peak_weight)


def test_determinism() -> None:
    first, second = tiny_instance(), tiny_instance()
    assert np.array_equal(first.thetas_init, second.thetas_init)
    assert np.array_equal(first.thetas_final, second.thetas_final)
    assert np.array_equal(first.overlaps, second.overlaps)


def test_worker_invariance() -> None:
    """Parallel and serial execution must produce identical ensembles."""
    serial, parallel = tiny_instance(num_workers=1), tiny_instance(num_workers=2)
    assert np.array_equal(serial.peak_weights, parallel.peak_weights)
    assert np.array_equal(serial.thetas_final, parallel.thetas_final)
    assert np.array_equal(serial.overlaps, parallel.overlaps)


def test_state_matches_projector() -> None:
    """The state-vector qnode and the projector qnode must agree on delta."""
    config = CircuitConfig(num_qubits=4, num_random_layers=3, num_peaking_layers=2)
    haar_layers = sample_haar_random_layers(config, np.random.default_rng(3))
    theta = np.random.default_rng(4).normal(0.0, 0.3, config.num_peaking_parameters)
    delta = float(build_peak_weight_fn(config, haar_layers)(theta))
    state = np.asarray(build_state_fn(config, haar_layers)(theta))
    assert np.isclose(delta, np.abs(state[0]) ** 2)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_overlap_matrix_of_known_states() -> None:
    """Overlaps of hand-built states come out exactly right."""
    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    states = np.stack([np.array([1.0, 0.0]), np.array([0.0, 1.0]), plus]).astype(
        complex
    )
    expected = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]])
    assert np.allclose(overlap_matrix(states), expected)


def test_restart_settings_validation() -> None:
    for invalid in (
        lambda: RestartSettings(max_steps=0),
        lambda: RestartSettings(learning_rate=0.0),
        lambda: RestartSettings(init_scale=-1.0),
        lambda: RestartSettings(ladder=(400, 200)),        # not increasing
        lambda: RestartSettings(ladder=(400, 400)),        # not strictly
        lambda: RestartSettings(max_steps=800, ladder=(400, 1600)),  # past the cap
        lambda: RestartSettings(learning_rate=0.05, min_learning_rate=0.1),
        lambda: RestartSettings(min_learning_rate=-1e-3),
        lambda: RestartSettings(rel_tol=-0.1),
        lambda: RestartSettings(polish_steps=-1),
        lambda: RestartSettings(init_mode="uniform"),
        lambda: RestartSettings(optimizer="rmsprop"),
    ):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    # The frozen protocol is the default, and the predicate that says so is
    # what decides whether an archive gets the converged keys appended.
    assert RestartSettings().is_frozen_protocol
    assert not RestartSettings(ladder=(10,), max_steps=10).is_frozen_protocol
    assert not RestartSettings(min_learning_rate=0.01).is_frozen_protocol
    assert not RestartSettings(polish_steps=1).is_frozen_protocol
    assert not RestartSettings(rel_tol=0.01).is_frozen_protocol


def test_frozen_protocol_is_bit_identical() -> None:
    """The archived protocol, against a loop written out in this file.

    Two settings: one that runs to the cap through three stepsize decays, one
    that leaves through the early-stop branch.
    """
    config, peak_weight_fn = tiny_landscape()
    cases = (
        RestartSettings(max_steps=40, min_steps=40, decay_every=10),
        RestartSettings(max_steps=60, min_steps=5, convergence_tol=1e-3),
    )
    early_stop_steps = None
    for settings in cases:
        assert settings.is_frozen_protocol
        outcome = optimize_one_restart(
            peak_weight_fn,
            config.num_peaking_parameters,
            settings,
            np.random.default_rng(11),
        )
        theta_init = np.random.default_rng(11).normal(
            0.0, settings.init_scale, config.num_peaking_parameters
        )
        assert np.array_equal(outcome.theta_init, theta_init)
        theta, weight, steps = frozen_reference_run(
            peak_weight_fn, theta_init, settings
        )
        assert np.array_equal(outcome.theta_final, theta)
        assert outcome.peak_weight == weight
        assert outcome.num_steps == steps
        assert outcome.stop_reason is None  # no readout under the frozen protocol
        if steps < settings.max_steps:
            early_stop_steps = steps
    # One case has to leave through the early-stop branch, or both cases test
    # the same path.
    assert early_stop_steps is not None, "the early-stop branch never fired"


def test_converged_nests_the_frozen_run() -> None:
    """The legacy readout must reproduce the frozen run, not approximate it."""
    config, peak_weight_fn = tiny_landscape()
    parameters = config.num_peaking_parameters
    frozen = RestartSettings()
    converged = RestartSettings(
        max_steps=200,
        ladder=(100, 200),
        min_learning_rate=0.05 * 2**-3,
        rel_tol=0.0,
        polish_steps=0,
        convergence_tol=0.0,
    )
    seed = 1
    frozen_run = optimize_one_restart(
        peak_weight_fn, parameters, frozen, np.random.default_rng(seed)
    )
    converged_run = optimize_one_restart(
        peak_weight_fn, parameters, converged, np.random.default_rng(seed)
    )
    assert np.array_equal(frozen_run.theta_init, converged_run.theta_init)
    # The frozen rule has to fire inside the converged budget for the identity
    # to be under test at all.
    assert frozen_run.num_steps < converged.max_steps
    assert converged_run.legacy_stop_step == frozen_run.num_steps
    assert converged_run.legacy_weight == frozen_run.peak_weight
    assert converged_run.running_max >= frozen_run.peak_weight


def test_converged_outcome_invariants() -> None:
    """Both stopping paths, through the ensemble the archive is written from."""
    config = az_scaling_config(4)
    common = dict(
        max_steps=40,
        min_steps=5,
        convergence_tol=0.0,
        decay_every=15,
        min_learning_rate=0.05 * 2**-2,
        ladder=(10, 20, 40),
        polish_steps=8,
    )
    # rel_tol = 0 can never fire against a non-decreasing running maximum, so
    # every restart runs to the cap; a huge rel_tol fires at the second rung,
    # which is what leaves the NaN padding to check.
    for rel_tol, expected in ((0.0, "cap"), (1e9, "rel_tol")):
        data = run_instance(
            config,
            RestartSettings(rel_tol=rel_tol, **common),
            num_restarts=3,
            base_seed=7,
            instance_index=0,
            num_workers=1,
        )
        readout = data.converged
        assert readout is not None
        assert set(readout["stop_reasons"].tolist()) == {expected}
        assert readout["ladder_weights"].shape == (3, 3)
        assert_converged_invariants(readout, data.peak_weights)
        filled = ~np.isnan(readout["ladder_weights"])
        if expected == "cap":
            assert filled.all()  # every rung was reached
        else:
            assert not filled.all()  # the rungs past the stop stay NaN


def test_restart_cache_roundtrip() -> None:
    """A resumed run reloads its restarts; a changed setting invalidates them."""
    outcome = RestartOutcome(
        theta_init=np.zeros(3),
        theta_final=np.ones(3),
        peak_weight=0.5,
        num_steps=7,
        seconds=1.25,
    )
    with tempfile.TemporaryDirectory() as cache:
        store_restart(cache, 0, outcome)
        assert (Path(cache) / "restart_0000.npz").exists(), (
            "store_restart left no archive: np.savez_compressed appends .npz "
            "to any name that does not end in it, so the file it writes is "
            "not the one os.replace then moves into place"
        )
        reloaded = load_restart(cache, 0)
        assert reloaded is not None
        assert np.array_equal(reloaded.theta_final, outcome.theta_final)
        assert (reloaded.num_steps, reloaded.seconds) == (7, 1.25)
        assert load_restart(cache, 1) is None

    config = az_scaling_config(4)
    settings = RestartSettings(max_steps=12, min_steps=3)
    other = RestartSettings(max_steps=12, min_steps=3, init_scale=0.2)
    assert settings_fingerprint(config, settings, 7, 0) != settings_fingerprint(
        config, other, 7, 0
    )
    with tempfile.TemporaryDirectory() as cache:
        run = dict(num_restarts=2, base_seed=7, instance_index=0, num_workers=1)
        first = run_instance(config, settings, restart_cache=cache, **run)
        second = run_instance(config, settings, restart_cache=cache, **run)
        assert np.array_equal(first.thetas_final, second.thetas_final)
        assert np.array_equal(first.peak_weights, second.peak_weights)
        assert np.array_equal(first.num_steps, second.num_steps)
        # restart_seconds is a wall-clock measurement: it can only come back
        # identical because the second pass read it instead of optimizing.
        assert np.array_equal(first.restart_seconds, second.restart_seconds)
        assert len(list(Path(cache).iterdir())) == 1
        run_instance(config, other, restart_cache=cache, **run)
        assert len(list(Path(cache).iterdir())) == 2


def test_worker_memory_model() -> None:
    """The pool is sized from a measurement, and the measurement is on record.

    The n = 16 catch-up was first launched with 96 workers on a 96 GB machine.
    One n = 16 restart holds 2.03 GB, so the pool asked for 195 GB; the kernel
    OOM-killed the campaign nine minutes in and it produced nothing. The rule
    in cloud/runner.py is what stands between that and a repeat, so the
    measured column it was fitted to is asserted here rather than left in a
    comment: peak resident set of one worker process, at the plateau.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "cloud"))
    from runner import memory_workers, total_memory_mb, worker_memory_mb

    measured = {8: 171.0, 10: 178.0, 12: 239.0, 14: 540.0, 16: 2030.0}
    for num_qubits, peak in measured.items():
        modelled = worker_memory_mb(num_qubits)
        assert abs(modelled - peak) / peak < 0.05, (
            f"the memory model gives {modelled:.0f} MB at n = {num_qubits} "
            f"against {peak:.0f} MB measured; re-fit it or the pool will be "
            f"sized wrong at the only size where that is fatal"
        )

    # The property that matters is not the fit but the bound: whatever the
    # pool is allowed on THIS machine, all of it running at once must still
    # fit on this machine. Checked wherever the physical size is readable,
    # which is every platform the campaign runs on.
    total = total_memory_mb()
    if total is not None:
        for jobs in (1, 2, 3):
            for num_qubits in measured:
                allowed = memory_workers({"n": num_qubits}, jobs)
                assert allowed >= 1
                demand = allowed * jobs * worker_memory_mb(num_qubits)
                assert demand <= total, (
                    f"{allowed} workers x {jobs} jobs at n = {num_qubits} "
                    f"want {demand / 1024:.0f} GB on a "
                    f"{total / 1024:.0f} GB machine"
                )

    # A job with no size (the analysis commands) is not memory-modelled, and
    # must not be silently throttled to one worker.
    assert memory_workers({}, 3) > 1000


if __name__ == "__main__":
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    failed = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as error:
            failed.append(name)
            print(f"{name}: FAILED  {error}")
        else:
            print(f"{name}: OK")
    if failed:
        raise SystemExit(
            f"{len(failed)}/{len(tests)} tests FAILED: {', '.join(failed)}"
        )
    print(f"all {len(tests)} tests passed")
