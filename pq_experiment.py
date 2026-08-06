"""P(q) measurement on the peaked-circuit optimization landscape.

First experiment of the research program in README.md (Section 3.4): for a
fixed random-circuit instance, run many independently initialized
optimizations of the peaking layers and record *every* final solution, not
just the best of the batch. The pairwise overlap distribution P(q) of the
resulting solution ensemble is the standard empirical diagnostic for
solution-space clustering (Gamarnik, arXiv:2109.14409), measured here, to our
knowledge for the first time, on the circuit-synthesis landscape of Aaronson
& Zhang (arXiv:2404.14493).

Protocol, per random-circuit instance:

  * geometry follows the paper's scaling regime tau_r = n, tau_p = n/2, the
    regime of the 1.189^-n law (arXiv:2404.14493, Section 3, Fig. 3c);
  * ``--restarts`` independent Adam runs with the reproduction's
    hyperparameters, each from its own initialization
    theta ~ N(0, ``--init-scale``); the default scale 0.1 matches the
    reproduction protocol, and larger scales serve as a control against
    clustering artifacts of the common near-identity starting region;
  * q_ij = |<psi_i|psi_j>|^2 for every pair of final output states, stored
    as a dense overlap matrix (the states themselves can be rebuilt from the
    saved angles).

Builds on the reproduction, expected as a sibling directory or at
$PEAKED_REPRO_PATH: https://github.com/Ilyes-Jamoussi/peaked-circuits-pennylane

Usage:
    python pq_experiment.py --pilot                     # timing estimate
    python pq_experiment.py --num-qubits 8 --instance 0
    python pq_experiment.py --num-qubits 12 --instance 0 --init-scale 1.0
"""

from __future__ import annotations

import os

# One BLAS thread per process: restarts are parallelized at the process
# level, and oversubscribed threads slow every worker down.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import hashlib
import json
import logging
import multiprocessing
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

def _locate_repro() -> Path:
    """Reproduction repo: PEAKED_REPRO_PATH, else the nearest enclosing sibling."""
    override = os.environ.get("PEAKED_REPRO_PATH")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for base in here.parents:
        candidate = base / "peaked-circuits-pennylane"
        if candidate.is_dir():
            return candidate
    return here.parent.parent / "peaked-circuits-pennylane"


REPRO_PATH = _locate_repro()
sys.path.insert(0, str(REPRO_PATH))
# Spawned workers re-import by module name; keep this file's directory on the
# path so the optional siblings (haar_angles, lbfgs_restart) resolve there too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pennylane as qml  # noqa: E402
from pennylane import numpy as pnp  # noqa: E402

from peaked_circuits import (  # noqa: E402
    PARAMETERS_PER_SU4_GATE as PARAMETERS_PER_GATE,
    CircuitConfig,
    _apply_peaking_section,
    _apply_random_section,
    build_peak_weight_fn,
    random_section_peak_weight,
    sample_haar_random_layers,
)

logger = logging.getLogger(__name__)

#: n values of the first measurement pass; within the range of the paper's
#: Fig. 3c (n = 8 to 16), so the delta scaling can be cross-checked.
GRID_QUBIT_COUNTS = (8, 10, 12, 14)


def az_scaling_config(
    num_qubits: int,
    num_random_layers: int | None = None,
    num_peaking_layers: int | None = None,
) -> CircuitConfig:
    """Circuit geometry of the 1.189^-n regime: tau_r = n, tau_p = n/2."""
    if num_random_layers is None:
        num_random_layers = num_qubits
    if num_peaking_layers is None:
        if num_qubits % 2:
            raise ValueError("tau_p = n/2 requires an even number of qubits.")
        num_peaking_layers = num_qubits // 2
    return CircuitConfig(num_qubits, num_random_layers, num_peaking_layers)


#: The frozen protocol's own stopping rule, kept as constants so that a
#: converged run can report where that rule *would* have fired without
#: carrying its tolerances. Never read these as defaults; they exist only
#: for the within-trajectory legacy readout of ``optimize_one_restart``.
FROZEN_MAX_STEPS = 400
FROZEN_MIN_STEPS = 100
FROZEN_CONVERGENCE_TOL = 1e-8

#: Code appended to the restart spawn key for each non-default init mode, so
#: that the existing ensembles (mode "normal") keep their exact seeds.
INIT_MODE_CODES = {"haar": 1}

#: The converged protocol, applied by ``--converged``.
#:
#: Why a floor rather than a stretched decay: keeping ``decay_every`` at 300
#: means steps 1..400 of a converged run are bit-identical to the frozen run
#: from the same seed, so the two grids nest. Without the floor the stepsize
#: reaches 0.05 * 2^-21 by step 6400 and the optimizer freezes instead of
#: converging, which is what makes the existing 1600-step control stall.
#:
#: Values measured by the scale pilot at n = 10, instance 0, four restarts,
#: rungs to 6400 with the stopping rule disabled. The gain per doubling ran
#: 3.24%, 3.47%, 0.003%, 0.001%: the run is converged by rung 3200, and the
#: geometric residual beyond it is below 1e-5. ``rel_tol = 0.002`` therefore
#: stops one rung after the gain collapses. Floors of 0.05*2^-3 and 0.05*2^-6
#: reach the same delta to 1e-6 -- the parameters differ by 0.079, the value
#: does not -- so the choice is not critical inside that range. The polish
#: gained 0.0000% at that point, which is itself worth reporting: the
#: converged run already stops at a stationary point. It is kept, at half the
#: budget first considered, because it costs little and closes the objection
#: that the returned parameters are not critical.
#:
#: The larger sizes are slower to converge and the GCP pilot re-measures all
#: five before REGISTRATION-CONVERGED.md is committed.
CONVERGED_PROTOCOL: dict[str, object] = {
    "max_steps": 12800,
    "ladder": (400, 800, 1600, 3200, 6400, 12800),
    "min_learning_rate": 0.05 * 2**-3,
    "rel_tol": 0.002,
    "polish_steps": 400,
    "convergence_tol": 0.0,
}


@dataclass(frozen=True)
class RestartSettings:
    """Per-restart Adam settings; defaults match the reproduction protocol.

    ``max_steps`` defaults to 400 rather than the reproduction's 2000: P(q)
    needs an ensemble of near-optimal solutions, not the polished optimum,
    and the reproduction's log shows the peak weight plateauing there.

    The last five fields are the converged protocol (2026-08). Their defaults
    leave the frozen protocol bit-for-bit unchanged: ``ladder`` empty disables
    every instrumented branch, and ``min_learning_rate = 0`` makes the decay
    line ``max(stepsize * factor, 0.0)``, which is the identity on positive
    stepsizes. ``is_frozen_protocol`` is the single predicate that decides.
    """

    max_steps: int = 400
    learning_rate: float = 0.05
    decay_every: int = 300
    decay_factor: float = 0.5
    convergence_tol: float = 1e-8
    min_steps: int = 100
    init_scale: float = 0.1
    optimizer: str = "adam"  # robustness matrix (2026-07-17); default = protocol
    min_learning_rate: float = 0.0  # floor under the decay; 0 = no floor
    ladder: tuple[int, ...] = ()  # steps at which the running max is recorded
    rel_tol: float = 0.0  # stop when a doubling gains less than this
    polish_steps: int = 0  # final anneal from the best point visited
    init_mode: str = "normal"  # "normal" | "haar"

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.learning_rate <= 0:
            raise ValueError("max_steps and learning_rate must be positive.")
        if self.init_scale < 0:
            raise ValueError("init_scale must be >= 0.")
        if self.optimizer not in ("adam", "sgd", "lbfgs"):
            raise ValueError("optimizer must be 'adam', 'sgd' or 'lbfgs'.")
        if self.init_mode not in ("normal",) + tuple(INIT_MODE_CODES):
            raise ValueError(f"unknown init_mode {self.init_mode!r}.")
        if self.min_learning_rate < 0 or self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must lie in [0, learning_rate].")
        if self.rel_tol < 0 or self.polish_steps < 0:
            raise ValueError("rel_tol and polish_steps must be >= 0.")
        if list(self.ladder) != sorted(set(self.ladder)):
            raise ValueError("ladder must be strictly increasing.")
        if self.ladder and self.ladder[-1] > self.max_steps:
            raise ValueError("ladder must not exceed max_steps.")

    @property
    def is_frozen_protocol(self) -> bool:
        """True when this run is the frozen protocol, instrumentation off."""
        return (
            self.min_learning_rate == 0.0
            and not self.ladder
            and self.rel_tol == 0.0
            and self.polish_steps == 0
            and self.init_mode == "normal"
            and self.optimizer in ("adam", "sgd")
        )


@dataclass(frozen=True)
class RestartOutcome:
    """Final solution of one independently initialized Adam run.

    The fields after ``seconds`` are the converged protocol's readout and stay
    ``None`` under the frozen protocol. ``legacy_stop_step`` and
    ``legacy_weight`` record where the frozen stopping rule *would* have fired
    on this very trajectory, which turns the truncation deficit into a
    within-trajectory identity instead of a comparison between archives.
    """

    theta_init: np.ndarray
    theta_final: np.ndarray
    peak_weight: float
    num_steps: int
    seconds: float
    stop_reason: str | None = None
    ladder_weights: np.ndarray | None = None
    running_max: float | None = None
    polish_gain: float | None = None
    legacy_stop_step: int | None = None
    legacy_weight: float | None = None


@dataclass(frozen=True)
class RestartTask:
    """Everything a worker process needs to run one restart."""

    circuit_config: CircuitConfig
    haar_layers: list
    settings: RestartSettings
    base_seed: int
    instance_index: int
    restart_index: int
    cache_dir: str | None = None


def settings_fingerprint(
    circuit_config: CircuitConfig,
    settings: RestartSettings,
    base_seed: int,
    instance_index: int,
) -> str:
    """Short hash of *everything* that determines a restart's outcome.

    The restart cache is keyed on this. Keying on (n, instance) alone would
    let a cache left over from an earlier protocol iteration be silently
    reused under different settings, which no downstream test could detect.
    """
    payload = {
        "num_qubits": circuit_config.num_qubits,
        "num_random_layers": circuit_config.num_random_layers,
        "num_peaking_layers": circuit_config.num_peaking_layers,
        "base_seed": base_seed,
        "instance_index": instance_index,
        **{field.name: getattr(settings, field.name) for field in fields(settings)},
    }
    payload["ladder"] = list(settings.ladder)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cache_path(cache_dir: str, restart_index: int) -> Path:
    return Path(cache_dir) / f"restart_{restart_index:04d}.npz"


def store_restart(cache_dir: str, index: int, outcome: RestartOutcome) -> None:
    """Persist one finished restart, atomically.

    A preemption in the middle of ``savez`` would otherwise leave a truncated
    file that the next boot reads as a completed restart.
    """
    path = _cache_path(cache_dir, index)
    payload = {
        "theta_init": outcome.theta_init,
        "theta_final": outcome.theta_final,
        "peak_weight": outcome.peak_weight,
        "num_steps": outcome.num_steps,
        "seconds": outcome.seconds,
        "instrumented": outcome.stop_reason is not None,
    }
    if outcome.stop_reason is not None:
        payload.update(
            stop_reason=outcome.stop_reason,
            ladder_weights=outcome.ladder_weights,
            running_max=outcome.running_max,
            polish_gain=outcome.polish_gain,
            legacy_stop_step=outcome.legacy_stop_step,
            legacy_weight=outcome.legacy_weight,
        )
    # Write through a handle: given a path, savez appends ".npz" when the name
    # does not already end in it, which would rename the temporary file out
    # from under the replace below.
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def load_restart(cache_dir: str, index: int) -> RestartOutcome | None:
    """Read a cached restart, or None if it is absent or unreadable."""
    path = _cache_path(cache_dir, index)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as blob:
            base = dict(
                theta_init=blob["theta_init"],
                theta_final=blob["theta_final"],
                peak_weight=float(blob["peak_weight"]),
                num_steps=int(blob["num_steps"]),
                seconds=float(blob["seconds"]),
            )
            if not bool(blob["instrumented"]):
                return RestartOutcome(**base)
            return RestartOutcome(
                **base,
                stop_reason=str(blob["stop_reason"]),
                ladder_weights=blob["ladder_weights"],
                running_max=float(blob["running_max"]),
                polish_gain=float(blob["polish_gain"]),
                legacy_stop_step=int(blob["legacy_stop_step"]),
                legacy_weight=float(blob["legacy_weight"]),
            )
    except (OSError, ValueError, KeyError, EOFError) as error:
        logger.warning("ignoring unreadable cache entry %s (%s)", path, error)
        return None


def restart_rng(task: RestartTask) -> np.random.Generator:
    """Deterministic per-restart generator, independent of scheduling.

    The spawn key includes the init scale so that runs at different scales
    draw statistically independent initializations rather than radially
    rescaled copies of one another.

    The key is only extended for a non-default ``init_mode``: appending an
    element unconditionally would change every seed in the existing archives.
    """
    key = (
        task.circuit_config.num_qubits,
        task.instance_index,
        task.restart_index,
        int(round(task.settings.init_scale * 1000)),
    )
    if task.settings.init_mode != "normal":
        key += (INIT_MODE_CODES[task.settings.init_mode],)
    return np.random.default_rng(np.random.SeedSequence(task.base_seed, spawn_key=key))


def draw_initial_parameters(
    num_parameters: int,
    settings: RestartSettings,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initial angles for one restart, per ``settings.init_mode``."""
    if settings.init_mode == "normal":
        return rng.normal(0.0, settings.init_scale, num_parameters)
    if settings.init_mode == "haar":
        from haar_angles import sample_haar_angles  # local: optional dependency

        if num_parameters % PARAMETERS_PER_GATE:
            raise ValueError("Haar init needs a whole number of two-qubit gates.")
        return sample_haar_angles(num_parameters // PARAMETERS_PER_GATE, rng)
    raise ValueError(f"unknown init_mode {settings.init_mode!r}.")


def optimize_one_restart(
    peak_weight_fn,
    num_parameters: int,
    settings: RestartSettings,
    rng: np.random.Generator,
) -> RestartOutcome:
    """One Adam run from a fresh initialization; returns its final solution.

    Same loop as the reproduction's ``maximize_peak_weight`` (learning-rate
    step decay, early stopping on the per-step cost change), except that the
    solution is returned unconditionally instead of competing in a batch.

    Under the converged protocol (``settings.ladder`` non-empty) the loop also
    carries an instrumented readout. It is written so that the sequence of
    ``step_and_cost`` calls, and therefore the trajectory, is *identical* to
    the frozen protocol's for as long as the two settings agree: every added
    statement reads state that the loop already computes. ``step_and_cost``
    returns the cost at the parameters it was *given*, so ``-current_cost`` is
    delta at the previous iterate and the running maximum costs no evaluation.
    """
    if settings.optimizer == "lbfgs":
        from lbfgs_restart import optimize_one_restart_lbfgs

        return optimize_one_restart_lbfgs(
            peak_weight_fn, num_parameters, settings, rng
        )

    theta_init = draw_initial_parameters(num_parameters, settings, rng)
    theta = pnp.array(theta_init, requires_grad=True)
    if settings.optimizer == "adam":
        optimizer = qml.AdamOptimizer(stepsize=settings.learning_rate)
    else:
        optimizer = qml.GradientDescentOptimizer(stepsize=settings.learning_rate)

    def cost(parameters):
        return -peak_weight_fn(parameters)

    instrumented = bool(settings.ladder)
    rungs = set(settings.ladder)
    ladder_weights: list[float] = []
    running_max, theta_running_max = -np.inf, theta_init
    previous_rung_max: float | None = None
    legacy_stop_step: int | None = None
    legacy_weight: float | None = None
    legacy_pending = False
    legacy_previous_cost = np.inf

    start = time.perf_counter()
    previous_cost = np.inf
    steps_taken = settings.max_steps
    stop_reason = "cap"
    for step in range(1, settings.max_steps + 1):
        theta_before = theta
        theta, current_cost = optimizer.step_and_cost(cost, theta)
        current_cost = float(current_cost)

        if instrumented:
            # cost was evaluated at theta_before, so this is delta there.
            delta_before = -current_cost
            if delta_before > running_max:
                running_max = delta_before
                theta_running_max = np.array(theta_before, copy=True)
            if legacy_pending:  # delta at the iterate the frozen run kept
                legacy_weight = delta_before
                legacy_pending = False
            if legacy_stop_step is None:
                if (
                    step > FROZEN_MIN_STEPS
                    and abs(legacy_previous_cost - current_cost)
                    < FROZEN_CONVERGENCE_TOL
                ) or step >= FROZEN_MAX_STEPS:
                    legacy_stop_step, legacy_pending = step, True
            legacy_previous_cost = current_cost

        if (
            step > settings.min_steps
            and abs(previous_cost - current_cost) < settings.convergence_tol
        ):
            steps_taken, stop_reason = step, "legacy_tol"
            break
        previous_cost = current_cost
        if step % settings.decay_every == 0:
            optimizer.stepsize = max(
                optimizer.stepsize * settings.decay_factor, settings.min_learning_rate
            )

        if instrumented and step in rungs:
            ladder_weights.append(running_max)
            if previous_rung_max is not None and previous_rung_max > 0:
                if running_max / previous_rung_max - 1.0 < settings.rel_tol:
                    steps_taken, stop_reason = step, "rel_tol"
                    break
            previous_rung_max = running_max

    if not instrumented:
        return RestartOutcome(
            theta_init=theta_init,
            theta_final=np.asarray(theta),
            peak_weight=float(peak_weight_fn(theta)),
            num_steps=steps_taken,
            seconds=time.perf_counter() - start,
        )

    final_weight = float(peak_weight_fn(theta))
    if final_weight > running_max:
        running_max, theta_running_max = final_weight, np.asarray(theta)
    if legacy_weight is None:  # the frozen rule fired on the last iteration
        legacy_weight = final_weight
    if legacy_stop_step is None:
        legacy_stop_step = steps_taken

    theta_best, best_weight = theta_running_max, running_max
    polish_gain = 0.0
    if settings.polish_steps:
        theta_best, best_weight = _polish(
            peak_weight_fn, cost, theta_running_max, settings
        )
        polish_gain = best_weight / running_max - 1.0

    return RestartOutcome(
        theta_init=theta_init,
        theta_final=np.asarray(theta_best),
        peak_weight=best_weight,
        num_steps=steps_taken,
        seconds=time.perf_counter() - start,
        stop_reason=stop_reason,
        ladder_weights=np.array(ladder_weights, dtype=float),
        running_max=running_max,
        polish_gain=polish_gain,
        legacy_stop_step=legacy_stop_step,
        legacy_weight=legacy_weight,
    )


def _polish(
    peak_weight_fn, cost, theta_start: np.ndarray, settings: RestartSettings
) -> tuple[np.ndarray, float]:
    """Anneal from the best point visited toward a stationary point.

    Starts at the floored learning rate and halves eight times over the
    budget, with no floor, so the iterate settles instead of hovering. Returns
    the best (parameters, delta) seen, so polishing can never lose ground.
    """
    theta = pnp.array(theta_start, requires_grad=True)
    stepsize = settings.min_learning_rate or settings.learning_rate
    optimizer = (
        qml.AdamOptimizer(stepsize=stepsize)
        if settings.optimizer == "adam"
        else qml.GradientDescentOptimizer(stepsize=stepsize)
    )
    decay_every = max(1, settings.polish_steps // 8)
    best_theta, best_weight = np.asarray(theta_start), float(peak_weight_fn(theta_start))
    for step in range(1, settings.polish_steps + 1):
        theta_before = theta
        theta, current_cost = optimizer.step_and_cost(cost, theta)
        if -float(current_cost) > best_weight:
            best_weight = -float(current_cost)
            best_theta = np.array(theta_before, copy=True)
        if step % decay_every == 0:
            optimizer.stepsize *= 0.5
    final_weight = float(peak_weight_fn(theta))
    if final_weight > best_weight:
        best_theta, best_weight = np.asarray(theta), final_weight
    return best_theta, best_weight


def run_restart_task(task: RestartTask) -> tuple[int, RestartOutcome]:
    """Worker entry point: rebuild the circuit and run one restart."""
    peak_weight_fn = build_peak_weight_fn(task.circuit_config, task.haar_layers)
    outcome = optimize_one_restart(
        peak_weight_fn,
        task.circuit_config.num_peaking_parameters,
        task.settings,
        restart_rng(task),
    )
    if task.cache_dir is not None:
        store_restart(task.cache_dir, task.restart_index, outcome)
    return task.restart_index, outcome


def build_state_fn(config: CircuitConfig, haar_layers: list):
    """Return theta -> C(theta)|0^n>, the full output state, for overlaps."""
    device = qml.device("default.qubit", wires=config.num_qubits)

    @qml.qnode(device)
    def state_fn(theta):
        _apply_random_section(config, haar_layers)
        _apply_peaking_section(config, theta)
        return qml.state()

    return state_fn


def overlap_matrix(states: np.ndarray) -> np.ndarray:
    """Pairwise overlaps q_ij = |<psi_i|psi_j>|^2 of stacked states (R, 2^n)."""
    gram = states @ states.conj().T
    return np.abs(gram) ** 2


@dataclass(frozen=True)
class InstanceData:
    """Solution ensemble of one circuit instance, ready to save."""

    thetas_init: np.ndarray
    thetas_final: np.ndarray
    peak_weights: np.ndarray
    num_steps: np.ndarray
    restart_seconds: np.ndarray
    argmax_indices: np.ndarray
    argmax_weights: np.ndarray
    overlaps: np.ndarray
    baseline_peak_weight: float
    elapsed_seconds: float
    #: Converged-protocol readout; None under the frozen protocol.
    converged: dict[str, np.ndarray] | None = None


def _collect_converged(outcomes: Sequence[RestartOutcome], rungs: int) -> dict | None:
    """Stack the per-restart converged readout, padding the ladder with NaN.

    Restarts stop at different rungs, so ``ladder_weights`` is ragged; the
    padded columns say "this restart had already stopped", which is exactly
    what the stop-reason census needs.
    """
    if not outcomes or outcomes[0].stop_reason is None:
        return None
    ladder = np.full((len(outcomes), rungs), np.nan)
    for row, outcome in enumerate(outcomes):
        recorded = np.asarray(outcome.ladder_weights, dtype=float)
        ladder[row, : len(recorded)] = recorded
    return {
        "stop_reasons": np.array([o.stop_reason for o in outcomes], dtype="<U12"),
        "ladder_weights": ladder,
        "running_max": np.array([o.running_max for o in outcomes], dtype=float),
        "polish_gain": np.array([o.polish_gain for o in outcomes], dtype=float),
        "legacy_stop_step": np.array(
            [o.legacy_stop_step for o in outcomes], dtype=np.int64
        ),
        "legacy_weight": np.array([o.legacy_weight for o in outcomes], dtype=float),
    }


def run_instance(
    circuit_config: CircuitConfig,
    settings: RestartSettings,
    num_restarts: int,
    base_seed: int,
    instance_index: int,
    num_workers: int,
    restart_cache: str | None = None,
) -> InstanceData:
    """Fix one random circuit and collect ``num_restarts`` final solutions.

    With ``restart_cache`` set, each restart is persisted as it finishes and
    reloaded on a later invocation. A converged restart at n = 16 runs for
    hours, so without a per-restart checkpoint a single spot preemption
    destroys the whole instance.
    """
    circuit_rng = np.random.default_rng(
        np.random.SeedSequence(
            base_seed, spawn_key=(circuit_config.num_qubits, instance_index)
        )
    )
    haar_layers = sample_haar_random_layers(circuit_config, circuit_rng)
    baseline = random_section_peak_weight(circuit_config, haar_layers)
    logger.info(
        "n = %d, tau_r = %d, tau_p = %d, instance %d: %d restarts, "
        "init scale %g, baseline peak weight %.2e",
        circuit_config.num_qubits,
        circuit_config.num_random_layers,
        circuit_config.num_peaking_layers,
        instance_index,
        num_restarts,
        settings.init_scale,
        baseline,
    )

    cache_dir: str | None = None
    outcomes: list[RestartOutcome | None] = [None] * num_restarts
    if restart_cache is not None:
        fingerprint = settings_fingerprint(
            circuit_config, settings, base_seed, instance_index
        )
        directory = Path(restart_cache) / (
            f"n{circuit_config.num_qubits}_i{instance_index}_{fingerprint}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        cache_dir = str(directory)
        for index in range(num_restarts):
            outcomes[index] = load_restart(cache_dir, index)
        recovered = sum(outcome is not None for outcome in outcomes)
        if recovered:
            logger.info(
                "restart cache %s: %d/%d restarts recovered",
                cache_dir,
                recovered,
                num_restarts,
            )

    tasks = [
        RestartTask(
            circuit_config=circuit_config,
            haar_layers=haar_layers,
            settings=settings,
            base_seed=base_seed,
            instance_index=instance_index,
            restart_index=index,
            cache_dir=cache_dir,
        )
        for index in range(num_restarts)
        if outcomes[index] is None
    ]

    start = time.perf_counter()
    if num_workers == 1:
        for done, task in enumerate(tasks, start=1):
            index, outcome = run_restart_task(task)
            outcomes[index] = outcome
            _log_restart(done, len(tasks), index, outcome)
    elif tasks:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as pool:
            futures = [pool.submit(run_restart_task, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                index, outcome = future.result()
                outcomes[index] = outcome
                _log_restart(done, len(tasks), index, outcome)
    elapsed = time.perf_counter() - start

    complete = [outcome for outcome in outcomes if outcome is not None]
    assert len(complete) == num_restarts
    peak_weights = np.array([outcome.peak_weight for outcome in complete])

    state_fn = build_state_fn(circuit_config, haar_layers)
    states = np.stack(
        [np.asarray(state_fn(outcome.theta_final)) for outcome in complete]
    )
    zero_string_weights = np.abs(states[:, 0]) ** 2
    if not np.allclose(zero_string_weights, peak_weights, atol=1e-9):
        logger.warning(
            "state amplitudes disagree with recorded peak weights (max diff %.2e)",
            float(np.max(np.abs(zero_string_weights - peak_weights))),
        )
    probabilities = np.abs(states) ** 2
    argmax_indices = probabilities.argmax(axis=1)
    argmax_weights = probabilities[np.arange(len(complete)), argmax_indices]

    logger.info(
        "instance done in %.1f s: delta best %.4f, mean %.4f, min %.4f; "
        "%d/%d restarts peak on the 0^n string",
        elapsed,
        peak_weights.max(),
        peak_weights.mean(),
        peak_weights.min(),
        int(np.sum(argmax_indices == 0)),
        num_restarts,
    )
    return InstanceData(
        thetas_init=np.stack([outcome.theta_init for outcome in complete]),
        thetas_final=np.stack([outcome.theta_final for outcome in complete]),
        peak_weights=peak_weights,
        num_steps=np.array([outcome.num_steps for outcome in complete]),
        restart_seconds=np.array([outcome.seconds for outcome in complete]),
        argmax_indices=argmax_indices,
        argmax_weights=argmax_weights,
        overlaps=overlap_matrix(states),
        baseline_peak_weight=baseline,
        elapsed_seconds=elapsed,
        converged=_collect_converged(complete, len(settings.ladder)),
    )


def _log_restart(done: int, total: int, index: int, outcome: RestartOutcome) -> None:
    logger.info(
        "restart %3d finished (%3d/%d)  delta = %.4f  steps = %d  [%.1f s]",
        index,
        done,
        total,
        outcome.peak_weight,
        outcome.num_steps,
        outcome.seconds,
    )


def save_instance(
    path: Path,
    circuit_config: CircuitConfig,
    settings: RestartSettings,
    data: InstanceData,
    base_seed: int,
    instance_index: int,
) -> None:
    """Persist the ensemble and everything needed to rebuild it exactly.

    Converged-protocol keys are appended *after* the frozen ones and only when
    the run was instrumented, so an archive written under the frozen protocol
    keeps its exact key set and ordering.
    """
    extra: dict[str, object] = {}
    if not settings.is_frozen_protocol:
        extra.update(
            min_learning_rate=settings.min_learning_rate,
            ladder=np.array(settings.ladder, dtype=np.int64),
            rel_tol=settings.rel_tol,
            polish_steps=settings.polish_steps,
            init_mode=settings.init_mode,
            optimizer=settings.optimizer,
        )
    if data.converged is not None:
        extra.update(data.converged)
    np.savez_compressed(
        path,
        thetas_init=data.thetas_init,
        thetas_final=data.thetas_final,
        peak_weights=data.peak_weights,
        num_steps=data.num_steps,
        restart_seconds=data.restart_seconds,
        argmax_indices=data.argmax_indices,
        argmax_weights=data.argmax_weights,
        overlap_matrix=data.overlaps,
        baseline_peak_weight=data.baseline_peak_weight,
        elapsed_seconds=data.elapsed_seconds,
        num_qubits=circuit_config.num_qubits,
        num_random_layers=circuit_config.num_random_layers,
        num_peaking_layers=circuit_config.num_peaking_layers,
        base_seed=base_seed,
        instance_index=instance_index,
        max_steps=settings.max_steps,
        learning_rate=settings.learning_rate,
        decay_every=settings.decay_every,
        decay_factor=settings.decay_factor,
        convergence_tol=settings.convergence_tol,
        min_steps=settings.min_steps,
        init_scale=settings.init_scale,
        **extra,
    )
    logger.info("saved instance data to %s", path)


def run_pilot(base_seed: int, num_workers: int, num_restarts: int = 2) -> None:
    """Measure seconds/step at each grid size and extrapolate the budget.

    Runs serially for clean timing; parallel workers share memory bandwidth,
    so the extrapolation is a lower bound on the real wall time.
    """
    pilot_steps = 60
    production = RestartSettings()
    total_hours = 0.0
    for num_qubits in GRID_QUBIT_COUNTS:
        config = az_scaling_config(num_qubits)
        settings = RestartSettings(max_steps=pilot_steps, min_steps=pilot_steps)
        circuit_rng = np.random.default_rng(
            np.random.SeedSequence(base_seed, spawn_key=(num_qubits, 0))
        )
        haar_layers = sample_haar_random_layers(config, circuit_rng)
        peak_weight_fn = build_peak_weight_fn(config, haar_layers)
        seconds_per_step = []
        for index in range(num_restarts):
            task = RestartTask(config, haar_layers, settings, base_seed, 0, index)
            outcome = optimize_one_restart(
                peak_weight_fn, config.num_peaking_parameters, settings, restart_rng(task)
            )
            seconds_per_step.append(outcome.seconds / outcome.num_steps)
        step_time = float(np.mean(seconds_per_step))
        restart_minutes = step_time * production.max_steps / 60
        instance_hours = restart_minutes * 200 / num_workers / 60
        total_hours += 3 * instance_hours
        logger.info(
            "n = %2d: %.3f s/step -> ~%.1f min/restart (%d steps), "
            "~%.2f h/instance (200 restarts, %d workers)",
            num_qubits,
            step_time,
            restart_minutes,
            production.max_steps,
            instance_hours,
            num_workers,
        )
    logger.info(
        "full grid estimate (3 instances per n): ~%.1f h wall time, "
        "lower bound (serial timing, shared memory bandwidth)",
        total_hours,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="measure seconds/step at each grid size and estimate the budget",
    )
    parser.add_argument("--num-qubits", type=int, default=None, help="n (even)")
    parser.add_argument(
        "--instance", type=int, default=0, help="circuit instance index"
    )
    parser.add_argument(
        "--restarts", type=int, default=200, help="independent Adam runs"
    )
    parser.add_argument(
        "--max-steps", type=int, default=400, help="Adam steps per restart"
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        default=0.1,
        help="std-dev of the initialization (0.1 = protocol, larger = control)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="parallel worker processes"
    )
    parser.add_argument("--base-seed", type=int, default=42, help="root RNG seed")
    parser.add_argument(
        "--random-layers",
        type=int,
        default=None,
        help="override tau_r (default: n, the A&Z scaling regime)",
    )
    parser.add_argument(
        "--peaking-layers",
        type=int,
        default=None,
        help="override tau_p (default: n/2, the A&Z scaling regime)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.05,
        help="Adam/SGD stepsize (0.05 = protocol; robustness matrix)",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=("adam", "sgd", "lbfgs"),
        help="optimizer (adam = protocol; robustness and optimizer-class arms)",
    )
    parser.add_argument(
        "--output", type=str, default="results/pq", help="output directory"
    )
    converged = parser.add_argument_group(
        "converged protocol",
        "Off by default: with none of these given the run is the frozen "
        "protocol, bit for bit. --converged sets the whole block at once.",
    )
    converged.add_argument(
        "--converged", action="store_true",
        help="apply the converged protocol (ladder, lr floor, rel-tol, polish)",
    )
    converged.add_argument(
        "--min-learning-rate", type=float, default=None,
        help="floor under the decayed stepsize (0 = none, the frozen protocol)",
    )
    converged.add_argument(
        "--ladder", type=str, default=None,
        help="comma-separated steps at which the running max is recorded",
    )
    converged.add_argument(
        "--rel-tol", type=float, default=None,
        help="stop once a budget doubling gains less than this fraction",
    )
    converged.add_argument(
        "--polish-steps", type=int, default=None,
        help="anneal steps from the best point visited (0 = none)",
    )
    converged.add_argument(
        "--convergence-tol", type=float, default=None,
        help="frozen early-stop tolerance; 0 disables it (converged protocol)",
    )
    converged.add_argument(
        "--init-mode", type=str, default="normal", choices=("normal", "haar"),
        help="initialization law (normal = protocol; haar = optimizer-class arm)",
    )
    parser.add_argument(
        "--restart-cache", type=str, default=None,
        help="directory of per-restart checkpoints, for resumable long runs",
    )
    args = parser.parse_args(argv)
    if not args.pilot and args.num_qubits is None:
        parser.error("--num-qubits is required unless --pilot is given")
    if args.restarts < 2:
        parser.error("--restarts must be >= 2 (P(q) needs pairs)")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def settings_from_args(args: argparse.Namespace) -> RestartSettings:
    """Build the restart settings, defaulting to the frozen protocol.

    ``--converged`` supplies the whole block; individual flags then override
    it, so the pilot can sweep one knob without restating the rest.
    """
    chosen = dict(CONVERGED_PROTOCOL) if args.converged else {}
    if args.converged and args.max_steps == 400:  # the flag's own default
        chosen["max_steps"] = CONVERGED_PROTOCOL["max_steps"]
    else:
        chosen["max_steps"] = args.max_steps
    if args.ladder is not None:
        chosen["ladder"] = tuple(
            int(rung) for rung in args.ladder.split(",") if rung.strip()
        )
    for name in ("min_learning_rate", "rel_tol", "polish_steps", "convergence_tol"):
        value = getattr(args, name)
        if value is not None:
            chosen[name] = value
    return RestartSettings(
        init_scale=args.init_scale,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        init_mode=args.init_mode,
        **chosen,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    args = parse_args(argv)
    if args.pilot:
        run_pilot(args.base_seed, args.workers)
        return

    circuit_config = az_scaling_config(
        args.num_qubits, args.random_layers, args.peaking_layers
    )
    settings = settings_from_args(args)
    data = run_instance(
        circuit_config,
        settings,
        num_restarts=args.restarts,
        base_seed=args.base_seed,
        instance_index=args.instance,
        num_workers=args.workers,
        restart_cache=args.restart_cache,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"pq_n{circuit_config.num_qubits}_i{args.instance}"
        f"_sigma{args.init_scale:g}.npz"
    )
    save_instance(path, circuit_config, settings, data, args.base_seed, args.instance)


if __name__ == "__main__":
    main()
