"""The out-of-class optimizer arm: L-BFGS-B in place of the frozen schedule.

The paper's central verdict is quantified on the *stable-local* class, and that
class is defined by a cadence: the per-step displacement is fixed by the
learning-rate schedule. A quasi-Newton method has no schedule -- its step is
whatever the line search accepts -- so it is not in the class by construction,
and the only published competitor uses one. Aaronson & Zhang (arXiv:2404.14493)
announce Adam in their text while their code runs
``TNOptimizer(..., optimizer='L-BFGS-B')``. Their position with respect to the
class is therefore undetermined, and this module is what determines it: the
same instances, the same parameterization, the same evaluation budget, one
optimizer swapped.

The arm is deliberately given every advantage, so that a null result is worth
reporting. It optimizes a better-conditioned objective; it runs at tolerances
tight enough to be non-binding; it may start from Haar-random gates rather than
near the identity (``--init-mode haar``); and it is given the whole evaluation
budget of the converged Adam cell it is measured against, not the smaller
budget Adam typically stops at. If it still fails to beat Adam under all of
that, the conclusion is robust, and no reader can attribute the paper's reach
values to the choice of optimizer.

What a matched budget is, and which way the asymmetry runs. The common unit is
the value-and-gradient evaluation, because that is what costs: one evaluation
simulates the circuit and differentiates it once, whichever optimizer asked for
it, and the optimizer's own arithmetic is not measurable beside it. One Adam
step is exactly one evaluation. One L-BFGS-B *iteration* is several, because
the line search probes. Matching the two in evaluations therefore buys this arm
fewer iterations than Adam has steps -- the asymmetry runs against it, not for
it -- and that is the direction the comparison needs: it is the setting in
which "L-BFGS-B did not reach further" cannot be answered with "it was given
less work to do". Matching iterations instead would hand the arm a multiple of
Adam's circuit budget and measure the budget rather than the optimizer.

Why -ln(delta) rather than -delta. The scipy L-BFGS-B tolerances ``pgtol`` and
``ftol`` are *absolute*. With delta ~ 2^-n they would be absurdly loose at
n = 16 -- the optimizer would declare convergence on a landscape it had barely
entered -- and correspondingly tight at n = 8, so the arm would not even be
self-consistent across sizes. Taking the logarithm is a monotone
reparameterization: it moves no optimum, but it makes the gradient scale-free,
so one tolerance means the same thing at every size.

Contract notes for the archive format. ``legacy_stop_step`` and
``legacy_weight`` record where the frozen 400-step rule would have fired along
the trajectory; that rule reads a step schedule, and there is none here, so the
quantity is undefined for this arm. It is reported as ``-1`` and NaN
respectively, *not* as None, for two reasons that are not stylistic.
``pq_experiment._collect_converged`` stacks ``legacy_stop_step`` into an int64
array, which None cannot enter; and ``store_restart`` writes None as an object
array, which ``load_restart`` then refuses under ``allow_pickle=False``, so a
None here would silently disable the per-restart cache and cost the whole
instance on the first spot preemption. ``-1`` and NaN are the sentinels, and
downstream analysis must exclude this arm from any truncation-deficit statistic
rather than read them as a step count and a weight.

The budget is not a default. Left alone, ``--optimizer lbfgs`` inherits the
frozen 400-step cap, three percent of the converged cell's, which is not the
comparison the optclass block registers; ``cloud/runner.py`` passes the matched
budget and the shared ladder on every arm of that block.

Usage:
    python lbfgs_restart.py                                # self-tests, ~11 s
    python pq_experiment.py --num-qubits 12 --instance 0 --optimizer lbfgs \\
        --max-steps 13200 --ladder 400,800,1600,3200,6400,12800
    python pq_experiment.py --num-qubits 12 --instance 0 --optimizer lbfgs \\
        --max-steps 13200 --ladder 400,800,1600,3200,6400,12800 \\
        --init-mode haar                                   # the A&Z arm

Cost: one evaluation costs about one Adam step, and a matched restart may spend
13200 of them, so an arm of the optclass block is provisioned at what the
converged Adam cell beside it is provisioned at. Both usually stop well short
-- Adam on its relative tolerance, this arm on the scipy convergence test -- but
the ceiling is what a spot campaign has to budget for. The line search's extra
probes are evaluations like any other and come out of that budget, not on top
of it.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

#: The evaluation budget of the converged Adam cell this arm is compared
#: against: ``CONVERGED_PROTOCOL``'s main loop plus its polish, which is the
#: most a converged restart can spend. ``settings.max_steps`` is read as an
#: evaluation count -- one Adam step is one value-and-gradient pass -- and this
#: is the ceiling on it.
#:
#: Parity is with that worst case and not with the rung where Adam's relative
#: tolerance typically fires, because Adam is free to stop early and keep what
#: it found, so a ceiling set at its typical stopping point would be a ceiling
#: on this arm alone. The earlier value here was 3200, and under it the
#: registered statistic ``rho_arm(n) = R_arm(n) / R_conv(n)`` compared a
#: truncated arm against an untruncated cell: rho would have measured the
#: ceiling rather than the optimizer, which is the exact bias the converged
#: campaign exists to remove. ``_self_tests`` pins this number to the protocol
#: it mirrors, so the two cannot drift apart unnoticed.
MATCHED_EVALUATION_BUDGET = 13200

#: Dimensionless once the objective is -ln(delta), so they carry the same
#: meaning at every n. Both are far below the resolution of the landscape.
PROJECTED_GRADIENT_TOL = 1e-6
FUNCTION_TOL = 1e-12

#: Guard for the logarithm. delta is a projector expectation and cannot be
#: negative in exact arithmetic; a maximizer never drives it toward zero. The
#: floor exists so that a roundoff-sized value cannot produce a NaN objective
#: and abort the restart.
MIN_WEIGHT = 1e-12


class _BudgetExhausted(Exception):
    """Raised from the objective once the matched evaluation budget is spent.

    scipy honours ``maxfun`` only between line-search probes and overshoots it
    by an evaluation or two. That is harmless in general and not harmless here:
    budget parity with the Adam arm is the whole claim of the comparison, so
    the cap is enforced on this side of the call instead.
    """


def optimize_one_restart_lbfgs(
    peak_weight_fn,
    num_parameters: int,
    settings,
    rng: np.random.Generator,
):
    """One L-BFGS-B run from a fresh initialization; same contract as Adam.

    ``num_steps`` is the number of value-and-gradient evaluations actually
    consumed, which is the budget unit shared with the Adam arm. ``peak_weight``
    and ``running_max`` are both the best delta seen *along* the run, including
    line-search trial points, which is the converged protocol's semantics: the
    line search regularly probes past the accepted iterate, and the final point
    is not always the best one visited.

    ``stop_reason`` is one of "converged", "maxfun", "maxiter", "linesearch"
    (an L-BFGS-B line-search abort, which the frozen protocol has no analogue
    for) or "abnormal". ``ladder_weights`` records the running maximum at the
    rungs of ``settings.ladder``, read as evaluation counts so that the ladder
    is directly comparable with Adam's. ``polish_gain`` is zero: the line
    search already plays that role, so no separate anneal is run and the
    anneal's share of the matched budget is spent inside the main run instead.
    That is why this arm's ``max_steps`` is Adam's main loop and polish added
    together while its archived ``polish_steps`` is 0 -- the two sides carry
    the same total and an auditor can add them up.
    ``legacy_stop_step`` and ``legacy_weight`` are meaningless without a step
    schedule and carry the sentinels ``-1`` and NaN; see the module header for
    why they are not None.
    """
    # Imported here rather than at module scope: pq_experiment imports this
    # module lazily, so a top-level import would close the cycle.
    from pq_experiment import RestartOutcome, draw_initial_parameters

    theta_init = draw_initial_parameters(num_parameters, settings, rng)
    # The ceiling can only ever clip *above* parity: a caller asking for more
    # than the Adam cell can spend would be handing this arm an advantage that
    # the comparison could not be corrected for afterwards. It never clips
    # below, so it can never be the reason the arm falls short.
    budget = min(int(settings.max_steps), MATCHED_EVALUATION_BUDGET)
    if settings.max_steps > MATCHED_EVALUATION_BUDGET:
        logger.warning(
            "L-BFGS-B budget clipped to the matched ceiling of %d evaluations "
            "(max_steps = %d would outspend the converged Adam cell)",
            budget,
            settings.max_steps,
        )
    rungs = [rung for rung in settings.ladder if rung <= budget]

    gradient_fn = qml.grad(peak_weight_fn)
    best_weight = -np.inf
    best_theta = np.asarray(theta_init, dtype=float)
    evaluations = 0
    ladder_weights: list[float] = []

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal best_weight, best_theta, evaluations
        if evaluations >= budget:
            raise _BudgetExhausted
        theta = pnp.array(flat, requires_grad=True)
        gradient = np.asarray(gradient_fn(theta), dtype=float)
        weight = float(gradient_fn.forward)
        evaluations += 1
        if weight > best_weight:
            best_weight, best_theta = weight, np.array(flat, dtype=float, copy=True)
        recorded = len(ladder_weights)
        while recorded < len(rungs) and evaluations >= rungs[recorded]:
            ladder_weights.append(best_weight)
            recorded += 1
        floored = max(weight, MIN_WEIGHT)
        # d(-ln delta)/dtheta = -(d delta/dtheta) / delta.
        return -float(np.log(floored)), -gradient / floored

    start = time.perf_counter()
    try:
        stop_reason = _stop_reason(
            minimize(
                objective,
                np.asarray(theta_init, dtype=float),
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxfun": budget,
                    "maxiter": budget,
                    "gtol": PROJECTED_GRADIENT_TOL,
                    "ftol": FUNCTION_TOL,
                },
            )
        )
    except _BudgetExhausted:
        stop_reason = "maxfun"
    seconds = time.perf_counter() - start

    # Rungs above the evaluations actually spent are left short, not padded
    # with the final weight: _collect_converged fills them with NaN, and NaN
    # is what the stop-reason census reads as "this restart had already
    # stopped". Padding would make an early convergence look like a run that
    # sat at the same weight for thousands of evaluations.
    return RestartOutcome(
        theta_init=theta_init,
        theta_final=best_theta,
        peak_weight=best_weight,
        num_steps=evaluations,
        seconds=seconds,
        stop_reason=stop_reason,
        ladder_weights=np.array(ladder_weights, dtype=float),
        running_max=best_weight,
        polish_gain=0.0,
        legacy_stop_step=-1,  # undefined without a step schedule; see the header
        legacy_weight=float("nan"),
    )


def _stop_reason(result) -> str:
    """Map the L-BFGS-B exit status onto the archive's stop-reason vocabulary."""
    if result.status == 0:
        return "converged"
    message = str(result.message).upper()
    if result.status == 1:
        return "maxfun" if "EVALUATIONS" in message else "maxiter"
    return "linesearch" if "LNSRCH" in message or "ABNORMAL" in message else "abnormal"


def _self_tests() -> None:
    """The matched-budget constant, a comparison against Adam at one, and the
    outcome's invariants."""
    # pq_experiment first: importing it is what puts the reproduction repo,
    # and therefore peaked_circuits, on the path.
    from pq_experiment import (
        CONVERGED_PROTOCOL,
        RestartSettings,
        az_scaling_config,
        optimize_one_restart,
    )
    from peaked_circuits import build_peak_weight_fn, sample_haar_random_layers

    # Parity is this arm's entire argument, so the number is checked rather
    # than trusted. A converged Adam restart spends one evaluation per
    # main-loop step and one per polish step, and no more.
    matched = int(CONVERGED_PROTOCOL["max_steps"]) + int(
        CONVERGED_PROTOCOL["polish_steps"]
    )
    assert MATCHED_EVALUATION_BUDGET == matched, (
        f"matched budget {MATCHED_EVALUATION_BUDGET} against the converged "
        f"protocol's {matched} evaluations (max_steps + polish_steps): the "
        "optclass arms would not be budget-matched, and rho_arm would measure "
        "the difference"
    )
    # And no rung of the shared ladder falls outside it, so the three arms
    # report their running maximum at the same points of the same axis. The
    # settings cloud/runner.py builds for this arm are constructed here too:
    # a combination RestartSettings rejects would fail on the box, not now.
    protocol_ladder = tuple(CONVERGED_PROTOCOL["ladder"])
    assert max(protocol_ladder) <= MATCHED_EVALUATION_BUDGET, protocol_ladder
    matched_settings = RestartSettings(
        max_steps=MATCHED_EVALUATION_BUDGET,
        optimizer="lbfgs",
        ladder=protocol_ladder,
    )
    assert matched_settings.polish_steps == 0, matched_settings.polish_steps

    config = az_scaling_config(4)
    layers = sample_haar_random_layers(
        config, np.random.default_rng(np.random.SeedSequence(42, spawn_key=(4, 0)))
    )
    peak_weight_fn = build_peak_weight_fn(config, layers)
    # Small enough that neither optimizer has saturated this tiny instance:
    # past ~100 evaluations both sit on delta = 0.7549 and the comparison
    # stops being able to fail.
    budget = 40

    adam = RestartSettings(max_steps=budget, min_steps=budget)
    quasi_newton = RestartSettings(max_steps=budget, optimizer="lbfgs")
    adam_weights, lbfgs_weights = [], []
    for seed in range(5):
        outcome = optimize_one_restart(
            peak_weight_fn,
            config.num_peaking_parameters,
            adam,
            np.random.default_rng(seed),
        )
        adam_weights.append(outcome.peak_weight)
        outcome = optimize_one_restart_lbfgs(
            peak_weight_fn,
            config.num_peaking_parameters,
            quasi_newton,
            np.random.default_rng(seed),
        )
        lbfgs_weights.append(outcome.peak_weight)
        shape = (config.num_peaking_parameters,)
        assert np.shape(outcome.theta_init) == shape, np.shape(outcome.theta_init)
        assert np.shape(outcome.theta_final) == shape, np.shape(outcome.theta_final)
        assert outcome.running_max <= outcome.peak_weight + 1e-15, (
            f"running_max {outcome.running_max} exceeds peak_weight "
            f"{outcome.peak_weight}"
        )
        assert 0 < outcome.num_steps <= budget, outcome.num_steps
        assert outcome.stop_reason in (
            "converged", "maxfun", "maxiter", "linesearch", "abnormal"
        ), outcome.stop_reason
        assert outcome.legacy_stop_step == -1 and np.isnan(outcome.legacy_weight)
        rebuilt = float(peak_weight_fn(pnp.array(outcome.theta_final)))
        assert abs(rebuilt - outcome.peak_weight) < 1e-12, rebuilt
        assert lbfgs_weights[-1] >= adam_weights[-1] - 1e-9, (
            f"seed {seed}: L-BFGS-B reaches {lbfgs_weights[-1]:.4f} against "
            f"Adam's {adam_weights[-1]:.4f} at a matched budget of {budget} "
            "evaluations; the arm is not receiving the advantages the "
            "comparison assumes."
        )

    adam_median = float(np.median(adam_weights))
    lbfgs_median = float(np.median(lbfgs_weights))

    ladder = RestartSettings(
        max_steps=budget, optimizer="lbfgs", ladder=(10, 20, budget)
    )
    outcome = optimize_one_restart_lbfgs(
        peak_weight_fn, config.num_peaking_parameters, ladder, np.random.default_rng(0)
    )
    reached = sum(rung <= outcome.num_steps for rung in (10, 20, budget))
    assert len(outcome.ladder_weights) == reached, outcome.ladder_weights
    assert np.all(np.diff(outcome.ladder_weights) >= -1e-15), outcome.ladder_weights

    print(
        f"self-tests passed (matched budget {budget} evaluations over 5 seeds: "
        f"L-BFGS-B median delta {lbfgs_median:.4f} vs Adam {adam_median:.4f}; "
        f"per seed L-BFGS-B "
        f"{' '.join(f'{value:.4f}' for value in lbfgs_weights)} vs Adam "
        f"{' '.join(f'{value:.4f}' for value in adam_weights)})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    _self_tests()
