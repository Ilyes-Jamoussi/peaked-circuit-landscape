"""Consolidation-campaign runner: the registered consolidation items, the
correction round, and the converged-protocol campaign.

Declarative job list -> pq_experiment invocations, resumable (skips any job
whose output npz already exists), one log per job, integrity gate
(pq_validate --allow-missing) once the whole campaign has finished.

Jobs run in a bounded pool of ``--jobs`` concurrent subprocesses, each given
``--workers / --jobs`` worker processes. A converged run at B = 32 restarts
occupies 32 of a 96-vCPU box on its own, so a strictly sequential campaign
wastes two thirds of the machine; ``--jobs 1`` is the old sequential
behaviour, unchanged.

``--list`` prints each job's evaluation budget beside it, so that the budget
parity the optclass block rests on is a property of the table a reader can
check rather than of the flags a reader has to reconstruct.

Usage (from the repo root):
    python cloud/runner.py --mini                    # miniature end-to-end run
    python cloud/runner.py --list                    # job table, with budgets
    python cloud/runner.py --only optclass --list    # the matched-budget block
    python cloud/runner.py --workers 96 --jobs 3     # full campaign
    python cloud/runner.py --only converged --workers 96 --jobs 3
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

#: How often the pool reaps finished subprocesses. Jobs run for minutes to
#: hours, so this only bounds the idle gap before a freed slot is refilled.
POLL_SECONDS = 0.5


_MISSING = object()


def _literal(node):
    """The node's value if it is a literal (or a dict of them), else _MISSING.

    A dict is taken entry by entry so that one non-literal value does not
    discard the whole constant: ``CONVERGED_PROTOCOL["min_learning_rate"]`` is
    written as an expression, and no caller here reads it.
    """
    if isinstance(node, ast.Dict):
        pairs = {}
        for key, value in zip(node.keys, node.values):
            try:
                pairs[ast.literal_eval(key)] = ast.literal_eval(value)
            except (TypeError, ValueError):
                continue
        return pairs
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        return _MISSING


def pq_experiment_constants() -> dict:
    """pq_experiment.py's module-level literal constants, without importing it.

    The runner is a launcher: importing pq_experiment would pull PennyLane in
    just to print a job table. Restating its constants here instead would let
    the protocol and the budget the optclass arms are matched to drift apart
    silently, which is the one thing that block cannot survive -- a stale
    number here comes back as a reach ratio confounded by truncation. So they
    are parsed out of the module that defines them.
    """
    tree = ast.parse((ROOT / "pq_experiment.py").read_text())
    constants: dict[str, object] = {}
    for statement in tree.body:
        if isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        elif isinstance(statement, ast.Assign):
            targets = statement.targets
        else:
            continue
        value = _literal(statement.value)
        if value is _MISSING:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
    return constants


_PQ = pq_experiment_constants()
_PROTOCOL = _PQ["CONVERGED_PROTOCOL"]

#: The frozen protocol's step cap, which is also pq_experiment's --max-steps
#: default: what a job that sets neither --max-steps nor --converged gets.
FROZEN_MAX_STEPS = int(_PQ["FROZEN_MAX_STEPS"])

#: The rungs at which every optclass arm reports its running maximum. Shared
#: so that the three arms are read at the same points of the same axis.
CONVERGED_LADDER = tuple(_PROTOCOL["ladder"])

#: Budget parity between a step method and a line-search method, in the only
#: unit they share: the value-and-gradient evaluation. One Adam step is one
#: such evaluation; one L-BFGS-B objective call is one, and its line search
#: spends several per iteration. A converged Adam restart spends at most
#: ``max_steps`` of them in the main loop plus ``polish_steps`` in the anneal,
#: so that sum -- not the ladder rung where its relative tolerance happens to
#: fire -- is the ceiling the L-BFGS arms must be given. The arms run no
#: anneal, so they take the whole sum as ``--max-steps``, and every optclass
#: archive then satisfies ``max_steps + polish_steps == MATCHED_EVALUATIONS``.
MATCHED_EVALUATIONS = int(_PROTOCOL["max_steps"]) + int(
    _PROTOCOL["polish_steps"]
)


def campaign(mini: bool = False) -> list[dict]:
    """The frozen measurement list. Each job maps to one pq_experiment run."""
    jobs = []

    def add(block, n, instance, restarts, output, **extra):
        jobs.append(dict(block=block, n=n, instance=instance,
                         restarts=restarts, output=output, **extra))

    if mini:
        for i in (0, 1):
            add("mini", 6, i, 10, "results/mini")
        add("mini_deep", 6, 0, 10, "results/mini_deep", random_layers=24)
        add("mini_shallow", 6, 0, 10, "results/mini_shallow", peaking_layers=1)
        # The bootstrap gate has to cover the converged protocol too, or a
        # fresh box is only proven able to run the frozen one. Toy budget,
        # but every new branch is live: ladder, floor, rel-tol, polish and
        # the restart cache.
        add("mini_converged", 6, 0, 10, "results/mini_converged",
            converged=True, max_steps=200, ladder=(50, 100, 200),
            min_learning_rate=0.0125, rel_tol=0.05, polish_steps=25,
            convergence_tol=0.0, restart_cache="results/mini_cache")
        return jobs

    # 1. P(q) consolidation: 15 new instances per size + n = 16.
    for n in (8, 10, 12, 14):
        for i in range(3, 18):
            add("pq_consolidation", n, i, 200, "results/pq")
    for i in range(4):
        add("pq_n16", 16, i, 200, "results/pq")

    # 2. Budget-800 ensembles (fresh 800-restart batches, same instances).
    for n in (10, 12):
        for i in (0, 1, 2):
            add("budget800", n, i, 800, "results/budget800")

    # 4. Deep-limit anchors (skipped automatically if B2 ran them locally).
    for n in (10, 12):
        add("deep_anchor", n, 0, 60, f"results/depth_ceiling/tau{4 * n}",
            random_layers=4 * n)

    # 5. Shallow peaking at n = 12 (corrected bound regime, 09 section 3).
    for i in (0, 1, 2):
        add("shallow_n12", 12, i, 60, "results/shallow_peaking",
            peaking_layers=3)

    # Robustness matrix on (n = 10, instance 0): init scale, learning
    # rate, optimizer. Variants live in subdirectories because the npz
    # filename only encodes (n, instance, sigma).
    for scale in (0.5, 1.0):
        add("robustness", 10, 0, 200, "results/robustness/init",
            init_scale=scale)
    for rate, tag in ((0.025, "lr_half"), (0.1, "lr_double")):
        add("robustness", 10, 0, 200, f"results/robustness/{tag}",
            learning_rate=rate)
    add("robustness", 10, 0, 200, "results/robustness/sgd", optimizer="sgd")

    jobs.extend(correction_campaign())
    jobs.extend(converged_campaign())

    # The k = 4 moment transfer is memory-bound, not core-bound: each worker
    # holds a chunked configuration sum, and run beside the grid it saturates
    # the memory bandwidth every other job depends on. It goes last (stable
    # sort, so nothing else moves) and carries `solo`, which empties the pool
    # before it starts.
    jobs.sort(key=lambda job: job["block"] == "moment_probe")
    return jobs


def correction_campaign() -> list[dict]:
    """Blocks added by the correction round; see REGISTRATION.md.

    These drive analysis scripts rather than pq_experiment, so they carry
    their command explicitly. Resumability is unchanged: a job whose output
    already exists is skipped.
    """
    jobs: list[dict] = []

    # 1. Corrugation at ONE matched resolution, three instances per size.
    #    n = 14 was never measured; n = 16 instance 0 already exists at this
    #    resolution and is skipped automatically.
    for n in (8, 10, 12, 14, 16):
        for instance in (0, 1, 2):
            tag = f"_i{instance}" if instance else ""
            jobs.append(dict(
                block="connectivity_m64", kind="cmd", n=n, instance=instance,
                output=f"results/connectivity/conn_n{n}{tag}_o150s1_m64.npz",
                argv=["analysis/connectivity.py", "--num-qubits", str(n),
                      "--instance", str(instance), "--outer", "150",
                      "--steps", "1", "--segments", "64"],
            ))

    # 2. Step-budget control: the frozen protocol at 4x the step cap, on the
    #    protocol's own restart seeds, to size what truncation costs.
    for n in (8, 10, 12, 14):
        for instance in (0, 1, 2):
            jobs.append(dict(
                block="step_budget", kind="pq", n=n, instance=instance,
                restarts=16, output="results/step_budget", max_steps=1600,
            ))
    jobs.append(dict(block="step_budget", kind="pq", n=16, instance=0,
                     restarts=16, output="results/step_budget", max_steps=1600))

    # 3. Probe dependence of the moment hierarchy, up to k = 4. Sorted to the
    #    end of the campaign and run alone; see the note in campaign().
    for n in (6, 8):
        jobs.append(dict(
            block="moment_probe", kind="cmd", n=n, instance=0,
            output=f"analysis/moment_probe_n{n}.log",
            argv=["analysis/moment_probe_scan.py", "--sizes", str(n),
                  "--max-k", "4", "--solutions", "2"],
            max_workers=24, solo=True,
        ))

    # 4. The 240-point kernel residuals against Eq. (5), with bootstrap
    #    errors: the number the paper quotes, with a committed log at last.
    #    The residual scan is serial and its parser has no --workers, so the
    #    flag every other cmd job takes has to be withheld here.
    for n in (8, 10):
        jobs.append(dict(
            block="kernel_residuals", kind="cmd", n=n, instance=0,
            output=f"analysis/kernel_exact_residuals_n{n}.log",
            argv=["analysis/check_kernel_exact_residuals.py", "--sizes", str(n)],
            takes_workers=False,
        ))

    return jobs


def converged_campaign() -> list[dict]:
    """Blocks that measure the reach at convergence, not at the 400-step cap.

    The published reach values are the best peak weight found within 400 Adam
    steps. At n >= 10 that cap binds nearly every restart and every
    instance-best, so those values are lower bounds whose shortfall grows with
    n -- the same signature as an accelerating decay. These blocks re-measure
    the same quantity with the cap removed.

    Ordered by cost, cheapest first: the n = 16 catch-up and the five-point
    pilot exercise the whole pipeline (and fix the provisional constants in
    CONVERGED_PROTOCOL) before the full grid is committed to.
    """
    jobs: list[dict] = []
    cache = "results/restart_cache"

    # 1. The frozen protocol at n = 16, instances 4..17. The paper's most
    #    contested point rests on four instances; this takes it to eighteen.
    #    Deliberately no converged flag: this block has to stay comparable
    #    with the published grid, so it must be the same protocol, bit for
    #    bit, and it lands in the same directory.
    for instance in range(4, 18):
        jobs.append(dict(block="frozen_n16_catchup", kind="pq", n=16,
                         instance=instance, restarts=200,
                         output="results/pq"))

    # 2. Scale pilot: one instance per size, 16 restarts, to measure the gain
    #    per budget doubling and settle the ladder, the floor and rel_tol
    #    before the grid inherits them.
    for n in (8, 10, 12, 14, 16):
        jobs.append(dict(block="converged_pilot", kind="pq", n=n, instance=0,
                         restarts=16, output="results/converged_pilot",
                         converged=True, restart_cache=cache))

    # 3. What the learning-rate floor costs. Too high and the run never
    #    settles, too low and the optimizer freezes short of the optimum,
    #    which is what stalls the existing 1600-step control. The middle arm
    #    is CONVERGED_PROTOCOL's own value, so it has the pilot's fingerprint
    #    and reuses its restarts straight out of the cache.
    for floor in (0.0125, 0.00625, 0.00078125):
        jobs.append(dict(block="lr_floor_sweep", kind="pq", n=12, instance=0,
                         restarts=16,
                         output=f"results/lr_floor_sweep/floor{floor:g}",
                         converged=True, min_learning_rate=floor,
                         restart_cache=cache))

    # 4. The campaign itself: the reach at convergence, 18 instances per size
    #    including n = 16, B = 32 restarts. This is the grid the reach law is
    #    refitted on.
    for n in (8, 10, 12, 14, 16):
        for instance in range(18):
            jobs.append(dict(block="converged", kind="pq", n=n,
                             instance=instance, restarts=32,
                             output="results/converged", converged=True,
                             restart_cache=cache))

    # 5. Is the ceiling Adam's, or the normal initialization's? A quasi-Newton
    #    method and a Haar-random start, crossed. Arms live in subdirectories
    #    because the npz filename only encodes (n, instance, sigma).
    #
    #    The registered statistic is rho_arm(n) = R_arm(n) / R_conv(n), whose
    #    fourth cell -- (Adam, normal) -- is the converged grid read at
    #    B0 = 16. Every arm here is therefore compared against a run that had
    #    MATCHED_EVALUATIONS value-and-gradient evaluations to spend, and an
    #    arm given fewer would lose to truncation rather than to quality: rho
    #    would measure the cap, which is the exact bias this campaign exists
    #    to remove. Budget is set explicitly on all three, never defaulted.
    #
    #    Adam reaches the ceiling as --converged (12800 main + 400 polish).
    #    L-BFGS-B has no anneal, so it takes the sum as its evaluation cap; it
    #    also has no step schedule, hence no learning-rate floor and no
    #    relative-tolerance stop to set, and --ladder alone puts its running
    #    maximum on the same rungs as Adam's. Its archive then reads
    #    polish_steps = 0, max_steps = MATCHED_EVALUATIONS, which is the parity
    #    an auditor can check without rerunning anything.
    lbfgs = dict(optimizer="lbfgs", max_steps=MATCHED_EVALUATIONS,
                 ladder=CONVERGED_LADDER)
    arms = (
        ("lbfgs_sigma", lbfgs),
        ("lbfgs_haar", dict(lbfgs, init_mode="haar")),
        ("adam_haar", dict(converged=True, init_mode="haar")),
    )
    #    At parity one n = 16 restart runs for hours, the regime that gave the
    #    converged block its restart cache: without one, a spot preemption
    #    costs the whole instance. The cache directory is keyed on the full
    #    settings fingerprint, so no arm can read another's entries, nor the
    #    converged grid's.
    for tag, flags in arms:
        for n in (8, 10, 12, 14, 16):
            for instance in (0, 1, 2):
                jobs.append(dict(block="optclass", kind="pq", n=n,
                                 instance=instance, restarts=16,
                                 output=f"results/optclass/{tag}",
                                 restart_cache=cache, **flags))

    return jobs


def output_path(job: dict) -> Path:
    if job.get("kind") == "cmd":
        return ROOT / job["output"]
    scale = job.get("init_scale", 0.1)
    return ROOT / job["output"] / (
        f"pq_n{job['n']}_i{job['instance']}_sigma{scale:g}.npz"
    )


def evaluation_budget(job: dict) -> int | None:
    """Worst-case value-and-gradient evaluations one restart of this job costs.

    None for a job that does not run pq_experiment. This is the unit the
    optclass arms are matched in -- one Adam step, one polish step and one
    L-BFGS-B objective call are each one evaluation -- so ``--list`` prints it
    and the parity of a block can be read off the job table instead of being
    taken on trust. It mirrors ``settings_from_args``: ``--converged`` supplies
    the block, an explicit key overrides it.
    """
    if job.get("kind") == "cmd":
        return None
    converged = "converged" in job
    protocol_steps = _PROTOCOL["max_steps"] if converged else FROZEN_MAX_STEPS
    protocol_polish = _PROTOCOL["polish_steps"] if converged else 0
    return int(job.get("max_steps", protocol_steps)) + int(
        job.get("polish_steps", protocol_polish)
    )


def log_path(job: dict) -> Path:
    """Where one job's console output goes -- one file per job.

    A shared per-directory log interleaves unusably once jobs run
    concurrently, and it also hides which shard produced a traceback. For a
    job whose artifact IS a log, this is the staging path: it is moved into
    place only on success, so a failed run leaves no skip marker behind.
    """
    target = output_path(job)
    if target.suffix == ".log":
        return target.with_suffix(".log.partial")
    return target.with_suffix(".log")


def cache_key(job: dict) -> tuple | None:
    """The restart-cache directory a job will write into, or None.

    Two runs with identical settings share one cache directory, and every
    entry is staged there under a fixed .partial name, so two concurrent
    writers can interleave into a single corrupt file. The pool refuses to
    run two such jobs at once. Keying on (root, n, instance) is coarser than
    the settings fingerprint the cache itself uses, which is the safe
    direction: it can only serialize more than strictly necessary.
    """
    if "restart_cache" not in job:
        return None
    return (job["restart_cache"], job["n"], job["instance"])


def command(job: dict, workers: int) -> list[str]:
    # A job may cap its own parallelism: the k = 4 moment transfer holds a
    # chunked configuration sum per worker and is memory-bound, not
    # core-bound, so it must not inherit a 96-way pool.
    workers = min(workers, job.get("max_workers", workers))
    if job.get("kind") == "cmd":
        argv = [PYTHON, *job["argv"]]
        if job.get("takes_workers", True):
            argv += ["--workers", str(workers)]
        return argv
    cmd = [PYTHON, "pq_experiment.py",
           "--num-qubits", str(job["n"]),
           "--instance", str(job["instance"]),
           "--restarts", str(job["restarts"]),
           "--workers", str(workers),
           "--output", job["output"]]
    if "max_steps" in job:
        cmd += ["--max-steps", str(job["max_steps"])]
    if "random_layers" in job:
        cmd += ["--random-layers", str(job["random_layers"])]
    if "peaking_layers" in job:
        cmd += ["--peaking-layers", str(job["peaking_layers"])]
    if "init_scale" in job:
        cmd += ["--init-scale", str(job["init_scale"])]
    if "learning_rate" in job:
        cmd += ["--learning-rate", str(job["learning_rate"])]
    if "optimizer" in job:
        cmd += ["--optimizer", str(job["optimizer"])]
    # The converged protocol. --converged sets the whole block; the
    # individual flags then override it, so a sweep restates one knob only.
    if "converged" in job:
        cmd += ["--converged"]
    if "min_learning_rate" in job:
        cmd += ["--min-learning-rate", str(job["min_learning_rate"])]
    if "ladder" in job:
        cmd += ["--ladder", ",".join(str(rung) for rung in job["ladder"])]
    if "rel_tol" in job:
        cmd += ["--rel-tol", str(job["rel_tol"])]
    if "polish_steps" in job:
        cmd += ["--polish-steps", str(job["polish_steps"])]
    if "convergence_tol" in job:
        cmd += ["--convergence-tol", str(job["convergence_tol"])]
    if "init_mode" in job:
        cmd += ["--init-mode", str(job["init_mode"])]
    if "restart_cache" in job:
        cmd += ["--restart-cache", job["restart_cache"]]
    return cmd


@dataclass
class Slot:
    """One subprocess the pool is currently holding open."""

    job: dict
    process: subprocess.Popen
    handle: TextIO
    log: Path
    target: Path
    start: float


def launch(job: dict, workers: int) -> Slot:
    target = output_path(job)
    target.parent.mkdir(parents=True, exist_ok=True)
    log = log_path(job)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(job, workers)
    print(f"run: {' '.join(cmd[1:])}", flush=True)
    # Append for an npz job so a resumed run keeps the record of what the
    # preempted attempt had already finished; truncate for a staged log,
    # which is the artifact itself and must contain one run only.
    handle = open(log, "w" if target.suffix == ".log" else "a")
    process = subprocess.Popen(cmd, cwd=ROOT, stdout=handle,
                               stderr=subprocess.STDOUT)
    return Slot(job=job, process=process, handle=handle, log=log,
                target=target, start=time.perf_counter())


def reap(slot: Slot, tagged: bool) -> str | None:
    """Close out a finished job; return a failure message, or None if it ran."""
    slot.handle.close()
    code = slot.process.returncode
    if code == 0 and slot.target.suffix == ".log":
        slot.log.replace(slot.target)
    status = "ok" if code == 0 else f"FAILED ({code})"
    tag = f"{slot.target.name} " if tagged else ""
    elapsed = time.perf_counter() - slot.start
    print(f"  -> {tag}{status} [{elapsed:.0f} s]", flush=True)
    if code != 0:
        return f"job failed, see {slot.log}"
    return None


def run_jobs(jobs: list[dict], workers: int, max_jobs: int) -> str | None:
    """Run the job list through a pool of at most ``max_jobs`` subprocesses.

    Returns None if everything succeeded, otherwise the first failure
    message. On failure nothing further is launched, but the jobs already in
    flight are waited out: killing them would leave half-written archives and
    orphaned worker pools on a machine that is about to be torn down.
    """
    per_job = max(1, workers // max_jobs)
    pending = list(jobs)
    running: list[Slot] = []
    held: set[tuple] = set()
    failure: str | None = None

    while pending or running:
        while pending and failure is None and len(running) < max_jobs:
            job = pending[0]
            # A solo job runs with the pool to itself, in both directions.
            if running and (job.get("solo")
                            or any(slot.job.get("solo") for slot in running)):
                break
            key = cache_key(job)
            if key is not None and key in held:
                break
            pending.pop(0)
            target = output_path(job)
            if target.exists():
                print(f"skip (exists): {target.name}")
                continue
            running.append(launch(job, workers if job.get("solo") else per_job))
            if key is not None:
                held.add(key)

        if not running:
            break
        time.sleep(POLL_SECONDS)
        for slot in list(running):
            if slot.process.poll() is None:
                continue
            running.remove(slot)
            held.discard(cache_key(slot.job))
            message = reap(slot, tagged=max_jobs > 1)
            if message and failure is None:
                failure = message

    return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--sizes", type=str, default=None,
                        help="comma-separated n filter, e.g. 10 or 8,10 "
                             "(shards one block across machines)")
    parser.add_argument("--instances", type=str, default=None,
                        help="comma-separated instance filter, e.g. 3,4,5 "
                             "(finer sharding within one size)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=1,
                        help="concurrent pq_experiment processes; each gets "
                             "workers/jobs workers (1 = strictly sequential)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    jobs = campaign(mini=args.mini)
    if args.only:
        jobs = [j for j in jobs if j["block"] == args.only]
    if args.sizes:
        wanted = {int(size) for size in args.sizes.split(",")}
        jobs = [j for j in jobs if j["n"] in wanted]
    if args.instances:
        chosen = {int(i) for i in args.instances.split(",")}
        jobs = [j for j in jobs if j["instance"] in chosen]

    if args.list:
        for job in jobs:
            target = output_path(job)
            done = "done" if target.exists() else "    "
            detail = (f"restarts={job['restarts']}" if "restarts" in job
                      else Path(job["argv"][0]).name)
            budget = evaluation_budget(job)
            evals = f"evals={budget}" if budget is not None else ""
            print(f"[{done}] {job['block']:>18} n={job['n']:>2} "
                  f"i={job['instance']:>2} {detail:<14} {evals:<11} "
                  f"{target.relative_to(ROOT)}")
        print(f"{len(jobs)} jobs")
        return

    failure = run_jobs(jobs, args.workers, args.jobs)
    if failure:
        raise SystemExit(failure)

    # Integrity gate on the main grid, once the whole campaign has finished.
    gate = subprocess.run(
        [PYTHON, "pq_validate.py", "--allow-missing"], cwd=ROOT
    )
    print("integrity gate:", "ok" if gate.returncode == 0 else "FAILED")


if __name__ == "__main__":
    main()
