"""Consolidation-campaign runner: the registered consolidation items
plus the robustness block.

Declarative job list -> sequential pq_experiment invocations, resumable
(skips any job whose output npz already exists), one log per block,
integrity gate (pq_validate --allow-missing) after each block.

Usage (from the repo root):
    python cloud/runner.py --mini              # miniature end-to-end dry run
    python cloud/runner.py --list              # show the job table
    python cloud/runner.py --workers 60        # full campaign
    python cloud/runner.py --only pq_consolidation --workers 60
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


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

    # 3. Probe dependence of the moment hierarchy, up to k = 4.
    for n in (6, 8):
        jobs.append(dict(
            block="moment_probe", kind="cmd", n=n, instance=0,
            output=f"analysis/moment_probe_n{n}.log",
            argv=["analysis/moment_probe_scan.py", "--sizes", str(n),
                  "--max-k", "4", "--solutions", "2"],
            max_workers=24,
        ))

    # 4. The 240-point kernel residuals against Eq. (5), with bootstrap
    #    errors: the number the paper quotes, with a committed log at last.
    for n in (8, 10):
        jobs.append(dict(
            block="kernel_residuals", kind="cmd", n=n, instance=0,
            output=f"analysis/kernel_exact_residuals_n{n}.log",
            argv=["analysis/check_kernel_exact_residuals.py", "--sizes", str(n)],
        ))

    return jobs


def output_path(job: dict) -> Path:
    if job.get("kind") == "cmd":
        return ROOT / job["output"]
    scale = job.get("init_scale", 0.1)
    return ROOT / job["output"] / (
        f"pq_n{job['n']}_i{job['instance']}_sigma{scale:g}.npz"
    )


def command(job: dict, workers: int) -> list[str]:
    # A job may cap its own parallelism: the k = 4 moment transfer holds a
    # chunked configuration sum per worker and is memory-bound, not
    # core-bound, so it must not inherit a 96-way pool.
    workers = min(workers, job.get("max_workers", workers))
    if job.get("kind") == "cmd":
        return [PYTHON, *job["argv"], "--workers", str(workers)]
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
    return cmd


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
    args = parser.parse_args()

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
            done = "done" if output_path(job).exists() else "    "
            detail = (f"restarts={job['restarts']}" if "restarts" in job
                      else Path(job["argv"][0]).name)
            print(f"[{done}] {job['block']:>18} n={job['n']:>2} "
                  f"i={job['instance']:>2} {detail}")
        return

    blocks_seen = []
    for job in jobs:
        target = output_path(job)
        if target.exists():
            print(f"skip (exists): {target.name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = command(job, args.workers)
        print(f"run: {' '.join(cmd[1:])}", flush=True)
        start = time.perf_counter()
        if target.suffix == ".log":
            # The log IS the artifact. Write beside it and move into place
            # only on success, so a failed run leaves no skip marker.
            staging = target.with_suffix(".log.partial")
            with open(staging, "w") as handle:
                result = subprocess.run(cmd, cwd=ROOT, stdout=handle,
                                        stderr=subprocess.STDOUT)
            if result.returncode == 0:
                staging.replace(target)
        else:
            log = target.parent / "runner.log"
            with open(log, "a") as handle:
                result = subprocess.run(cmd, cwd=ROOT, stdout=handle,
                                        stderr=subprocess.STDOUT)
        status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
        print(f"  -> {status} [{time.perf_counter() - start:.0f} s]", flush=True)
        if result.returncode != 0:
            raise SystemExit(f"job failed, see {log}")
        if job["block"] not in blocks_seen:
            blocks_seen.append(job["block"])

    # Integrity gate on the main grid after the campaign.
    gate = subprocess.run(
        [PYTHON, "pq_validate.py", "--allow-missing"], cwd=ROOT
    )
    print("integrity gate:", "ok" if gate.returncode == 0 else "FAILED")


if __name__ == "__main__":
    main()
