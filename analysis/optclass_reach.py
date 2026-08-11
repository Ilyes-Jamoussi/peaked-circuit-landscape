"""Does any distinct in-class optimizer reach further than the protocol's Adam?

Registered prediction 3 said the reachable profile is optimizer-independent
within the stable-local class. Plain SGD, inside the class as registered,
falsified it downward at the frozen budget. What remains open is the upward
question, the one the hardness claims stand on: whether Adam's reach is the
class's reach or merely Adam's. REGISTRATION-CONVERGED.md registers the
``optclass`` block to answer it -- optimizer in {Adam, L-BFGS-B} crossed with
initialization in {normal, Haar}, instances 0-2 per size, B = 16 restarts at
the matched 13,200-evaluation budget -- and fixes what each outcome costs
before the data exist:

  (a) rho <= 1 within errors at every size: the ceiling is the class's.
  (b) rho > 1 at 2 sigma at some size, flat in n: the level is Adam's; every
      absolute reach value is relabelled, the comparison with the fitted base
      of Ref. [aaronson2024peaked] is withdrawn, the scaling claims survive.
  (c) rho > 1 and growing with n: the third registered falsifier of
      C-hardness is triggered by our own measurement, and C-hardness is
      withdrawn, not weakened.

The statistic is the reach ratio at matched instances and matched budget,
rho_arm(n) = R_arm(n) / R_conv(n), each side read at B0 = 16 through the exact
without-replacement estimator of budget_scan (on the 16-restart arms that
estimator IS the sample max; on the 32-restart converged cells it is a genuine
expectation -- same estimand, different variance). The registered wording
("with the standard error over those three instances") admits two
constructions, so both are computed and printed:

  primary  -- paired: the ratio is formed per instance, r_i = R_arm,i /
              R_conv,i, and rho is the mean of the three r_i with their
              standard error. This is the reading that honours "at matched
              instances"; it cancels the instance-to-instance level spread
              and therefore makes branches (b) and (c) EASIER to trigger.
  secondary -- literal: rho = mean(R_arm) / mean(R_conv), relative errors
              added in quadrature.

Tie-break, fixed here before any archive exists: if the two constructions
select different branches, the block verdict is the branch that costs the
paper MORE (order of cost (c) > (b) > (a)), and both constructions are
reported either way. Growth is the registered "its own log-step":
S = ln rho(n_max) - ln rho(n_min), growing iff S - 2 SE(S) > 0, with a
weighted-least-squares slope of ln rho on n printed as a secondary check.
"Above" at a size means rho - 2 SE(rho) > 1. Branches are evaluated per arm
in the order (c), (b), (a), first that applies; the block verdict is the most
costly branch any single arm triggers, since one arm outreaching Adam settles
the question the block asks.

A null from a broken arm is worth nothing, so arm-health diagnostics print
beside every rho: the stop-reason histogram, evaluations consumed against the
13,200 ceiling, the delta spread, and the fraction of restarts peaking on the
0^n string. An arm that fails these is reported as uninformative, not as
evidence.

Self-tests: the estimator is exact on a constant ensemble and reduces to the
sample max at 16 of 16; an arm identical to its denominator lands in branch
(a), a flat 1.2x arm in (b), a growing arm in (c); an under-sized arm archive,
a missing matched denominator, and a broken L-BFGS sentinel are each refused
by name.

Usage:
    python analysis/optclass_reach.py    # seconds; partial grids reported
"""

from __future__ import annotations

import glob
import math
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from budget_scan import expected_max  # noqa: E402
from converged_reach import INSTRUMENTATION_KEYS, weighted_fit  # noqa: E402

OPTCLASS = ROOT / "results" / "optclass"
CONVERGED = ROOT / "results" / "converged"
SGD_CONVERGED = ROOT / "results" / "sgd_converged"
FROZEN_SGD = ROOT / "results" / "robustness" / "sgd"

#: The three arms of the registered 2 x 2; the fourth cell, (Adam, normal),
#: is the converged grid itself read at B0 on the same instances.
ARMS = (
    ("lbfgs_sigma", "L-BFGS-B, sigma init", "lbfgs", "normal"),
    ("lbfgs_haar", "L-BFGS-B, Haar init", "lbfgs", "haar"),
    ("adam_haar", "Adam, Haar init", "adam", "haar"),
)
INSTANCES = (0, 1, 2)
ARM_RESTARTS = 16
CONVERGED_RESTARTS = 32
B0 = 16
MATCHED_EVALUATIONS = 13200
LBFGS_LEGACY_STOP = -1

#: Cost order of the registered branches; the tie-break and the block verdict
#: both take a maximum over this ordering.
BRANCH_COST = {"a": 0, "b": 1, "c": 2}
BRANCH_TEXT = {
    "a": "rho <= 1 within errors at every size: the ceiling is the class's, "
         "not Adam's",
    "b": "rho > 1 at 2 sigma at some size, flat in n: the level is "
         "optimizer-dependent; absolute reach values are relabelled as "
         "Adam's and the comparison with the fitted base of "
         "Ref. [aaronson2024peaked] is withdrawn",
    "c": "rho > 1 and growing with n: the third registered falsifier of "
         "C-hardness is triggered; C-hardness is withdrawn, not weakened",
}


def scalar(blob, key: str) -> str:
    """A 0-d metadata entry as a plain string."""
    return str(np.asarray(blob[key]).item())


def load_arm(tag: str, optimizer: str, init_mode: str,
             root: Path | None = None) -> dict[tuple[int, int], dict]:
    """Arm archives keyed (n, instance), gated on identity and sentinels.

    The three arm directories share one file-name convention with every other
    grid in the corpus, so the directory proves nothing about what an archive
    is; the ``optimizer`` and ``init_mode`` metadata written by
    pq_experiment are the identity, and an archive whose identity does not
    match its directory is refused by name rather than averaged into the
    wrong cell. For the L-BFGS arms the legacy fields are undefined by
    construction, so the only legal readout is the sentinel triple
    (legacy_stop_step = -1, legacy_weight = NaN, polish_gain = 0); anything
    else means the readout came from somewhere it could not have.
    """
    cells: dict[tuple[int, int], dict] = {}
    base = OPTCLASS if root is None else root
    for name in sorted(glob.glob(str(base / tag / "pq_n*_sigma0.1.npz"))):
        blob = np.load(name)
        short = Path(name).name
        missing = [k for k in INSTRUMENTATION_KEYS if k not in blob.files]
        if missing:
            raise ValueError(
                f"{short} ({tag}): converged instrumentation key "
                f"'{missing[0]}' is absent; a frozen-protocol archive cannot "
                "enter the optimizer-class block."
            )
        got_optimizer = scalar(blob, "optimizer")
        got_init = scalar(blob, "init_mode")
        if (got_optimizer, got_init) != (optimizer, init_mode):
            raise ValueError(
                f"{short}: arm directory '{tag}' expects "
                f"({optimizer}, {init_mode}) but the archive says "
                f"({got_optimizer}, {got_init}); refusing to average it "
                "into the wrong cell."
            )
        values = blob["peak_weights"]
        if len(values) < ARM_RESTARTS:
            raise ValueError(
                f"{short} ({tag}): {len(values)} restarts, the registered "
                f"block requires {ARM_RESTARTS}; an under-sized cell would "
                "enter rho with a hidden variance advantage."
            )
        if optimizer == "lbfgs":
            stop = blob["legacy_stop_step"]
            legacy = blob["legacy_weight"]
            polish = blob["polish_gain"]
            if not (bool(np.all(stop == LBFGS_LEGACY_STOP))
                    and bool(np.all(np.isnan(legacy)))
                    and bool(np.all(polish == 0.0))):
                raise ValueError(
                    f"{short} ({tag}): L-BFGS sentinel triple broken "
                    f"(legacy_stop_step={LBFGS_LEGACY_STOP}, "
                    "legacy_weight=NaN, polish_gain=0 expected); the "
                    "readout is not this arm's."
                )
        num_qubits = int(re.search(r"pq_n(\d+)_i(\d+)_", short).group(1))
        instance = int(re.search(r"pq_n(\d+)_i(\d+)_", short).group(2))
        cells[(num_qubits, instance)] = {
            "peak_weights": values,
            "num_steps": blob["num_steps"],
            "stop_reasons": np.asarray(blob["stop_reasons"]).tolist(),
            "argmax_indices": blob["argmax_indices"],
            "baseline": float(np.asarray(blob["baseline_peak_weight"])),
        }
    return cells


def load_denominators(sizes) -> dict[tuple[int, int], np.ndarray]:
    """The (Adam, normal) cell: converged archives on the matched instances.

    A missing matched instance is refused by name, never dropped: the primary
    statistic is paired, and silently shrinking the pair set would change
    what rho estimates.
    """
    cells: dict[tuple[int, int], np.ndarray] = {}
    for num_qubits in sizes:
        for instance in INSTANCES:
            path = CONVERGED / f"pq_n{num_qubits}_i{instance}_sigma0.1.npz"
            if not path.exists():
                raise ValueError(
                    f"denominator missing: {path.name} -- the arm holds "
                    f"(n={num_qubits}, i={instance}) but the converged "
                    "grid does not; the paired statistic cannot be formed."
                )
            blob = np.load(path)
            missing = [k for k in INSTRUMENTATION_KEYS if k not in blob.files]
            if missing:
                raise ValueError(
                    f"{path.name}: converged instrumentation key "
                    f"'{missing[0]}' is absent; the denominator must be "
                    "converged data."
                )
            if (scalar(blob, "optimizer"), scalar(blob, "init_mode")) != (
                    "adam", "normal"):
                raise ValueError(
                    f"{path.name}: the denominator cell must be "
                    "(adam, normal)."
                )
            values = blob["peak_weights"]
            if len(values) < CONVERGED_RESTARTS:
                raise ValueError(
                    f"{path.name}: {len(values)} restarts, expected "
                    f"{CONVERGED_RESTARTS}."
                )
            cells[(num_qubits, instance)] = values
    return cells


def rho_paired(arm: dict, denominator: dict, num_qubits: int
               ) -> tuple[float, float]:
    """Mean and SE over instances of the per-instance reach ratio."""
    ratios = np.array([
        expected_max(arm[(num_qubits, i)]["peak_weights"], B0)
        / expected_max(denominator[(num_qubits, i)], B0)
        for i in INSTANCES
    ])
    return float(ratios.mean()), float(
        ratios.std(ddof=1) / math.sqrt(len(ratios)))


def rho_literal(arm: dict, denominator: dict, num_qubits: int
                ) -> tuple[float, float]:
    """mean(R_arm)/mean(R_conv) with relative errors in quadrature."""
    top = np.array([expected_max(arm[(num_qubits, i)]["peak_weights"], B0)
                    for i in INSTANCES])
    bottom = np.array([expected_max(denominator[(num_qubits, i)], B0)
                       for i in INSTANCES])
    count = math.sqrt(len(INSTANCES))
    value = top.mean() / bottom.mean()
    relative = math.hypot(top.std(ddof=1) / count / top.mean(),
                          bottom.std(ddof=1) / count / bottom.mean())
    return float(value), float(value * relative)


def growth(series: dict[int, tuple[float, float]]
           ) -> tuple[float, float] | tuple[None, None]:
    """S = ln rho(n_max) - ln rho(n_min) and its propagated error."""
    if len(series) < 2:
        return None, None
    low, high = min(series), max(series)
    (rho_low, se_low), (rho_high, se_high) = series[low], series[high]
    return (float(math.log(rho_high) - math.log(rho_low)),
            float(math.hypot(se_high / rho_high, se_low / rho_low)))


def wls_slope(series: dict[int, tuple[float, float]]
              ) -> tuple[float, float] | tuple[None, None]:
    """Secondary check: WLS slope of ln rho on n, with its error."""
    if len(series) < 3:
        return None, None
    sizes = np.array(sorted(series), dtype=float)
    rho = np.array([series[int(n)][0] for n in sizes])
    se = np.array([series[int(n)][1] for n in sizes])
    if np.any(se <= 0):
        se = np.maximum(se, 1e-12)
    _, slope, slope_err, _ = weighted_fit(sizes, np.log(rho), se / rho)
    return float(slope), float(slope_err)


def branch_of(series: dict[int, tuple[float, float]]) -> dict:
    """The registered branch for one arm, evaluated (c) -> (b) -> (a)."""
    above = sorted(n for n, (rho, se) in series.items() if rho - 2 * se > 1.0)
    s_value, s_error = growth(series)
    growing = (s_value is not None and s_value - 2 * s_error > 0.0)
    if above and growing:
        letter = "c"
    elif above:
        letter = "b"
    else:
        letter = "a"
    return {"branch": letter, "above": above, "S": s_value,
            "S_err": s_error, "growing": growing}


def health(arm: dict, num_qubits: int) -> dict:
    """Pooled arm diagnostics at one size; a broken arm must not read as a null."""
    weights = np.concatenate([arm[(num_qubits, i)]["peak_weights"]
                              for i in INSTANCES])
    steps = np.concatenate([arm[(num_qubits, i)]["num_steps"]
                            for i in INSTANCES])
    reasons = Counter(sum((arm[(num_qubits, i)]["stop_reasons"]
                           for i in INSTANCES), []))
    argmax = np.concatenate([arm[(num_qubits, i)]["argmax_indices"]
                             for i in INSTANCES])
    return {
        "stop": dict(sorted(reasons.items())),
        "median_evals": float(np.median(steps)),
        "max_evals": int(steps.max()),
        "best": float(weights.max()),
        "mean": float(weights.mean()),
        "min": float(weights.min()),
        "peak_on_zero": float(np.mean(argmax == 0)),
    }


def synthetic_series(levels: dict[int, tuple[float, ...]]
                     ) -> dict[int, tuple[float, float]]:
    """Per-size (rho, SE) from explicit per-instance ratios, for the tests."""
    series = {}
    for num_qubits, ratios in levels.items():
        values = np.array(ratios, dtype=float)
        series[num_qubits] = (float(values.mean()), float(
            values.std(ddof=1) / math.sqrt(len(values))))
    return series


def self_tests() -> None:
    constant = np.full(ARM_RESTARTS, 0.375)
    assert abs(expected_max(constant, B0) - 0.375) < 1e-12
    sample = np.random.default_rng(11).exponential(size=ARM_RESTARTS)
    assert abs(expected_max(sample, B0) - sample.max()) < 1e-12, (
        "E[max of 16] on 16 restarts must be the sample max"
    )

    # Branch logic on synthetic ratio series: identical arms are (a), a flat
    # excess is (b), a growing excess is (c). The jitter keeps the SEs finite
    # so the 2-sigma thresholds are exercised rather than degenerate.
    flat_one = synthetic_series({8: (1.0, 1.001, 0.999),
                                 10: (0.999, 1.0, 1.001),
                                 12: (1.0, 0.998, 1.002)})
    assert branch_of(flat_one)["branch"] == "a", branch_of(flat_one)
    flat_high = synthetic_series({8: (1.2, 1.21, 1.19),
                                  10: (1.2, 1.19, 1.21),
                                  12: (1.21, 1.2, 1.19)})
    assert branch_of(flat_high)["branch"] == "b", branch_of(flat_high)
    rising = synthetic_series({8: (1.0, 1.01, 0.99),
                               10: (1.15, 1.16, 1.14),
                               12: (1.4, 1.41, 1.39)})
    assert branch_of(rising)["branch"] == "c", branch_of(rising)

    # Loader gates, each refused by name: an under-sized arm cell, a missing
    # matched denominator, and a broken L-BFGS sentinel.
    base = {
        "peak_weights": np.full(ARM_RESTARTS, 0.3),
        "num_steps": np.full(ARM_RESTARTS, 5000, dtype=np.int64),
        "stop_reasons": np.array(["converged"] * ARM_RESTARTS, dtype="<U12"),
        "argmax_indices": np.zeros(ARM_RESTARTS, dtype=np.int64),
        "baseline_peak_weight": 2.0 ** -8,
        "optimizer": "lbfgs",
        "init_mode": "normal",
        "ladder_weights": np.full((ARM_RESTARTS, 6), 0.3),
        "running_max": np.full(ARM_RESTARTS, 0.3),
        "legacy_weight": np.full(ARM_RESTARTS, np.nan),
        "legacy_stop_step": np.full(ARM_RESTARTS, LBFGS_LEGACY_STOP,
                                    dtype=np.int64),
        "polish_gain": np.zeros(ARM_RESTARTS),
    }
    global OPTCLASS, CONVERGED
    kept_optclass, kept_converged = OPTCLASS, CONVERGED
    try:
        with tempfile.TemporaryDirectory() as scratch:
            OPTCLASS = Path(scratch) / "optclass"
            CONVERGED = Path(scratch) / "converged"
            (OPTCLASS / "lbfgs_sigma").mkdir(parents=True)
            CONVERGED.mkdir()
            name = "pq_n8_i0_sigma0.1.npz"

            short = dict(base, peak_weights=np.full(8, 0.3))
            np.savez_compressed(OPTCLASS / "lbfgs_sigma" / name, **short)
            try:
                load_arm("lbfgs_sigma", "lbfgs", "normal")
            except ValueError as refusal:
                assert "8 restarts" in str(refusal), refusal
            else:
                raise AssertionError("an under-sized arm cell was accepted")

            broken = dict(base, legacy_stop_step=np.full(
                ARM_RESTARTS, 400, dtype=np.int64))
            np.savez_compressed(OPTCLASS / "lbfgs_sigma" / name, **broken)
            try:
                load_arm("lbfgs_sigma", "lbfgs", "normal")
            except ValueError as refusal:
                assert "sentinel" in str(refusal), refusal
            else:
                raise AssertionError("a broken L-BFGS sentinel was accepted")

            np.savez_compressed(OPTCLASS / "lbfgs_sigma" / name, **base)
            arm = load_arm("lbfgs_sigma", "lbfgs", "normal")
            assert (8, 0) in arm
            try:
                load_denominators({8})
            except ValueError as refusal:
                assert "denominator missing" in str(refusal), refusal
            else:
                raise AssertionError("a missing denominator was not refused")
    finally:
        OPTCLASS, CONVERGED = kept_optclass, kept_converged

    print("self-tests passed (E[max of 16] is the sample max on 16 restarts "
          "and exact on a constant; identical arm -> branch (a), flat 1.2x "
          "-> (b), growing -> (c); an under-sized cell, a broken L-BFGS "
          "sentinel and a missing matched denominator are each refused by "
          "name)")


def disclosures(arms: dict, denominator: dict, verdicts: dict) -> None:
    """Post-verdict disclosures. Nothing here enters the registered rule.

    Three facts a reader needs to weigh the verdict, printed rather than left
    for a referee to derive:

    (i)  rho(8) = 1 exactly for every arm -- at n = 8 each arm reaches each
         instance's ceiling -- so S = ln rho(n_max) - ln rho(8) collapses to
         ln rho(n_max), and the 'growing at 2 sigma' test is algebraically
         the 'above 1 at 2 sigma at the largest size' test. One exceedance,
         at the largest size, on an exact anchor: one fact, not two. The WLS
         slope over all sizes inherits the anchor (a zero SE enters with
         weight ~1e24), so it is recomputed here without the anchor.
    (ii) The SE at the deciding size is over three instances, i.e. two
         degrees of freedom: 2 SE corresponds to ~82% two-sided confidence
         (t_{0.975,2} = 4.30), not 95%. The registered threshold is 2 SE and
         it applies as registered; this states what it is worth.
    (iii) The per-instance ratios behind every mean, so the deciding cell is
         inspectable at the level the error bar is computed from.
    """
    print("\n5. Disclosures (post-verdict; nothing here enters the "
          "registered rule)\n")
    print("   rho(8) = 1.0000 +/- 0.0000 exactly for every arm: S equals "
          "ln rho(n_max)\n   on an exact anchor, and 'growing at 2 sigma' "
          "coincides with 'above 1 at\n   2 sigma at the largest size'. "
          "SE at each size is over 3 instances\n   (2 dof): 2 SE ~ 82% "
          "two-sided, t_{0.975,2} = 4.30.\n")
    print(f"{'arm':>12} {'n':>4}   per-instance rho (paired)")
    for tag, record in arms.items():
        for num_qubits in record["sizes"]:
            per = [expected_max(record["cells"][(num_qubits, i)]
                                ["peak_weights"], B0)
                   / expected_max(denominator[(num_qubits, i)], B0)
                   for i in INSTANCES]
            print(f"{tag:>12} {num_qubits:>4}   "
                  + "  ".join(f"{v:.4f}" for v in per))
    print(f"\n{'arm':>12}   WLS slope of ln rho, anchor dropped (n >= 10)")
    for tag, record in arms.items():
        series = {n: rho_paired(record["cells"], denominator, n)
                  for n in record["sizes"] if n >= 10}
        slope, err = wls_slope(series)
        text = "-" if slope is None else f"{slope:+.4f} +/- {err:.4f}"
        print(f"{tag:>12}   {text}")


ARCH_CONTROL = ROOT / "results" / "optclass_arch"
ARCH_CONTROL_SIZE = 12


def arch_control(denominator_all: dict | None = None) -> None:
    """The architecture control, read under the rule committed before it ran.

    The n = 8-14 arms were computed on Apple silicon and the n = 16 cells on
    x86, with the converged denominator all x86: the one size whose ratio is
    within a single architecture is the size that decides the verdict, and
    the switch falls between n = 14 and n = 16. This control recomputes the
    deciding arm (lbfgs_sigma) at n = 12 on x86, identical settings, and the
    reading rule -- fixed in cloud/runner.py before any control data existed
    -- is z = |ln rho_x86 - ln rho_ARM| / hypot(SE_x86, SE_ARM) on the
    paired construction. z < 2 closes the confound; z >= 2 puts the
    optimizer-class verdict under an explicit architecture reservation.
    """
    print("\n6. Architecture control (lbfgs_sigma at n = 12, x86 vs "
          "Apple silicon)")
    if not ARCH_CONTROL.exists():
        print("   (no control archives yet; run "
              "cloud/runner.py --only optclass_arch)")
        return
    x86 = load_arm("lbfgs_sigma", "lbfgs", "normal", root=ARCH_CONTROL)
    if not all((ARCH_CONTROL_SIZE, i) in x86 for i in INSTANCES):
        print("   (control incomplete; it decides nothing yet)")
        return
    arm = load_arm("lbfgs_sigma", "lbfgs", "normal")
    denominator = (denominator_all if denominator_all is not None
                   else load_denominators({ARCH_CONTROL_SIZE}))
    rho_x, se_x = rho_paired(x86, denominator, ARCH_CONTROL_SIZE)
    rho_a, se_a = rho_paired(arm, denominator, ARCH_CONTROL_SIZE)
    z = (abs(math.log(rho_x) - math.log(rho_a))
         / math.hypot(se_x / rho_x, se_a / rho_a))
    print(f"   rho_x86({ARCH_CONTROL_SIZE})  = {rho_x:.4f} +/- {se_x:.4f}"
          f"   per instance "
          + " ".join(f"{expected_max(x86[(ARCH_CONTROL_SIZE, i)]['peak_weights'], B0) / expected_max(denominator[(ARCH_CONTROL_SIZE, i)], B0):.4f}"
                     for i in INSTANCES))
    print(f"   rho_ARM({ARCH_CONTROL_SIZE})  = {rho_a:.4f} +/- {se_a:.4f}")
    print(f"   z = |ln rho_x86 - ln rho_ARM| / hypot(SE) = {z:.2f}")
    if z < 2.0:
        print("   CONTROL VERDICT: z < 2 -- the architecture switch is "
              "within noise at the\n   controlled size; the confound "
              "between architecture and size does not\n   carry the "
              "optimizer-class verdict.")
    else:
        print("   CONTROL VERDICT: z >= 2 -- the architecture switch moves "
              "rho by more than\n   its combined error; the optimizer-class "
              "verdict must carry an explicit\n   architecture reservation "
              "in the manuscript.")


def exploratory_sgd() -> None:
    """The non-registered matched-budget SGD run, reported as exploratory."""
    path = SGD_CONVERGED / "pq_n10_i0_sigma0.1.npz"
    print("\n4. Exploratory (NOT registered): SGD at the matched budget\n")
    if not path.exists():
        print(f"   {path.relative_to(ROOT)} not found; run\n"
              "     python pq_experiment.py --num-qubits 10 --instance 0 "
              "--restarts 16 \\\n       --converged --optimizer sgd "
              "--output results/sgd_converged")
        return
    blob = np.load(path)
    assert scalar(blob, "optimizer") == "sgd", "the exploratory cell is SGD"
    values = blob["peak_weights"]
    steps = blob["num_steps"]
    reasons = Counter(np.asarray(blob["stop_reasons"]).tolist())
    reach_sgd = expected_max(values, B0)
    conv = np.load(CONVERGED / "pq_n10_i0_sigma0.1.npz")["peak_weights"]
    reach_adam = expected_max(conv, B0)
    frozen = sorted(glob.glob(str(FROZEN_SGD / "pq_n10_i0_*.npz")))
    frozen_reach = (expected_max(np.load(frozen[0])["peak_weights"], B0)
                    if frozen else None)
    print(f"   n = 10, instance 0, B = {B0}, budget "
          f"{MATCHED_EVALUATIONS} evaluations (33x the frozen 400)")
    print(f"   R_sgd(B0={B0})          = {reach_sgd:.4f}   "
          f"(raw best of {len(values)}: {values.max():.4f})")
    print(f"   matched Adam cell       = {reach_adam:.4f}   "
          f"ratio {reach_sgd / reach_adam:.3f}")
    if frozen_reach is not None:
        print(f"   frozen SGD (400 steps)  = {frozen_reach:.4f}   "
              f"matched-budget gain x{reach_sgd / frozen_reach:.2f}")
    print(f"   stop reasons {dict(sorted(reasons.items()))}, median "
          f"evaluations {np.median(steps):.0f} of {MATCHED_EVALUATIONS}")


def main() -> None:
    self_tests()

    arms = {}
    for tag, label, optimizer, init_mode in ARMS:
        cells = load_arm(tag, optimizer, init_mode)
        sizes = sorted({n for (n, _) in cells})
        complete = {n for n in sizes
                    if all((n, i) in cells for i in INSTANCES)}
        partial = sorted(set(sizes) - complete)
        if partial:
            print(f"note: {tag} has incomplete sizes {partial} "
                  "(campaign in flight); they enter nothing below.")
        arms[tag] = {"label": label, "cells": cells,
                     "sizes": sorted(complete)}

    covered = sorted({n for record in arms.values() for n in record["sizes"]})
    if not covered:
        print(f"\nNo complete optclass cells under {OPTCLASS} yet; run\n"
              "  python cloud/runner.py --only optclass")
        exploratory_sgd()
        return
    denominator = load_denominators(covered)

    print("\n1. Denominator: the (Adam, normal) converged cell at "
          f"B0 = {B0}, instances 0-2\n")
    print(f"{'n':>4} {'R_conv':>8} {'+/-':>8}   per instance")
    for num_qubits in covered:
        per = np.array([expected_max(denominator[(num_qubits, i)], B0)
                        for i in INSTANCES])
        print(f"{num_qubits:>4} {per.mean():>8.4f} "
              f"{per.std(ddof=1) / math.sqrt(len(per)):>8.4f}   "
              + " ".join(f"{v:.4f}" for v in per))

    print("\n2. Arm health (a broken arm reads as 'uninformative', "
          "never as a null)\n")
    print(f"{'arm':>12} {'n':>4} {'best':>7} {'mean':>7} {'min':>8} "
          f"{'on-0^n':>7} {'med.ev':>7} {'max.ev':>7}  stop reasons")
    for tag, record in arms.items():
        for num_qubits in record["sizes"]:
            info = health(record["cells"], num_qubits)
            print(f"{tag:>12} {num_qubits:>4} {info['best']:>7.4f} "
                  f"{info['mean']:>7.4f} {info['min']:>8.2e} "
                  f"{info['peak_on_zero']:>7.2f} "
                  f"{info['median_evals']:>7.0f} {info['max_evals']:>7} "
                  f" {info['stop']}")

    print("\n3. The reach ratio rho_arm(n), both registered constructions\n")
    print(f"{'arm':>12} {'n':>4} {'paired':>8} {'+/-':>7} "
          f"{'literal':>8} {'+/-':>7}")
    verdicts = {}
    for tag, record in arms.items():
        if not record["sizes"]:
            print(f"{tag:>12}    (no complete sizes yet; this arm decides "
                  "nothing)")
            continue
        paired_series, literal_series = {}, {}
        for num_qubits in record["sizes"]:
            paired_series[num_qubits] = rho_paired(
                record["cells"], denominator, num_qubits)
            literal_series[num_qubits] = rho_literal(
                record["cells"], denominator, num_qubits)
            p_rho, p_se = paired_series[num_qubits]
            l_rho, l_se = literal_series[num_qubits]
            print(f"{tag:>12} {num_qubits:>4} {p_rho:>8.4f} {p_se:>7.4f} "
                  f"{l_rho:>8.4f} {l_se:>7.4f}")
        paired_verdict = branch_of(paired_series)
        literal_verdict = branch_of(literal_series)
        chosen = max((paired_verdict, literal_verdict),
                     key=lambda v: BRANCH_COST[v["branch"]])
        slope, slope_err = wls_slope(paired_series)
        verdicts[tag] = {"paired": paired_verdict, "literal": literal_verdict,
                         "chosen": chosen, "slope": slope,
                         "slope_err": slope_err}
        s_text = ("-" if paired_verdict["S"] is None else
                  f"{paired_verdict['S']:+.4f} +/- "
                  f"{paired_verdict['S_err']:.4f}")
        s_lit = ("-" if literal_verdict["S"] is None else
                 f"{literal_verdict['S']:+.4f} +/- "
                 f"{literal_verdict['S_err']:.4f}")
        print(f"{'':>12} S(literal) = {s_lit}")
        w_text = ("-" if slope is None
                  else f"{slope:+.4f} +/- {slope_err:.4f}")
        disagree = (paired_verdict["branch"] != literal_verdict["branch"])
        print(f"{'':>12} S(paired) = {s_text}; WLS slope of ln rho = "
              f"{w_text}; above at 2 sigma: "
              f"{paired_verdict['above'] or 'no size'}; branch "
              f"paired ({paired_verdict['branch']}) / literal "
              f"({literal_verdict['branch']})"
              + (" -> tie-break to the costlier "
                 f"({chosen['branch']})" if disagree else ""))

    exploratory_sgd()

    if not verdicts:
        print("\nNo arm holds a complete size yet; no verdict.")
        return
    block = max(verdicts.values(),
                key=lambda v: BRANCH_COST[v["chosen"]["branch"]])
    letter = block["chosen"]["branch"]
    trigger = [tag for tag, v in verdicts.items()
               if v["chosen"]["branch"] == letter]
    missing_sizes = sorted(set((8, 10, 12, 14, 16)) - set(covered))
    print(f"\nVERDICT (registered branch {letter}): {BRANCH_TEXT[letter]}."
          f"\n  decided by arm(s) {', '.join(trigger)} on sizes {covered};"
          f"\n  registered sizes not measured here: "
          f"{missing_sizes or 'none'} (disclosed in the manuscript, "
          "App. B).")

    disclosures(arms, denominator, verdicts)
    arch_control(denominator_all=(
        denominator if ARCH_CONTROL_SIZE in covered else None))


if __name__ == "__main__":
    main()
