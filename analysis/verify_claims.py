"""Recompute the manuscript's quoted numbers from the archives.

The README promises that every number in the paper traces to a committed
log. This script makes that checkable rather than asserted: it recomputes
each recomputable claim from results/ and compares against the value the
manuscript prints, to the precision the manuscript prints it.

A claim that cannot be recomputed from the archives (an exact transfer, a
proof constant, a value read from a scan log) is listed as such rather
than silently omitted, so the gap between "checked here" and "checked
elsewhere" stays visible.

Usage:
    python analysis/verify_claims.py        # ~2 min
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from budget_scan import slope_vs_logb  # noqa: E402

RESULTS = ROOT / "results"
FAILURES: list[str] = []
CHECKED = 0


def claim(label: str, computed, quoted, tolerance: float) -> None:
    """Compare a recomputed value against the number the manuscript prints."""
    global CHECKED
    CHECKED += 1
    ok = abs(computed - quoted) <= tolerance
    status = "ok " if ok else "FAIL"
    print(f"  [{status}] {label:<52} computed {computed:>10.4g}  "
          f"manuscript {quoted:>10.4g}")
    if not ok:
        FAILURES.append(f"{label}: computed {computed:.6g}, quoted {quoted:g}")


def mean_of_best():
    best: dict[int, list[float]] = {}
    for name in sorted(glob.glob(str(RESULTS / "pq/pq_n*_sigma0.1.npz"))):
        match = re.search(r"pq_n(\d+)_i(\d+)_", name)
        values = np.load(name)["peak_weights"]
        if len(values) < 200:
            continue
        best.setdefault(int(match.group(1)), []).append(float(values.max()))
    sizes = np.array(sorted(best), dtype=float)
    mean = np.array([np.mean(best[int(n)]) for n in sizes])
    stderr = np.array([np.std(best[int(n)], ddof=1) / np.sqrt(len(best[int(n)]))
                       for n in sizes])
    return sizes, mean, stderr, best


def weighted_line(x, y, sy):
    w = 1.0 / sy**2
    s, sx, sxx = w.sum(), (w * x).sum(), (w * x * x).sum()
    sy_, sxy = (w * y).sum(), (w * x * y).sum()
    det = s * sxx - sx**2
    slope = (s * sxy - sx * sy_) / det
    intercept = (sxx * sy_ - sx * sxy) / det
    residual = y - (intercept + slope * x)
    return intercept, slope, np.sqrt(s / det), float((w * residual**2).sum()), det, s, sx


def section_reach():
    print("\nSec. VI, reach law")
    sizes, mean, stderr, best = mean_of_best()
    y, sy = np.log(mean), stderr / mean

    _, slope, slope_err, chi2, *_ = weighted_line(sizes, y, sy)
    claim("full-grid fixed base", float(np.exp(-slope)), 1.219, 5e-4)
    claim("full-grid base uncertainty (unscaled)",
          float(np.exp(-slope) * slope_err), 0.006, 5e-4)
    claim("full-grid chi2/dof", chi2 / (len(sizes) - 2), 7.1, 0.05)

    three = np.array([[np.load(RESULTS / f"pq/pq_n{n}_i{i}_sigma0.1.npz")
                       ["peak_weights"].max() for i in (0, 1, 2)]
                      for n in (8, 10, 12, 14)])
    m3, s3 = three.mean(axis=1), three.std(axis=1, ddof=1) / np.sqrt(3)
    n3 = np.array([8.0, 10, 12, 14])
    _, sl3, err3, chi3, *_ = weighted_line(n3, np.log(m3), s3 / m3)
    claim("three-instance base (F1)", float(np.exp(-sl3)), 1.195, 5e-4)
    claim("three-instance base uncertainty", float(np.exp(-sl3) * err3),
          0.011, 5e-4)
    claim("three-instance chi2/dof", chi3 / 2, 0.3, 0.05)

    head = sizes <= 14
    inter, sl, _, chi_head, det, s, sx = weighted_line(sizes[head], y[head],
                                                      sy[head])
    claim("n<=14 chi2", chi_head, 9.4, 0.05)
    claim("n<=14 p-value", float(1 - stats.chi2.cdf(chi_head, 2)), 0.009, 5e-4)
    predicted = inter + sl * 16.0
    claim("extrapolated delta at n=16", float(np.exp(predicted)), 0.1606, 5e-4)
    # Quoted raw in Sec. VI A and drawn as the arrow label of Fig. 2.
    claim("n=16 shortfall, own error (log units)",
          (predicted - y[-1]) / sy[-1], 4.4, 0.05)

    steps = -np.diff(y)
    step_err = np.sqrt(sy[:-1] ** 2 + sy[1:] ** 2)
    claim("first log-step", steps[0], 0.296, 5e-4)
    claim("third log-step", steps[2], 0.487, 5e-4)
    claim("steepening significance",
          (steps[2] - steps[0]) / np.sqrt(step_err[0] ** 2 + step_err[2] ** 2),
          3.0, 0.05)
    claim("local base, first interval", float(np.exp(steps[0] / 2)), 1.16, 5e-3)
    claim("local base, last interval", float(np.exp(steps[3] / 2)), 1.31, 5e-3)


def section_facts():
    print("\nSec. VI A and Fig. 4, stalled fractions and pair overlaps")
    stalled, band, pairs = [], 0, 0
    for n in (8, 10, 12, 14, 16):
        fractions = []
        for name in sorted(glob.glob(str(RESULTS / f"pq/pq_n{n}_i*_sigma0.1.npz"))):
            data = np.load(name)
            weights, overlaps = data["peak_weights"], data["overlap_matrix"]
            fractions.append(float((weights < 0.9 * weights.max()).mean()))
            keep = np.flatnonzero(weights >= 0.8 * weights.max())
            floor = np.outer(weights[keep], weights[keep])
            hat = (overlaps[np.ix_(keep, keep)] - floor) / (1 - floor)
            upper = hat[np.triu_indices(len(keep), 1)]
            if n == 8:
                band += int(((upper >= 0.25) & (upper <= 0.75)).sum())
                pairs += len(upper)
        stalled.append(float(np.mean(fractions)))
    for n, value, quoted in zip((8, 10, 12, 14, 16), stalled,
                                (0.005, 0.13, 0.44, 0.62, 0.77)):
        claim(f"stalled fraction at n={n}", value, quoted, 5e-3)
    claim("intermediate-band pairs at n=8", band, 32, 0)
    claim("pair count at n=8 (x 1e5)", pairs / 1e5, 3.6, 0.05)


def section_budget():
    print("\nSec. VI, restart budget")
    ratios = {}
    for n in (10, 12):
        high, low = [], []
        for i in (0, 1, 2):
            values = np.load(RESULTS / f"budget800/pq_n{n}_i{i}_sigma0.1.npz")[
                "peak_weights"]
            high.append(slope_vs_logb(values, (100, 200, 400, 800)))
            low.append(slope_vs_logb(values, (25, 50, 100, 200)))
        ratios[n] = float(np.mean(high) / np.mean(low))
        if n == 12:
            claim("budget-800 slope vs the null at n=12",
                  float(np.mean(high)) * 2**n, 9.0, 0.5)
    claim("within-ensemble slope ratio at n=10", ratios[10], 0.61, 0.02)
    claim("within-ensemble slope ratio at n=12", ratios[12], 0.63, 0.02)

    for n, quoted in ((12, 14.0), (14, 51.0), (16, 174.0)):
        slopes = [slope_vs_logb(np.load(f)["peak_weights"], (25, 50, 100, 200))
                  for f in sorted(glob.glob(
                      str(RESULTS / f"pq/pq_n{n}_i*_sigma0.1.npz")))]
        claim(f"median slope over the null at n={n}",
              float(np.median(slopes)) * 2**n, quoted, 1.0)


def section_step_budget():
    print("\nSec. VI and App. B, step-budget control")
    for n, quoted in ((8, 0.61), (10, 0.94), (12, 0.98), (14, 0.99), (16, 0.98)):
        at_cap, best_at_cap, instances = [], 0, 0
        for name in sorted(glob.glob(str(RESULTS / f"pq/pq_n{n}_i*_sigma0.1.npz"))):
            data = np.load(name)
            cap, taken = int(data["max_steps"]), data["num_steps"]
            at_cap.append(taken >= cap)
            best_at_cap += int(taken[int(np.argmax(data["peak_weights"]))] >= cap)
            instances += 1
        claim(f"restarts at the 400-step cap, n={n}",
              float(np.concatenate(at_cap).mean()), quoted, 0.01)
        if n == 8:
            claim("instance-best restarts at the cap, n=8",
                  best_at_cap, 3, 0)
        else:
            claim(f"instance-best restarts at the cap, n={n}",
                  best_at_cap, instances, 0)


def section_atom():
    print("\nSec. V, the atom and its disappearance")
    for n, quoted in ((8, 5.9e-7), (10, 1.8e-2), (12, 3.1e-2), (14, 4.8e-2),
                      (16, 6.0e-2)):
        spreads = []
        for name in sorted(glob.glob(str(RESULTS / f"pq/pq_n{n}_i*_sigma0.1.npz"))):
            values = np.sort(np.load(name)["peak_weights"])[::-1]
            spreads.append((values[0] - values[9]) / values[0])
        claim(f"relative top-10 spread at n={n}", float(np.median(spreads)),
              quoted, max(0.1 * quoted, 1e-8))
        if n == 8:
            claim("atomic instances at n=8 (spread < 1e-5)",
                  int((np.array(spreads) < 1e-5).sum()), 13, 0)

    shallow = [np.sort(np.load(RESULTS / f"shallow_peaking/pq_n8_i{i}_sigma0.1.npz")
                       ["peak_weights"])[::-1] for i in (0, 1, 2)]
    claim("widest 60-restart span at (8, tau_p=2), x 1e-3",
          max(float(v[0] - v[-1]) for v in shallow) * 1e3, 5.1, 0.05)


def section_corrugation():
    print("\nSec. VI C, corrugation")
    path = RESULTS / "connectivity/conn_n8_o150s1_m32.npz"
    data = np.load(path)
    ratios = np.array([data[k][2] / min(data[k][0], data[k][1])
                       for k in data.files if k.startswith("pair")
                       and k.endswith("_scalars")])
    claim("restored n=8 32-segment median rho", float(np.median(ratios)),
          0.714, 5e-4)
    claim("pairs in that archive", len(ratios), 12, 0)

    print("  -- the registered sweep (Sec. VI C) --")
    medians, adjacency = {}, {}
    for n in (8, 10, 12, 14, 16):
        per, adjacent = [], []
        for instance in (0, 1, 2):
            tag = f"_i{instance}" if instance else ""
            archive = RESULTS / f"connectivity/conn_n{n}{tag}_o150s1_m64.npz"
            data = np.load(archive)
            keys = [k for k in data.files
                    if k.startswith("pair") and k.endswith("_scalars")]
            values = [data[k][2] / min(data[k][0], data[k][1]) for k in keys]
            adjacent += [float(data[k][6]) for k in keys]
            assert len(values) == 12, (n, instance, len(values))
            per.append(float(np.median(values)))
        medians[n] = per
        adjacency[n] = adjacent

    # App. B quotes the adjacency diagnostic of the 180 paths actually used.
    for n, quoted in ((8, 0.894), (10, 0.829), (12, 0.773), (14, 0.699),
                      (16, 0.667)):
        claim(f"adjacency median at n={n}",
              float(np.median(adjacency[n])), quoted, 5e-4)
    span = np.concatenate([adjacency[n] for n in adjacency])
    claim("adjacency minimum over the 180 paths", float(span.min()), 0.562, 5e-4)
    claim("adjacency maximum over the 180 paths", float(span.max()), 0.935, 5e-4)
    claim("paths in the sweep", len(span), 180, 0)
    grid = np.array([8.0, 10, 12, 14, 16])
    mean = np.array([np.mean(medians[int(n)]) for n in grid])
    stderr = np.array([np.std(medians[int(n)], ddof=1) / np.sqrt(3)
                       for n in grid])
    for n, value, quoted in zip((8, 10, 12, 14, 16), mean,
                                (0.726, 0.560, 0.444, 0.310, 0.226)):
        claim(f"sweep mean rho at n={n}", float(value), quoted, 5e-4)
    for n, value, quoted in zip((8, 10, 12, 14, 16), stderr,
                                (0.025, 0.005, 0.014, 0.027, 0.017)):
        claim(f"sweep instance SE at n={n}", float(value), quoted, 5e-4)

    def wls(x, y, sigma):
        w = 1.0 / sigma**2
        design = np.vstack([np.ones_like(x), x]).T
        weights = np.diag(w)
        beta = np.linalg.solve(design.T @ weights @ design,
                               design.T @ weights @ y)
        residual = y - design @ beta
        return beta, float(residual @ weights @ residual)

    log_rho, sigma = np.log(mean), stderr / mean
    beta_e, chi_e = wls(grid, log_rho, sigma)
    beta_p, chi_p = wls(np.log(grid), log_rho, sigma)
    claim("sweep exponential base", float(np.exp(beta_e[1])), 0.871, 5e-4)
    claim("sweep exponential chi2", chi_e, 3.4, 0.05)
    claim("sweep exponential p", float(1 - stats.chi2.cdf(chi_e, 3)), 0.34, 5e-3)
    claim("sweep power-law exponent", float(beta_p[1]), -1.50, 5e-3)
    claim("sweep power-law chi2", chi_p, 14.4, 0.05)
    claim("sweep power-law p", float(1 - stats.chi2.cdf(chi_p, 3)), 0.0024, 5e-4)
    intervals = [-(log_rho[k + 1] - log_rho[k])
                 / (np.log(grid[k + 1]) - np.log(grid[k])) for k in range(4)]
    for k, quoted in enumerate((1.16, 1.28, 2.32, 2.38)):
        claim(f"sweep interval exponent {k + 1}", float(intervals[k]), quoted,
              5e-3)

    print("  -- the first pass, as recorded in App. C item 8 --")
    base = np.array([0.714, 0.574, 0.477, 0.237])
    sizes = np.array([8.0, 10, 12, 16])

    def rms(rho):
        y = np.log(rho)
        e = np.sqrt(((y - np.polyval(np.polyfit(sizes, y, 1), sizes)) ** 2).mean())
        p = np.sqrt(((y - np.polyval(np.polyfit(np.log(sizes), y, 1),
                                     np.log(sizes))) ** 2).mean())
        return e, p, float(np.exp(np.polyfit(sizes, y, 1)[0]))

    exp_rms, pow_rms, fitted = rms(base)
    claim("first-pass fitted base", fitted, 0.8709, 5e-4)
    claim("first-pass exponential log-RMS", exp_rms, 0.0605, 5e-4)
    claim("first-pass power-law log-RMS", pow_rms, 0.1016, 5e-4)
    claim("n=8 refinement, per cent", 100 * (1 - 0.714 / 0.846), 15.6, 0.1)

    corrected = base.copy()
    corrected[1:3] *= 0.714 / 0.846
    _, _, moved = rms(corrected)
    claim("base after correcting the interior points", moved, 0.8752, 5e-4)

    low, high = 0.70, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        trial = base.copy()
        trial[1:3] *= mid
        e, p, _ = rms(trial)
        if e < p:
            high = mid
        else:
            low = mid
    claim("correction at which the power law overtakes, per cent",
          100 * (1 - high), 14.4, 0.1)


def section_truncation():
    print("\nSec. VI and App. B, what truncation costs")
    deficits = {}
    for n in (8, 10, 12, 14, 16):
        gains = []
        for name in sorted(glob.glob(str(RESULTS / f"step_budget/pq_n{n}_i*_sigma0.1.npz"))):
            long_run = np.load(name)
            instance = int(long_run["instance_index"])
            count = len(long_run["peak_weights"])
            short = np.load(RESULTS / f"pq/pq_n{n}_i{instance}_sigma0.1.npz")
            assert np.allclose(short["thetas_init"][:count],
                               long_run["thetas_init"]), (n, instance)
            gains.append(float(long_run["peak_weights"].max()
                               / short["peak_weights"][:count].max() - 1))
        deficits[n] = float(np.mean(gains))
    for n, quoted in ((8, 0.001), (10, 0.023), (12, 0.031), (14, 0.047),
                      (16, 0.109)):
        claim(f"truncation deficit at n={n}", deficits[n], quoted, 1e-3)

    sizes, mean, stderr, _ = mean_of_best()
    y, sy = np.log(mean), stderr / mean
    corrected = np.log(mean * (1 + np.array([deficits[int(n)] for n in sizes])))
    raw_steps, cor_steps = -np.diff(y), -np.diff(corrected)
    claim("steepening statistic, raw", raw_steps[2] - raw_steps[0], 0.191, 5e-4)
    claim("steepening statistic, corrected", cor_steps[2] - cor_steps[0],
          0.197, 5e-4)

    head = sizes <= 14
    inter, sl, _, chi_head, det, s, sx = weighted_line(sizes[head], y[head],
                                                      sy[head])
    predicted = inter + sl * 16.0
    shortfall = predicted - corrected[-1]
    claim("n=16 corrected delta", float(np.exp(corrected[-1])), 0.139, 5e-4)
    claim("n=16 corrected shortfall, own error", shortfall / sy[-1], 2.5, 0.05)


def section_moments_probe():
    print("\nSec. III D, probe dependence")
    import re as _re
    for n, quoted in ((6, {"sigma=0": (0.9697, 0.8915), "sigma=0.1": (0.9715, 0.8973)}),
                      (8, {"sigma=0": (0.9810, 0.9294), "sigma=0.1": (0.9817, 0.9318)})):
        log = ROOT / f"analysis/moment_probe_n{n}.log"
        if not log.exists():
            print(f"  (no scan log for n={n})")
            continue
        for line in log.read_text().splitlines():
            f = line.split()
            if len(f) >= 7 and f[0] in quoted:
                claim(f"r3 at n={n}, {f[0]}", float(f[5]), quoted[f[0]][0], 5e-4)
                claim(f"r4 at n={n}, {f[0]}", float(f[6]), quoted[f[0]][1], 5e-4)


def main() -> None:
    print("Recomputing the manuscript's numbers from results/.")
    section_reach()
    section_facts()
    section_budget()
    section_step_budget()
    section_atom()
    section_corrugation()
    section_truncation()
    section_moments_probe()

    print(f"\n{CHECKED} claims recomputed, {len(FAILURES)} disagreement(s).")
    print("\nNot recomputable here, checked by their own scripts and logs:")
    for item in (
        "the exact kernel and its Monte-Carlo validation (kernel_exact.log)",
        "the 240-point residuals against Eq. (5) "
        "(check_kernel_exact_residuals.py)",
        "the moment hierarchy and its probe dependence (moment_probe_scan.py)",
        "the SU(4) surjectivity hypothesis (gate_surjectivity.log)",
        "the Hessian flat band and the gauge dimension (hessian_solutions.log)",
        "the geometric-measure identity (geometric_measure.log)",
        "the entanglement and MPS ceilings (ceiling_bound.log, mps_ceiling.log)",
        "the proof constants of Prop. 1 and Lemma 1 (analytic)",
    ):
        print(f"  - {item}")
    if FAILURES:
        print("\nFAILURES:")
        for failure in FAILURES:
            print(f"  {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
