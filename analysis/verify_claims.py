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

from budget_scan import expected_max, slope_vs_logb  # noqa: E402

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


def mean_of_best(published_n16: bool = False):
    """Mean-of-best per size over results/pq.

    The registered catch-up took n = 16 from four instances to eighteen,
    and REGISTRATION-CONVERGED.md requires every n = 16 number to be
    reported twice: on instances 0-3, the set the published values rest
    on, and on all eighteen. ``published_n16`` selects the first reading.
    """
    best: dict[int, list[float]] = {}
    for name in sorted(glob.glob(str(RESULTS / "pq/pq_n*_sigma0.1.npz"))):
        match = re.search(r"pq_n(\d+)_i(\d+)_", name)
        size, instance = int(match.group(1)), int(match.group(2))
        if published_n16 and size == 16 and instance > 3:
            continue
        values = np.load(name)["peak_weights"]
        if len(values) < 200:
            continue
        best.setdefault(size, []).append(float(values.max()))
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
    print("\nSec. VI, reach law (published reading: n=16 on instances 0-3)")
    sizes, mean, stderr, best = mean_of_best(published_n16=True)
    y, sy = np.log(mean), stderr / mean

    _, slope, slope_err, chi2, *_ = weighted_line(sizes, y, sy)
    claim("full-grid fixed base", float(np.exp(-slope)), 1.219, 5e-4)
    claim("full-grid base uncertainty (unscaled)",
          float(np.exp(-slope) * slope_err), 0.006, 5e-4)
    claim("full-grid chi2/dof", chi2 / (len(sizes) - 2), 7.1, 0.05)
    scaled_err = float(np.exp(-slope) * slope_err) * float(
        np.sqrt(chi2 / (len(sizes) - 2)))
    claim("full-grid base uncertainty (scaled by misfit)", scaled_err,
          0.017, 5e-4)
    claim("gap to the 1.189 base, scaled errors",
          (float(np.exp(-slope)) - 1.189) / scaled_err, 1.7, 0.05)
    claim("gap to the 1.189 base, fit's own error",
          (float(np.exp(-slope)) - 1.189)
          / float(np.exp(-slope) * slope_err), 4.6, 0.05)
    claim("their fitted law at n=16 (5e-4 x 1.189^34)",
          5e-4 * 1.189**34, 0.18, 5e-3)

    # The second registered reading: all eighteen n = 16 instances.
    sizes18, mean18, stderr18, best18 = mean_of_best()
    y18, sy18 = np.log(mean18), stderr18 / mean18
    _, slope18, slope_err18, chi218, *_ = weighted_line(sizes18, y18, sy18)
    base18 = float(np.exp(-slope18))
    claim("full-grid fixed base (all 18)", base18, 1.224, 5e-4)
    claim("full-grid chi2/dof (all 18)", chi218 / (len(sizes18) - 2),
          8.7, 0.05)
    scaled18 = float(base18 * slope_err18) * float(
        np.sqrt(chi218 / (len(sizes18) - 2)))
    claim("full-grid base uncertainty, scaled (all 18)", scaled18,
          0.018, 5e-4)
    claim("gap to the 1.189 base, scaled errors (all 18)",
          (base18 - 1.189) / scaled18, 2.0, 0.05)
    claim("gap to the 1.189 base, fit's own error (all 18)",
          (base18 - 1.189) / float(base18 * slope_err18), 5.85, 0.05)

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
    claim("n=16 mean-of-best (published)", float(mean[-1]), 0.126, 5e-4)
    claim("n=16 shortfall, own error (log units)",
          (predicted - y[-1]) / sy[-1], 4.4, 0.05)
    b16 = np.log(best[16])
    claim("all four n=16 instances below the extrapolation",
          float((b16 < predicted).all()), 1, 0)
    claim("most-favourable n=16 instance, instance sigmas on ln delta",
          float((predicted - b16.max()) / np.std(b16, ddof=1)), 1.3, 0.05)
    # The all-18 reading of the same point: the n <= 14 extrapolation is
    # untouched (the catch-up adds only n = 16 instances), the instance
    # mean falls further below it, and the widened spread puts exactly
    # one instance above it.
    claim("n=16 mean-of-best (all 18)", float(mean18[-1]), 0.124, 5e-4)
    claim("n=16 shortfall, own error (all 18)",
          (predicted - y18[-1]) / sy18[-1], 5.6, 0.05)
    b16_18 = np.log(best18[16])
    claim("n=16 instances above the extrapolation (all 18)",
          float((b16_18 >= predicted).sum()), 1, 0)
    claim("most-favourable n=16 instance, sigmas (all 18)",
          float((predicted - b16_18.max()) / np.std(b16_18, ddof=1)),
          -0.7, 0.05)

    steps = -np.diff(y)
    step_err = np.sqrt(sy[:-1] ** 2 + sy[1:] ** 2)
    claim("first log-step", steps[0], 0.296, 5e-4)
    claim("third log-step", steps[2], 0.487, 5e-4)
    claim("steepening significance",
          (steps[2] - steps[0]) / np.sqrt(step_err[0] ** 2 + step_err[2] ** 2),
          3.0, 0.05)
    claim("local base, first interval", float(np.exp(steps[0] / 2)), 1.16, 5e-3)
    claim("local base, last interval", float(np.exp(steps[3] / 2)), 1.31, 5e-3)
    steps18 = -np.diff(y18)
    claim("local base, last interval (all 18)",
          float(np.exp(steps18[3] / 2)), 1.32, 5e-3)


def section_facts():
    print("\nSec. VI A and Fig. 4, stalled fractions and pair overlaps")
    stalled, stalled16_all, band, pairs = [], [], 0, 0
    band16_all, pairs16_all = 0, 0
    for n in (8, 10, 12, 14, 16):
        fractions = []
        for name in sorted(glob.glob(str(RESULTS / f"pq/pq_n{n}_i*_sigma0.1.npz"))):
            data = np.load(name)
            instance = int(re.search(r"_i(\d+)_", name).group(1))
            weights, overlaps = data["peak_weights"], data["overlap_matrix"]
            fraction = float((weights < 0.9 * weights.max()).mean())
            keep = np.flatnonzero(weights >= 0.8 * weights.max())
            floor = np.outer(weights[keep], weights[keep])
            hat = (overlaps[np.ix_(keep, keep)] - floor) / (1 - floor)
            upper = hat[np.triu_indices(len(keep), 1)]
            if n == 16:
                stalled16_all.append(fraction)
                band16_all += int(((upper >= 0.25) & (upper <= 0.75)).sum())
                pairs16_all += len(upper)
                if instance > 3:  # published reading rests on instances 0-3
                    continue
            fractions.append(fraction)
            if n == 8:
                band += int(((upper >= 0.25) & (upper <= 0.75)).sum())
                pairs += len(upper)
        stalled.append(float(np.mean(fractions)))
    for n, value, quoted in zip((8, 10, 12, 14, 16), stalled,
                                (0.005, 0.13, 0.44, 0.62, 0.77)):
        claim(f"stalled fraction at n={n}", value, quoted, 5e-3)
    claim("stalled fraction at n=16 (all 18)",
          float(np.mean(stalled16_all)), 0.65, 5e-3)
    claim("intermediate-band pairs at n=16 (all 18)", band16_all, 0, 0)
    claim("pair count at n=16, all 18 (x 1e5)", pairs16_all / 1e5, 2.5, 0.05)
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
        slopes, slopes_all = [], []
        for f in sorted(glob.glob(str(RESULTS / f"pq/pq_n{n}_i*_sigma0.1.npz"))):
            value = slope_vs_logb(np.load(f)["peak_weights"],
                                  (25, 50, 100, 200))
            slopes_all.append(value)
            if n == 16 and int(re.search(r"_i(\d+)_", f).group(1)) > 3:
                continue
            slopes.append(value)
        claim(f"median slope over the null at n={n}",
              float(np.median(slopes)) * 2**n, quoted, 1.0)
        if n == 16:
            claim("median slope over the null at n=16 (all 18)",
                  float(np.median(slopes_all)) * 2**n, 111.0, 1.0)


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
        spreads, spreads_all = [], []
        for name in sorted(glob.glob(str(RESULTS / f"pq/pq_n{n}_i*_sigma0.1.npz"))):
            values = np.sort(np.load(name)["peak_weights"])[::-1]
            spread = (values[0] - values[9]) / values[0]
            spreads_all.append(spread)
            if n == 16 and int(re.search(r"_i(\d+)_", name).group(1)) > 3:
                continue
            spreads.append(spread)
        claim(f"relative top-10 spread at n={n}", float(np.median(spreads)),
              quoted, max(0.1 * quoted, 1e-8))
        if n == 8:
            claim("atomic instances at n=8 (spread < 1e-5)",
                  int((np.array(spreads) < 1e-5).sum()), 13, 0)
        if n == 16:
            claim("relative top-10 spread at n=16 (all 18)",
                  float(np.median(spreads_all)), 3.7e-2, 3.7e-3)

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

    for tag, published, quoted in (
        ("", True, {"delta": 0.139, "own": 2.5, "prop_raw": 3.4,
                    "prop_cor": 2.0}),
        (" (all 18)", False, {"delta": 0.138, "own": 3.4, "prop_raw": 4.1,
                              "prop_cor": 2.4}),
    ):
        sizes, mean, stderr, _ = mean_of_best(published_n16=published)
        y, sy = np.log(mean), stderr / mean
        # The corrected-steepening claims that once sat here left the
        # manuscript with the second verdict of REGISTRATION-CONVERGED.md:
        # log-linearity of the deficit is rejected (section_converged), so
        # only the level corrections below remain quoted.
        corrected = np.log(mean * (1 + np.array([deficits[int(n)]
                                                 for n in sizes])))

        head = sizes <= 14
        inter, sl, _, chi_head, det, s, sx = weighted_line(
            sizes[head], y[head], sy[head])
        predicted = inter + sl * 16.0
        shortfall = predicted - corrected[-1]
        claim(f"n=16 corrected delta{tag}", float(np.exp(corrected[-1])),
              quoted["delta"], 5e-4)
        claim(f"n=16 corrected shortfall, own error{tag}",
              shortfall / sy[-1], quoted["own"], 0.05)
        # Propagate the extrapolation's own variance at x = 16 through the
        # weighted-line covariance (Sxx recovered from det, s, sx).
        sxx = (det + sx * sx) / s
        var_pred = (sxx - 2 * 16.0 * sx + 16.0 * 16.0 * s) / det
        total = float(np.sqrt(sy[-1] ** 2 + var_pred))
        claim(f"n=16 shortfall, propagated{tag}",
              (predicted - y[-1]) / total, quoted["prop_raw"], 0.05)
        claim(f"n=16 corrected shortfall, propagated{tag}",
              shortfall / total, quoted["prop_cor"], 0.05)


def section_converged():
    print("\nAbstract, Secs. I, IV, VI A and App. B, the converged campaign")
    directory = RESULTS / "converged"
    if not directory.exists():
        print("  (no converged archives)")
        return
    from convergence_diagnostics import deficit_linearity, fill_ladder

    sizes = (8, 10, 12, 14)
    conv_mean, conv_err, frozen_mean = [], [], []
    lnd_mean, lnd_err, last_gain = [], [], []
    for n in sizes:
        conv, lnd, ladders = [], [], []
        for name in sorted(glob.glob(str(directory
                                         / f"pq_n{n}_i*_sigma0.1.npz"))):
            blob = np.load(name)
            best = expected_max(np.asarray(blob["peak_weights"],
                                           dtype=float), 16)
            conv.append(best)
            lnd.append(np.log(best / expected_max(
                np.asarray(blob["legacy_weight"], dtype=float), 16)))
            ladders.append(fill_ladder(
                np.asarray(blob["ladder_weights"], dtype=float)))
        frozen = [
            expected_max(np.asarray(np.load(path)["peak_weights"],
                                    dtype=float), 16)
            for path in sorted(glob.glob(str(RESULTS
                                             / f"pq/pq_n{n}_i*_sigma0.1.npz")))
        ]
        conv, lnd = np.array(conv), np.array(lnd)
        conv_mean.append(conv.mean())
        conv_err.append(conv.std(ddof=1) / np.sqrt(len(conv)))
        frozen_mean.append(np.mean(frozen))
        lnd_mean.append(lnd.mean())
        lnd_err.append(lnd.std(ddof=1) / np.sqrt(len(lnd)))
        rungs = np.array([
            [expected_max(ladder[:, k], 16) for k in (-2, -1)]
            for ladder in ladders
        ])
        last_gain.append(rungs[:, 1].mean() / rungs[:, 0].mean() - 1.0)

    y = np.log(np.array(conv_mean))
    sy = np.array(conv_err) / np.array(conv_mean)
    _, slope, _, chi_conv, _, _, _ = weighted_line(
        np.array(sizes, dtype=float), y, sy)
    claim("converged fixed-base chi2 (registered statistic)", chi_conv,
          7.4, 0.05)
    claim("converged fixed-base p", float(stats.chi2.sf(chi_conv, 2)),
          0.0245, 5e-4)
    claim("converged base per qubit", float(np.exp(-slope)), 1.200, 5e-4)
    for label, quoted_base in (("8->10", 1.16), ("10->12", 1.20),
                               ("12->14", 1.27)):
        i = ("8->10", "10->12", "12->14").index(label)
        claim(f"converged local base {label}",
              float(np.exp((y[i] - y[i + 1]) / 2)), quoted_base, 5e-3)

    t_conv = float((y[2] - y[3]) - (y[0] - y[1]))
    t_err = float(np.sqrt(np.sum(sy**2)))
    y_frozen = np.log(np.array(frozen_mean))
    t_frozen = float((y_frozen[2] - y_frozen[3]) - (y_frozen[0] - y_frozen[1]))
    claim("steepening at convergence (registered statistic)", t_conv,
          0.171, 5e-4)
    claim("steepening error at convergence", t_err, 0.064, 5e-4)
    claim("steepening at the frozen budget (registered statistic)",
          t_frozen, 0.190, 5e-4)
    claim("truncation share of the steepening", 1.0 - t_conv / t_frozen,
          0.10, 5e-3)
    claim("steepening rise at convergence, sigma", t_conv / t_err, 2.7, 0.05)

    # The second verdict is registered on the five sizes (3 dof); the n = 16
    # deficits are appended to the n <= 14 series before the test.
    lnd16 = []
    for name in sorted(glob.glob(str(directory / "pq_n16_i*_sigma0.1.npz"))):
        blob = np.load(name)
        lnd16.append(np.log(
            expected_max(np.asarray(blob["peak_weights"], dtype=float), 16)
            / expected_max(np.asarray(blob["legacy_weight"], dtype=float),
                           16)))
    if lnd16:
        lnd16 = np.array(lnd16)
        chi_lin, _, p_lin = deficit_linearity(
            (8, 10, 12, 14, 16),
            lnd_mean + [float(lnd16.mean())],
            lnd_err + [float(lnd16.std(ddof=1) / np.sqrt(len(lnd16)))])
        claim("deficit log-linearity chi2 (second verdict, 5 sizes)",
              chi_lin, 16.0, 0.05)
        claim("deficit log-linearity p (5 sizes)", p_lin, 0.0011, 5e-5)

    claim("acceptance: largest last-doubling gain n<=14, %",
          100.0 * max(last_gain), 0.107, 5e-3)
    above = 100.0 * (np.array(conv_mean) / np.array(frozen_mean) - 1.0)
    claim("converged level above frozen, min %", float(above.min()),
          0.1, 0.02)
    claim("converged level above frozen, max %", float(above.max()),
          7.0, 0.1)

    # The converged n = 16 row. Per REGISTRATION-CONVERGED.md it enters the
    # reported table and the full-grid statistic but not the registered
    # n <= 14 test; its own numbers are checked here.
    conv16, ladders16 = [], []
    for name in sorted(glob.glob(str(directory / "pq_n16_i*_sigma0.1.npz"))):
        blob = np.load(name)
        conv16.append(expected_max(np.asarray(blob["peak_weights"],
                                              dtype=float), 16))
        ladders16.append(fill_ladder(
            np.asarray(blob["ladder_weights"], dtype=float)))
    if conv16:
        conv16 = np.array(conv16)
        mean16 = float(conv16.mean())
        err16 = float(conv16.std(ddof=1) / np.sqrt(len(conv16)))
        claim("converged R16 at n=16", mean16, 0.1312, 5e-5)
        claim("converged R16 SE at n=16", err16, 0.0061, 5e-5)
        claim("converged local base 14->16",
              float(np.exp((y[3] - np.log(mean16)) / 2)), 1.295, 5e-3)
        t_full = float((y[3] - np.log(mean16)) - (y[0] - y[1]))
        t_full_err = float(np.hypot(np.hypot(sy[0], sy[1]),
                                    np.hypot(sy[3], err16 / mean16)))
        claim("full-grid steepening at convergence", t_full, 0.2152, 5e-4)
        claim("full-grid steepening error at convergence", t_full_err,
              0.0706, 5e-4)
        intercept_c, slope_c = weighted_line(
            np.array(sizes, dtype=float), y, sy)[:2]
        extrapolated = float(np.exp(intercept_c + slope_c * 16))
        claim("converged n=16 vs n<=14 extrapolation, %",
              100.0 * (mean16 / extrapolated - 1.0), -19.8, 0.1)
        claim("converged n=16 shortfall, sigma",
              float((np.log(extrapolated) - np.log(mean16))
                    / (err16 / mean16)), 4.7, 0.05)
        rungs16 = np.array([
            [expected_max(ladder[:, k], 16) for k in (-2, -1)]
            for ladder in ladders16
        ])
        claim("acceptance: last-doubling gain at n=16, %",
              100.0 * (rungs16[:, 1].mean() / rungs16[:, 0].mean() - 1.0),
              0.362, 5e-3)


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


def section_probe_ladder():
    print("\nSec. III D, m2 ladder and solutions (m2 scan logs)")
    text = {
        6: ((1.123, 1.118, 1.098, 1.040, 1.040),
            (1.00, 0.96, 0.82, 0.54, 0.59), 1.021, 0.51),
        8: ((1.168, 1.164, 1.153, 1.080, 1.062),
            (1.00, 0.95, 0.84, 0.66, 0.53), 1.036, 0.23),
    }
    order = ("sigma=0", "sigma=0.1", "sigma=0.2", "sigma=0.5", "sigma=1")
    for n, (m2_text, purity_text, sol_m2, sol_purity) in text.items():
        log = ROOT / f"analysis/moment_probe_n{n}_m2.log"
        if not log.exists():
            print(f"  (no m2 scan log for n={n})")
            continue
        ladder, solutions = {}, []
        for line in log.read_text().splitlines():
            f = line.split()
            if len(f) >= 3 and f[0] in order:
                ladder[f[0]] = (float(f[1]), float(f[2]))
            elif len(f) >= 3 and f[0].startswith("solution"):
                solutions.append((float(f[1]), float(f[2])))
        for key, m2_q, pur_q in zip(order, m2_text, purity_text):
            claim(f"m2 at n={n}, {key}", ladder[key][1], m2_q, 5e-4)
            claim(f"purity at n={n}, {key}", ladder[key][0], pur_q, 5e-3)
        claim(f"m2 at the solutions, n={n}",
              float(np.mean([m2 for _, m2 in solutions])), sol_m2, 5e-4)
        claim(f"purity at the solutions, n={n}",
              float(np.mean([p for p, _ in solutions])), sol_purity, 5e-3)
    if (ROOT / "analysis/moment_probe_n8_m2.log").exists():
        m2_init = ladder["sigma=0.1"][1]
        claim("excess at the initialization, per cent",
              100 * (m2_init - 1), 16.4, 0.05)
        claim("excess at the product probe, per cent",
              100 * (ladder["sigma=0"][1] - 1), 16.8, 0.05)
        claim("excess at the solutions, per cent",
              100 * (float(np.mean([m for _, m in solutions])) - 1),
              3.6, 0.05)
        haar_floor = 256.0 / 257.0
        claim("value-variance enhancement at n=8",
              (2 * m2_init - 1) / (2 * haar_floor - 1), 1.34, 5e-3)
    gm_log = ROOT / "analysis/geometric_measure.log"
    if gm_log.exists():
        for line in gm_log.read_text().splitlines():
            if line.strip().startswith("n = 4, instance 0"):
                gap = float(line.split("gap")[1])
                claim("n=4 closed-form gap", gap, 6e-8, 1.5e-9)


def section_pacing():
    print("\nSec. VI D, optimizer pacing (paired traces)")
    log = ROOT / "analysis/optimizer_pacing.log"
    if not log.exists():
        print("  (no pacing log)")
        return
    rows, ratio = {}, None
    for line in log.read_text().splitlines():
        f = line.split()
        if len(f) == 5 and f[0] in ("adam", "sgd"):
            rows[f[0]] = tuple(float(x) for x in f[1:])
        elif line.startswith("paired ratios"):
            ratio = float(line.split("|dtheta| adam/sgd =")[1].split(",")[0])
    if len(rows) < 2:
        print("  (pacing log incomplete)")
        return
    claim("median |grad|, adam", rows["adam"][1], 3.9e-2, 5e-4)
    claim("median |grad|, sgd", rows["sgd"][1], 4.0e-2, 5e-4)
    claim("median |dtheta|, adam", rows["adam"][2], 6.9e-2, 5e-4)
    claim("median |dtheta|, sgd", rows["sgd"][2], 1.5e-3, 5e-5)
    claim("pacing ratio |dtheta| adam/sgd", ratio, 47, 0.5)
    claim("total path, adam", rows["adam"][3], 38, 0.5)
    claim("total path, sgd", rows["sgd"][3], 0.7, 0.06)


def section_optclass():
    print("\nSec. VI D + Table 2, the optimizer-class block")
    directory = RESULTS / "optclass"
    if not directory.exists():
        print("  (no optclass archives)")
        return
    # The denominator quoted in the Table 2 caption: the (Adam, normal)
    # converged cell at B0 = 16, instances 0-2, mean and SE over instances.
    quoted = {8: (0.7205, 0.0176), 10: (0.4806, 0.0352),
              12: (0.3679, 0.0193), 14: (0.2473, 0.0125),
              16: (0.1388, 0.0077)}
    for n, (mean_quoted, err_quoted) in quoted.items():
        cells = [RESULTS / f"converged/pq_n{n}_i{i}_sigma0.1.npz"
                 for i in (0, 1, 2)]
        if not all(path.exists() for path in cells):
            print(f"  (converged denominator incomplete at n={n})")
            continue
        reach = np.array([expected_max(np.load(path)["peak_weights"], 16)
                          for path in cells])
        claim(f"R_conv at B0=16, n={n}", float(reach.mean()),
              mean_quoted, 5e-5)
        claim(f"R_conv SE over instances, n={n}",
              float(reach.std(ddof=1) / np.sqrt(len(reach))),
              err_quoted, 5e-5)
    # The exploratory matched-budget SGD control quoted in Sec. VI D.
    sgd_path = RESULTS / "sgd_converged/pq_n10_i0_sigma0.1.npz"
    frozen_sgd = RESULTS / "robustness/sgd/pq_n10_i0_sigma0.1.npz"
    conv_cell = RESULTS / "converged/pq_n10_i0_sigma0.1.npz"
    if sgd_path.exists() and frozen_sgd.exists() and conv_cell.exists():
        reach_sgd = expected_max(np.load(sgd_path)["peak_weights"], 16)
        reach_adam = expected_max(np.load(conv_cell)["peak_weights"], 16)
        reach_frozen = expected_max(np.load(frozen_sgd)["peak_weights"], 16)
        claim("matched-budget SGD at B0=16, n=10 i0", reach_sgd,
              0.200, 5e-4)
        claim("frozen SGD at B0=16, n=10 i0", reach_frozen, 0.084, 5e-4)
        claim("matched Adam cell at B0=16, n=10 i0", reach_adam,
              0.4514, 5e-5)
        claim("matched-budget SGD over matched Adam",
              reach_sgd / reach_adam, 0.44, 5e-3)
    else:
        print("  (matched-budget SGD control incomplete)")
    # Table 2: rho and S per arm, through the same loaders and statistics
    # as analysis/optclass_reach.py, against the numbers the table prints.
    from optclass_reach import (ARMS, growth, load_arm, load_denominators,
                                rho_literal, rho_paired, wls_slope)
    quoted_rho = {
        "lbfgs_sigma": {8: (1.0000, 0.0000), 10: (1.0056, 0.0036),
                        12: (0.9988, 0.0058), 14: (1.0078, 0.0067),
                        16: (1.0389, 0.0156)},
        "lbfgs_haar": {8: (1.0000, 0.0000), 10: (0.9929, 0.0091),
                       12: (0.9928, 0.0065), 14: (0.9866, 0.0166),
                       16: (0.9257, 0.0507)},
        "adam_haar": {8: (1.0000, 0.0000), 10: (0.9805, 0.0125),
                      12: (0.9643, 0.0088), 14: (0.9822, 0.0154),
                      16: (0.9265, 0.0563)},
    }
    # The literal construction, mean(R_arm)/mean(R_conv): the second
    # registered reading. Table 2 carries both columns so that the
    # tie-break (paired triggers, literal does not) is inspectable.
    quoted_literal = {
        "lbfgs_sigma": {8: (1.0000, 0.0345), 10: (1.0056, 0.1043),
                        12: (0.9985, 0.0724), 14: (1.0082, 0.0756),
                        16: (1.0395, 0.0871)},
        "lbfgs_haar": {8: (1.0000, 0.0345), 10: (0.9938, 0.1079),
                       12: (0.9935, 0.0781), 14: (0.9882, 0.0820),
                       16: (0.9310, 0.1110)},
        "adam_haar": {8: (1.0000, 0.0345), 10: (0.9811, 0.1057),
                      12: (0.9652, 0.0781), 14: (0.9836, 0.0804),
                      16: (0.9319, 0.1142)},
    }
    # S over the completed grid: the registered window is (n_min, n_max),
    # now (8, 16). rho(8) = 1 exactly, so S collapses to ln rho(16); the
    # anchor-free WLS slope is the disclosure the manuscript quotes with it.
    quoted_growth = {"lbfgs_sigma": (0.0381, 0.0151),
                     "lbfgs_haar": (-0.0772, 0.0548),
                     "adam_haar": (-0.0764, 0.0608)}
    quoted_growth_literal = {"lbfgs_sigma": (0.0387, 0.0906),
                             "lbfgs_haar": (-0.0715, 0.1241),
                             "adam_haar": (-0.0706, 0.1273)}
    quoted_per_instance_16 = {"lbfgs_sigma": (1.0691, 1.0308, 1.0168)}
    quoted_slope_free = {"lbfgs_sigma": (0.0016, 0.0016)}
    try:
        denominator = load_denominators(set(next(iter(quoted_rho.values()))))
    except ValueError as refusal:
        print(f"  ({refusal})")
        return
    for tag, _, optimizer, init_mode in ARMS:
        cells = load_arm(tag, optimizer, init_mode)
        series = {}
        for n, (rho_quoted, se_quoted) in sorted(quoted_rho[tag].items()):
            if not all((n, i) in cells for i in (0, 1, 2)):
                print(f"  ({tag} incomplete at n={n})")
                continue
            series[n] = rho_paired(cells, denominator, n)
            claim(f"rho {tag} n={n}", series[n][0], rho_quoted, 5e-5)
            claim(f"rho SE {tag} n={n}", series[n][1], se_quoted, 5e-5)
            lit_quoted, lit_se_quoted = quoted_literal[tag][n]
            literal = rho_literal(cells, denominator, n)
            claim(f"rho literal {tag} n={n}", literal[0], lit_quoted, 5e-5)
            claim(f"rho literal SE {tag} n={n}", literal[1],
                  lit_se_quoted, 5e-5)
        s_value, s_error = growth(series)
        if s_value is not None:
            s_quoted, s_se_quoted = quoted_growth[tag]
            claim(f"S {tag}", s_value, s_quoted, 5e-5)
            claim(f"S SE {tag}", s_error, s_se_quoted, 5e-5)
        literal_series = {
            n: rho_literal(cells, denominator, n)
            for n in series
        }
        s_lit, s_lit_err = growth(literal_series)
        if s_lit is not None:
            lit_quoted, lit_se_quoted = quoted_growth_literal[tag]
            claim(f"S literal {tag}", s_lit, lit_quoted, 5e-5)
            claim(f"S literal SE {tag}", s_lit_err, lit_se_quoted, 5e-5)
        if tag in quoted_per_instance_16 and (16, 0) in cells:
            per = [float(expected_max(cells[(16, i)]["peak_weights"], 16)
                         / expected_max(denominator[(16, i)], 16))
                   for i in (0, 1, 2)]
            for i, value in enumerate(per):
                claim(f"rho per instance {tag} n=16 i={i}", value,
                      quoted_per_instance_16[tag][i], 5e-5)
        if tag in quoted_slope_free:
            slope, slope_err = wls_slope(
                {n: v for n, v in series.items() if n >= 10})
            if slope is not None:
                claim(f"anchor-free WLS slope {tag}", slope,
                      quoted_slope_free[tag][0], 5e-5)
                claim(f"anchor-free WLS slope SE {tag}", slope_err,
                      quoted_slope_free[tag][1], 5e-5)
        if tag == "lbfgs_sigma":
            from optclass_reach import ARCH_CONTROL
            if ARCH_CONTROL.exists():
                x86 = load_arm("lbfgs_sigma", "lbfgs", "normal",
                               root=ARCH_CONTROL)
                if all((12, i) in x86 for i in (0, 1, 2)):
                    rho_x86 = rho_paired(x86, denominator, 12)
                    rho_arm12 = rho_paired(cells, denominator, 12)
                    claim("arch control rho_x86(12)", rho_x86[0],
                          0.9988, 5e-5)
                    z = (abs(np.log(rho_x86[0]) - np.log(rho_arm12[0]))
                         / np.hypot(rho_x86[1] / rho_x86[0],
                                    rho_arm12[1] / rho_arm12[0]))
                    claim("arch control z", float(z), 0.00, 5e-3)
        if tag == "lbfgs_sigma" and all(
                (n, i) in cells for n in (8, 14, 16) for i in (0, 1, 2)):
            arm_reach = {
                n: float(np.mean([expected_max(
                    cells[(n, i)]["peak_weights"], 16) for i in (0, 1, 2)]))
                for n in (8, 14, 16)
            }
            claim("lbfgs_sigma reach at n=8", arm_reach[8], 0.72, 5e-3)
            claim("lbfgs_sigma reach at n=16", arm_reach[16], 0.14, 5e-3)
            claim("lbfgs_sigma local base 14->16",
                  float(np.exp((np.log(arm_reach[14])
                                - np.log(arm_reach[16])) / 2)), 1.31, 5e-3)
            adam = {n: float(np.mean([expected_max(denominator[(n, i)], 16)
                                      for i in (0, 1, 2)]))
                    for n in (14, 16)}
            claim("Adam local base 14->16, matched instances",
                  float(np.exp((np.log(adam[14])
                                - np.log(adam[16])) / 2)), 1.33, 5e-3)


def section_moments_solution():
    print("\nSec. III / App. B, the moment ladder at a converged solution "
          "point (committed log)")
    log = ROOT / "analysis/moment_probe_converged.log"
    if not log.exists():
        print("  (no converged-solution moment log)")
        return
    rows = []
    certified = None
    for line in log.read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith("converged") and len(fields) == 9:
            rows.append(tuple(float(x) for x in fields[1:7]))
        if "certified truncation" in line and certified is None:
            certified = float(line.split("|m4| <=")[1])
    if not rows:
        print("  (log carries no solution rows)")
        return
    # The five rows are the five best restarts of ONE archive (n = 8,
    # instance 0), not five instances: their agreement to 1e-4 documents
    # uniqueness modulo gauge on this instance at this size, nothing wider.
    first = np.array(rows[0])
    spread = float(np.max(np.abs(np.array(rows) - first)))
    claim("solution-row spread across the five best restarts", spread,
          0.0, 1e-4)
    claim("m2 at the converged solution point", first[1], 1.036, 5e-4)
    claim("m3 at the converged solution point", first[2], 1.110, 5e-4)
    claim("m4 at the converged solution point", first[3], 1.228, 5e-4)
    claim("m4 truncation certificate", certified, 0.054, 5e-4)
    claim("r3 at the converged solution point", first[4], 0.9985, 5e-5)
    claim("r4 at the converged solution point", first[5], 0.994, 5e-4)


#: Coverage floor. A verify run whose corpus is quietly short prints
#: "0 disagreement(s)" over whatever it did check, which reads as a pass;
#: that exact incident happened once. The inventory below fails loudly
#: instead, and the claim count is held to a floor at the end.
EXPECTED_ARCHIVES = (
    ("pq/pq_n8_i*_sigma0.1.npz", 18),
    ("pq/pq_n10_i*_sigma0.1.npz", 18),
    ("pq/pq_n12_i*_sigma0.1.npz", 18),
    ("pq/pq_n14_i*_sigma0.1.npz", 18),
    ("pq/pq_n16_i*_sigma0.1.npz", 18),
    ("converged/pq_n*_sigma0.1.npz", 90),
    ("optclass/lbfgs_sigma/pq_n*_sigma0.1.npz", 15),
    ("optclass/lbfgs_haar/pq_n*_sigma0.1.npz", 15),
    ("optclass/adam_haar/pq_n*_sigma0.1.npz", 15),
    ("optclass_arch/lbfgs_sigma/pq_n12_i*_sigma0.1.npz", 3),
    ("converged_pilot/pq_n*_i0_sigma0.1.npz", 5),
    ("budget800/pq_n*_sigma0.1.npz", 6),
    ("step_budget/pq_n*_sigma0.1.npz", 13),
)
CLAIM_FLOOR = 280


def section_inventory():
    print("\nCorpus inventory (a short corpus must fail, not shrink the log)")
    for pattern, expected in EXPECTED_ARCHIVES:
        count = len(glob.glob(str(RESULTS / pattern)))
        status = "ok " if count == expected else "FAIL"
        print(f"  [{status}] {pattern:<48} {count:>3} of {expected}")
        if count != expected:
            FAILURES.append(f"inventory {pattern}: {count} archives, "
                            f"expected {expected}")


def section_trajectories():
    print("\nSec. VI C, trajectory commitment factors (trajectories.npz)")
    path = ROOT / "analysis/trajectories.npz"
    if not path.exists():
        print("  (no trajectory archive)")
        return
    data = np.load(path)
    medians = {}
    for n, restarts in ((8, 12), (12, 8)):
        r90, r50 = [], []
        for restart in range(restarts):
            deltas = data[f"n{n}_r{restart}_deltas"]
            q_final = data[f"n{n}_r{restart}_qfinal"]
            final = deltas[-1]
            committed = np.flatnonzero(q_final > 0.5)
            above90 = np.flatnonzero(deltas >= 0.9 * final)
            above50 = np.flatnonzero(deltas >= 0.5 * final)
            if not (len(committed) and len(above90) and len(above50)):
                continue
            if above90[0] == 0 or above50[0] == 0:
                continue
            r90.append(committed[0] / above90[0])
            r50.append(committed[0] / above50[0])
        medians[n] = (float(np.median(r90)), float(np.median(r50)))
    claim("commitment / t90 factor at n=12", medians[12][0], 1.9, 0.05)
    claim("commitment / t50 factor at n=12", medians[12][1], 8.4, 0.05)
    claim("commitment / t90 factor at n=8", medians[8][0], 0.86, 0.005)


def section_deep_anchors():
    print("\nSec. VI A, deep-anchor excess ratios (depth_ceiling/)")
    ratios = {}
    for n in (10, 12):
        per_instance = []
        for i in (0, 1, 2):
            deep = np.load(RESULTS / f"depth_ceiling/tau{4 * n}"
                           / f"pq_n{n}_i{i}_sigma0.1.npz")["peak_weights"]
            shallow = np.load(RESULTS / "pq"
                              / f"pq_n{n}_i{i}_sigma0.1.npz")["peak_weights"]
            per_instance.append(float(shallow.max() / deep.max()))
        ratios[n] = float(np.mean(per_instance))
    claim("deep-anchor excess ratio, mean at n=10", ratios[10], 1.8, 0.05)
    claim("deep-anchor excess ratio, mean at n=12", ratios[12], 3.1, 0.05)


def section_probe_draws():
    print("\nSec. III C, independent probe draws at n=6 (committed logs)")
    draws = {}
    text = (ROOT / "analysis/moment_probe_n6.log").read_text()
    match = re.search(r"sigma=0\.1\s+[\d.]+\s+([\d.]+)", text)
    draws["probe"] = float(match.group(1))
    for line in (ROOT / "analysis/third_moment.log").read_text().splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == "6":
            draws["scan"] = float(fields[1])
            scan_m3 = float(fields[2])
            break
    match = re.search(r"n = 6: .*m2 check: ([\d.]+)",
                      (ROOT / "analysis/envelope_exact.log").read_text())
    draws["envelope"] = float(match.group(1))
    values = np.array(sorted(draws.values()))
    claim("m2 range over the three draws, per cent",
          100 * (values[-1] - values[0]) / values.mean(), 1.5, 0.05)
    r3_scan = scan_m3 / draws["scan"] ** 3
    match = re.search(r"sigma=0\.1\s+[\d.]+\s+[\d.]+\s+([\d.]+)",
                      (ROOT / "analysis/moment_probe_n6.log").read_text())
    r3_probe = float(match.group(1)) / draws["probe"] ** 3
    claim("r3 difference between the two draws, per cent",
          100 * abs(r3_scan - r3_probe), 0.13, 0.02)


def section_kernel_validation():
    print("\nSec. III B, kernel validation tolerances (committed logs)")
    ratios = []
    for line in (ROOT / "analysis/kernel_exact.log").read_text().splitlines():
        match = re.search(r"exact = ([+-][\d.]+)\s+MC = ([+-][\d.]+) "
                          r"\+/- ([\d.]+)", line)
        if match:
            exact, mc, se = map(float, match.groups())
            if se > 0:
                ratios.append(abs(exact - mc) / se)
    claim("tightest kernel point, standard-error ratio",
          max(ratios), 2.2, 0.01)
    log = ROOT / "analysis/check_kernel_exact_residuals.log"
    if log.exists() and "residual vs exact kernel" in log.read_text():
        match = re.search(r"residual vs exact kernel : ([+-][\d.]+) \+/- "
                          r"([\d.]+)", log.read_text())
        claim("mean residual vs the exact kernel",
              float(match.group(1)), -0.004, 5e-4)
        claim("bootstrap error of that residual",
              float(match.group(2)), 0.0035, 5e-4)
    else:
        print("  (residuals log not yet committed)")


def section_figure_certificates():
    print("\nFig. 2 caption, truncation certificates (committed logs)")
    norm6 = 24.0 / 64.0**4
    for line in (ROOT / "analysis/fourth_moment.log").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[0] in ("2", "6"):
            m4, dropped = float(fields[3]), float(fields[5].split("[")[0])
            if fields[0] == "6":
                claim("certificate at (6, 6), per cent of m4",
                      100 * dropped / norm6 / m4, 0.33, 0.005)
            else:
                claim("certificate at (6, 2), per cent of m4",
                      100 * dropped / norm6 / m4, 15.9, 0.1)
    text = (ROOT / "analysis/fourth_moment_n8.log").read_text()
    for depth, quoted, tol in (("8", 0.98, 0.01), ("2", 125, 1)):
        match = re.search(
            rf"^\s*{depth}\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s.*"
            rf"\n\s*certified \|m4 error\| <= ([\d.e+-]+)",
            text, re.M)
        m4, certificate = float(match.group(1)), float(match.group(2))
        claim(f"certificate at (8, {depth}), per cent of m4",
              100 * certificate / m4, quoted, tol)


def section_cap_fractions():
    print("\nApp. B, restarts at the extended cap (step_budget/)")
    for n, quoted in ((12, 9), (14, 17), (16, 3)):
        at_cap = 0
        for path in sorted((RESULTS / "step_budget")
                           .glob(f"pq_n{n}_i*_sigma0.1.npz")):
            run = np.load(path)
            at_cap += int((run["num_steps"] >= int(run["max_steps"])).sum())
        claim(f"restarts at the 1600 cap, n={n}", at_cap, quoted, 0)


def main() -> None:
    print("Recomputing the manuscript's numbers from results/.")
    section_reach()
    section_facts()
    section_budget()
    section_step_budget()
    section_atom()
    section_corrugation()
    section_truncation()
    section_converged()
    section_moments_solution()
    section_moments_probe()
    section_probe_ladder()
    section_pacing()
    section_optclass()
    section_trajectories()
    section_deep_anchors()
    section_probe_draws()
    section_kernel_validation()
    section_figure_certificates()
    section_cap_fractions()
    section_inventory()

    if CHECKED < CLAIM_FLOOR:
        FAILURES.append(f"coverage: {CHECKED} claims recomputed, the floor "
                        f"is {CLAIM_FLOOR}; a section went missing")

    print(f"\n{CHECKED} claims recomputed, {len(FAILURES)} disagreement(s).")
    print("\nNot recomputable here, checked by their own scripts and logs:")
    for item in (
        "the exact kernel and its Monte-Carlo validation (kernel_exact.log)",
        "the 240-point residuals against Eq. (5) "
        "(check_kernel_exact_residuals.py)",
        "the moment hierarchy and its probe dependence (moment_probe_scan.py)",
        "the SU(4) surjectivity hypothesis (gate_surjectivity.log)",
        "the Hessian flat band at the Adam stop (hessian_solutions.log) "
        "and the derived gauge count 3S, its saturation and its "
        "constructive families (gauge_dimension.log)",
        "the probe-ray gauge count 3S + 9B + 4E, the polished flat band, "
        "the ray-speed separation and the R_z rank (probe_gauge.log)",
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
