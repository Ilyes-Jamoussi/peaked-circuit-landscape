# Registration: the reach law at convergence

This file is committed **before** the runs it governs exist. Its purpose is to
fix the protocol, the statistic, the test and the verdict rule for a
re-measurement of the reach law with the optimizer's step cap removed, so that
the outcome cannot be chosen after the data are seen. The git history of this
repository is the timestamp.

One measurement precedes this file, because its only product is the schedule
fixed below: a scale pilot that settles the five constants of the converged
protocol. It contributes no reach value, no fit and no verdict, and no number
in this file is computed from it. Every other block starts after the commit
that adds this file.

Its provenance, stated exactly because a registration that overstates what has
already run is worth nothing. The pilot ran on the author's Apple-silicon
machine, not on the campaign's x86 instances: instance 0 at n = 10, 12 and 14,
four restarts each, rungs to 6400 with the stopping rule disabled
(`--converged --ladder 400,800,1600,3200,6400 --max-steps 6400 --rel-tol 0`),
plus one repeat at n = 10 with the learning-rate floor lowered from
0.05 x 2^-3 to 0.05 x 2^-6. Its archives are committed under
`results/converged_pilot/` with the machine fingerprint that says which
architecture produced them.

That architecture does not matter here and would matter for a published
number, which is why the distinction is drawn rather than glossed. Optimizer
trajectories are chaotic over thousands of steps and do not reproduce bit for
bit across architectures; what the pilot fixes is a schedule -- the order of
magnitude at which the gain per doubling collapses, and whether the floor's
value changes the answer -- and neither is architecture-sensitive. Every
number this registration governs is measured on x86.

## Why this measurement is being repeated

The reach values published so far are the best peak weight found within 400
Adam steps. Three measured facts make them unfit to carry a statement about
functional form.

1. **The cap binds, and it binds unequally.** The fraction of restarts stopping
   at the cap is 0.608 at n = 8 and 0.944, 0.978, 0.990, 0.983 at
   n = 10, 12, 14, 16; the number of instances whose *best* restart stops there
   is 3 of 18 at n = 8 and 18/18, 18/18, 18/18, 4/4 above it. The median step
   count is 400 at every size. At n >= 10 the quantity measured is what a
   truncated optimizer attains, not what the protocol attains.

2. **The published values are lower bounds whose shortfall grows with n.**
   Quadrupling the cap on the same restart seeds raises the best reachable
   peakedness by 0.1, 2.3, 3.1, 4.7 and 10.9 % at n = 8 to 16. A shortfall that
   grows with n subtracts more from the large sizes than from the small, which
   is the same signature as a decay that accelerates. Both the steepening
   statistic and the p = 0.009 rejection are computed from levels biased in the
   direction of the claim they support.

3. **The control that sizes the shortfall may not have converged either.** At
   the 1600-step cap, 9 of 48 restarts at n = 12, 17 of 48 at n = 14 and 3 of
   16 at n = 16 still end at the cap, and the median step count reaches
   1226-1489 at n = 14. On the face of it the deficits of fact 2 are then
   themselves lower bounds, and no statement of the form "truncation explains
   15 % of the steepening" could be closed with them.

   **The pilot settles this one against the objection at every size it tests,
   and it is recorded here before the campaign rather than after.** On
   instance 0, same four seeds, the 1600-step control captures essentially the
   whole deficit:

   | n | control (1600) | converged | control captures | last gain |
   |---|---|---|---|---|
   | 10 | 6.79 % | 6.80 % | 100 % | 0.001 % |
   | 12 | 6.56 % | 6.56 % | 100 % | 0.00 % |
   | 14 | 1.83 % | 1.83 % | 100 % | 0.00 % |
   | 16 | 10.85 % | 10.95 % | 99 % | 0.00 % |

   The reason the cap fraction misleads: the restarts still sitting at the
   1600-step cap are not the ones carrying the best of the batch. The weaker a
   restart, the more it gains from further budget -- the rank-gain correlation
   over the control ensembles is +0.225 -- so a batch best can be fully
   converged while a third of its restarts are not. The 9/48, 17/48 and 3/16
   counts above are therefore not evidence about the published deficits.

   Fact 3 is left standing, in the weakened form this table gives it, rather
   than deleted. It is measured on one instance per size out of eighteen, at
   four restarts out of thirty-two, so it bounds the objection without closing
   it; the campaign closes it on the full grid. The point of registering the
   result now is that it removes the campaign's freedom to be read as a rescue:
   the published correction is, on this evidence, already sound, and the
   campaign's job is to replace a corrected number by a measured one rather
   than to save a number in danger.

## The measurement

The converged protocol, applied by `pq_experiment.py --converged`:

| Setting | Value |
|---|---|
| `max_steps` | 12800 |
| `ladder` | 400, 800, 1600, 3200, 6400, 12800 |
| `min_learning_rate` | 0.00625 (= 0.05 x 2^-3) |
| `rel_tol` | 0.002 |
| `polish_steps` | 400 |
| `convergence_tol` | 0.0 |

These values are fixed by the scale pilot before this commit and do not move
again. They and `CONVERGED_PROTOCOL` in `pq_experiment.py` agree at this
commit; a disagreement between the two invalidates this registration.

What the pilot measured, and what each constant is answering:

- **The gain per doubling collapses, and it collapses sharply.** Best-of-4
  gain at each doubling, instance 0, stopping rule disabled:

  | n | 400 -> 800 | 800 -> 1600 | 1600 -> 3200 | 3200 -> 6400 |
  |---|---|---|---|---|
  | 10 | 3.24 % | 3.47 % | 0.003 % | 0.001 % |
  | 12 | 2.53 % | 3.94 % | 0.00 % | 0.00 % |
  | 14 | 1.85 % | 0.00 % | 0.00 % | 0.00 % |

  Every size tested is finished by rung 3200 and the geometric residual past it
  is below 1e-5. `rel_tol = 0.002` stops one rung after that collapse rather
  than at it. `max_steps = 12800` leaves two further rungs that these sizes
  will mostly not use; it is kept for n = 16, where the deficit is largest and
  which has no converged run behind it at this commit.
- **The floor is not a sensitive choice.** Floors of 0.05 x 2^-3 and
  0.05 x 2^-6, same seeds, same rungs, reach the same delta to within 1e-6
  (0.4632566 against 0.4632568 on the best restart) while their final
  parameters differ by 0.079. The floor prevents the freeze; its exact value,
  inside that range, does not select the answer.
- **The polish earns nothing at a converged point,** which is worth reporting
  rather than hiding: its median gain at n = 10 was 0.0000 %, i.e. the run
  already stops at a stationary point. It is kept at 400 steps, half the
  budget first considered, because it costs about 3 % of a restart and closes
  the objection that the returned parameters are not critical. Its measured
  gain is reported per size.

One caution the pilot also settles, against the expectation that motivated it:
at n = 10, instance 0, the 1600-step control already captures the whole
deficit. On matched seeds the frozen best-of-4 is 0.43375, the 1600-step
control 0.46319 and the converged run 0.46326 -- 6.79 % against 6.80 %. Fact 3
above is therefore a statement about n >= 12, where restarts still end at the
extended cap, and not about the grid as a whole. The per-instance spread is
what makes this easy to misread: at the same size and budget, instance 0 gains
6.8 % while instances 1 and 2 gain 0.02 % and 0.01 %.

`decay_every` stays at 300, so steps 1 to 400 of a converged restart are
bit-identical to the frozen restart from the same seed and the two grids nest.
The floor exists because without it the stepsize reaches 0.05 x 2^-21 by step
6400 and the optimizer freezes instead of converging, which is precisely what
stalls the existing 1600-step control.

The grid: **18 instances at each of n = 8, 10, 12, 14, 16**, including n = 16,
and **B = 32 restarts** per instance, written to `results/converged/`.

Both counts are set from the archives, not from taste. Subsampling the
published 200-restart ensembles:

- Keeping the **first 32 restarts** of each batch moves the full-grid
  steepening from 0.241 to 0.250 and the n <= 14 rejection from p = 0.0092 to
  p = 0.0138. A six-fold cut in restart budget leaves the verdict where it was:
  restarts are not the lever.
- Keeping **instances 0 to 11** (n <= 14, where the test lives; n = 16 keeps its
  four) moves the same p from 0.0092 to 0.176. Keeping instances 0 to 2, the
  original grid, gives p = 0.71. The rejection is carried by the eighteenth
  instance, not by the two-hundredth restart.

So the instance count is held at 18 and the saved restarts are spent on steps.
One converged instance costs at most 5.3 times a frozen one (32 x 13200 against
200 x 400 step-units), and less in practice, since `rel_tol` stops restarts
early.

## The analysis, fixed in advance

**(i) Statistic.** `R_conv(n)`: per instance, `expected_max(peak_weights, 16)`,
the exact expectation of the best of sixteen restarts drawn without replacement
from the thirty-two (`analysis/budget_scan.py`, committed and self-tested); per
size, the mean over the eighteen instances, with uncertainty the **standard
error over those instances**. That standard error, not a within-instance
bootstrap, is the error bar of record. The archive's own best-of-B, the
statistic the manuscript publishes, is tabulated beside it at every size so
that the change of estimator is visible rather than assumed away; it decides
nothing.

B0 = 16 rather than the raw best-of-B, for two reasons fixed here. It is the
largest budget at which the main grid (B = 32) and the optimizer-class arms
(B = 16) compare without extrapolation, and it is a budget the published
200-restart grid also supports, so frozen and converged reach are compared at
equal restart budget rather than at equal file size. Re-reading the published
grid at B0 = 16 gives local bases 1.168, 1.222, 1.284, 1.313, a full-grid
steepening of 0.236 and p = 0.0107 on n <= 14, against 1.159, 1.212, 1.276,
1.308, 0.241 and p = 0.0092 at B = 200. The change of estimator is recorded
here, before the new data exist, and it does not move the published verdict.

**(ii) Test.** Weighted least squares of `ln R_conv` against `n` over
**n <= 14**, weights `1/sigma^2` with `sigma` the standard error on
`ln R_conv`, and weighted `chi^2` on **2 degrees of freedom** for the
fixed-base model: the same fit function and the same head of the grid as the
published p = 0.009. Alongside it,
`T = log-step(12 -> 14) - log-step(8 -> 10)`, where
`log-step(a -> b) = ln R_conv(a) - ln R_conv(b)`, with its standard error
propagated from the instance errors of the four sizes it involves. The n = 16
point enters the reported table and the full-grid statistic but not the
registered test, exactly as in the published analysis.

## The verdict

The branches are evaluated in the order (c), (b), (a); the first that applies
is the verdict.

**(c) T <= -2 sigma.** The steepening is declared, in those words, an
**artifact of the step budget**. The title and the abstract are rewritten
around what survives: the exact finite-depth covariance kernel with the closed
third and fourth moments, the exact depth-independence of the second-order
amplitude data against a measured reach that is not depth-independent, the
correction of the barren-plateau diagnosis, the geometry of the atom and
`delta*`, and connectivity at the trap scale with its corrugation floor. The
reach law is reported as a protocol-bound level with no claim on its functional
form, and registered prediction 2 is re-graded: falsified as stated at the
frozen budget, **not** superseded by a steepening law.

**(b) The fixed base is not rejected at p < 0.05 on the converged grid.** The
claim leaves the abstract and Sec. VI A and is replaced, word for word, by:

> At the frozen 400-step budget, no fixed base fits the reachable peakedness
> across the measured grid (p = 0.009); at convergence the same measurement
> does not exclude one, so the rejection is a statement about what the protocol
> reaches at a fixed budget and not about the landscape.

Conjecture C-reach drops its rate clause, its first sentence becoming:

> Restarts are inefficient substitutes for signal: the budget required to
> maintain a fixed fraction of the ceiling `delta*` grows with n, at a rate
> these data do not determine.

Registered prediction 2 loses its "superseded by a steepening law" clause,
which becomes "superseded by a level shortfall at the frozen budget".

**(a) The fixed base is rejected at p < 0.05 on n <= 14 and T >= +2 sigma.**
The steepening survives convergence. Both grids stand in the manuscript, the
frozen one labelled as the protocol-bound measurement it is and the converged
one as the landscape statement, and the abstract keeps its rejection with the
converged p replacing the frozen one. Nothing is withdrawn.

**The residual cell.** One case is left by the three branches: the fixed base
rejected at p < 0.05 with T strictly between -2 and +2 sigma. It is named now
because it is the outcome most likely to be argued over afterwards. The
rejection stands and is reported; the abstract's monotone-base clause is
replaced, word for word, by:

> the local base grows over the grid, by an amount that stays under two
> standard errors at convergence, so these data establish that the decay is not
> fixed-base without establishing that it accelerates.

## Second verdict: does the truncation deficit cancel?

Independent of the main verdict and settled on its own evidence.

The manuscript's defence of the steepening against truncation is that a deficit
whose logarithm is linear in n subtracts the same amount from every log-step
and so cancels exactly from T. The cancellation is an identity
(`analysis/step_budget_control.py`, self-test), but it is conditional on the
log-linearity, which the manuscript asserts and does not test. The measured
deficit factors, 1.001, 1.023, 1.031, 1.047, 1.109, give increments of `ln d`
of 0.0217, 0.0078, 0.0154 and 0.0575: a factor 7.4 between the smallest and the
largest, which is not a straight line.

Registered test: weighted least squares of `ln d(n)` against `n` over the five
sizes, weighted `chi^2` on **3 degrees of freedom**. `d(n)` is measured on the
converged campaign itself as the within-trajectory ratio
`expected_max(peak_weights, 16) / expected_max(legacy_weight, 16)` per
instance, so it depends on neither seed matching nor architecture, with the
standard error over the eighteen instances as its error bar. **If log-linearity
is rejected at p < 0.05, the cancellation sentence is removed from Sec. VI A
and Appendix B and the argument is not made in any other form**, whatever the
main verdict says.

## When the numbers count as converged

Acceptance criterion: at every size, the relative gain of the last budget
doubling, `R_conv` read at ladder rung 12800 against rung 6400 at B0 = 16, must
be **<= 0.5 %**. A restart that stopped early carries its last ladder value
forward to every later rung, which is what it would have reported had it
continued under its own stopping rule; the `NaN` padding of `ladder_weights` is
filled forward before the ladder is read, so no restart is dropped from a rung
it did not reach.

If the criterion fails at any size, the numbers are published as **lower
bounds** with a one-sided Richardson band upward: with `L_k` the ladder value at
rung k, `g = L_K - L_{K-1}` and `r = g / (L_{K-1} - L_{K-2})`, the band is
`[L_K, L_K + g r / (1 - r)]` for `0 < r < 1`, and undefined otherwise, in which
case the raw value and the last gain are reported alone. The statistic, the
test and the verdict are then computed on the raw values **and** on the upper
edge of the band, and both are reported, whichever way they differ.

## The n = 16 catch-up

The frozen protocol goes from four instances to eighteen at n = 16 (block
`frozen_n16_catchup`: instances 4 to 17, 200 restarts, identical settings, same
output directory). Every n = 16 number in the manuscript is then reported
twice: on instances 0 to 3, the set the published values rest on, and on all
eighteen. Both are reported whichever direction the change goes, and the
four-instance value is not dropped once the eighteen-instance one is known.

## The optimizer class

Registered prediction 3 was confirmed only for hyperparameter variation of the
frozen protocol; the one distinct optimizer run, plain SGD, left the
stable-local class by a criterion adopted after that run. Block `optclass`
closes the 2 x 2: optimizer in {Adam, L-BFGS-B} crossed with initialization in
{normal, Haar}, three instances (0, 1, 2) per size, B = 16 restarts. The
(Adam, normal) cell is the converged grid read at B0 = 16 on the same three
instances, so all four cells share instances, sizes and budget.

Statistic: the reach ratio at matched instances and matched budget,
`rho_arm(n) = R_arm(n) / R_conv(n)`, each side at B0 = 16 on instances 0 to 2,
with the standard error over those three instances. What each outcome costs:

- **`rho <= 1` within errors at every size.** The ceiling is the class's, not
  Adam's. Registered prediction 3 is recorded as confirmed across optimizer
  classes, which is more than it claimed, and the third falsifier of C-hardness
  stays untriggered.
- **`rho > 1` at 2 sigma at some size, flat in n.** The level is
  optimizer-dependent: every absolute reach value in the manuscript is
  relabelled as Adam's rather than the class's, and the quantitative comparison
  with the fitted base of Ref. [aaronson2024peaked] is withdrawn. The scaling
  claims survive.
- **`rho > 1` and growing with n** (its own log-step positive at 2 sigma). The
  third registered falsifier of C-hardness is triggered by our own measurement.
  C-hardness is **withdrawn, not weakened**, and "an optimizer class whose
  reach does not shrink" moves from a falsifier the paper proposes to one it
  reports.

## Provenance

- Runner: `python cloud/runner.py --only converged --workers 96 --jobs 3`.
  Blocks `frozen_n16_catchup`, `converged_pilot`, `lr_floor_sweep`,
  `converged`, `optclass`, all declared in `cloud/runner.py` at this commit.
  The first two are the pilot of the opening section; they are declared so
  that the schedule can be re-derived on x86 by anyone who wants to, and
  re-running them changes no number this file governs.
- Pinned code: this repository at the commit that adds this file, and the
  reproduction repository `peaked-circuits-pennylane` at `a9b890d`, with the
  package versions of `requirements.txt` (Python 3.13, pennylane 0.45.1,
  numpy 2.5.1, scipy 1.18.0).
- Machine: GCP `n2-highcpu-96` spot instances (Intel), zone
  `northamerica-northeast1-b`, project `project-b2dccd70-9346-4730-9ef`. x86
  throughout: the frozen and converged grids nest bit for bit only within one
  architecture, so no block moves to AMD.
- Each block writes `results/<block>/ENV_<block>.txt` before its first job -
  the two commit hashes, `uname -srm`, the CPU model, the interpreter and
  package versions, the machine type and the zone - and that file is committed
  with the block's log. A preempted spot VM leaves nothing behind, so a block
  that is not fingerprinted before it starts is not fingerprinted at all.
- Nothing launches before the commit that adds this file, the scale pilot and
  the floor sweep of the opening paragraph excepted.

## What failure costs, sentence by sentence

REGISTRATION.md notes that every registered prediction failing so far was read
as strengthening the conjecture, and asks its own failure to remove a sentence
from the abstract. It does not say which. This one does, because the reach law
is the paper's headline result and the temptation to keep it under softer
wording is proportionate.

| Branch | Removed from the abstract | Removed from the body |
|---|---|---|
| (a) | nothing; "(p = 0.009)" becomes the converged p and the clause "at the frozen budget" is added to the campaign description | nothing |
| (b) | "rejects a fixed-base decay law on the grid alone (p = 0.009)", replaced by the sentence above | Sec. I "no fixed base fits the reachable peakedness across the measured range, which leaves the n = 50 estimate without support"; Sec. VI A "At fixed protocol budget, no fixed base fits the reachable peakedness over the measured grid"; C-reach's rate clause |
| residual cell | "the local base grows monotonically, from 1.16 to 1.28 per qubit on the grid and to 1.31 including the four-instance n = 16 point", replaced by the sentence above | the "3.0-standard-error increase" reading of Sec. VI A |
| (c) | the whole campaign sentence, and the reach half of "Both scaling laws falsify predictions registered before the campaign, in the direction of greater hardness"; the title is rewritten | Sec. VI A's steepening claim, prediction 2's supersession clause, and the C-reach rate clause |
| second verdict, independently | nothing | Sec. VI A and Appendix B: "a deficit whose logarithm is close to linear in n" and the cancellation argument resting on it |

The protocol constants, the estimator, the fit head (n <= 14), the degrees of
freedom and the 2-sigma thresholds are frozen above. Every n = 16 number is
reported at both instance counts, and every number is reported at both ends of
the Richardson band whenever the convergence acceptance fails. No statistic is
selected after the data are seen.
