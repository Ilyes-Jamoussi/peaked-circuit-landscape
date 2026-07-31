# Registration: the resolution-and-instance sweep

This file is committed **before** the runs it governs exist. Its purpose is to fix
the statistic, the test and the verdict rule for the corrugation measurement, so
that the outcome cannot be chosen after the data are seen. The git history of this
repository is the timestamp.

## Why this measurement is being repeated

The corrugation result reported so far rests on four sizes with one instance each,
at two different string resolutions (32 segments at n = 8, 10, 12 and 64 at n = 16),
and n = 14 was never run. Three facts make that basis too narrow to carry a
statement about functional form.

1. **Refinement lowers rho.** Re-running the twelve n = 8 pairs at 32 segments
   instead of 16 moved the median from 0.846 to 0.714, a 15.6 % downward correction,
   and moved the adjacency diagnostic from 0.64 to 0.85.

2. **A correction of that size reverses the verdict.** The two least-resolved sizes
   are n = 10 and n = 12 (adjacency 0.74 and 0.65, against 0.85 at n = 8). Applying
   a 15.6 % downward correction to rho(10) and rho(12) moves the fitted base from
   0.8709 to 0.8752 — a *slower* decay — and flips the fit comparison: the best
   fixed power law overtakes the exponential once the correction exceeds 14.4 %.
   The interval exponents go from (0.98, 1.02, 2.43) to (1.74, 1.02, 1.84), so a
   single fixed exponent is no longer excluded.

3. **The error bars omit the dominant variance.** They resample twelve pairs drawn
   from one instance, so they carry no instance-to-instance component. Against
   those same bars the exponential is itself rejected (chi^2 = 11.4 on 2 dof,
   p = 0.0034), which is the signature of underestimated uncertainty rather than of
   a good fit.

## The measurement

One matched resolution, **64 segments**, at **n = 8, 10, 12, 14, 16**, with
**three instances per size** and twelve disjoint near-optimal pairs each, plus the
four scrambled controls per (size, instance). Schedule `--outer 150 --steps 1
--segments 64`, matching the existing fine runs. Near-optimality filter
epsilon = 0.2, unchanged.

## The analysis, fixed in advance

**(i) Statistic.** `rho = string_min / min(delta_A, delta_B)`. Per (size, instance):
the median over the twelve pairs. Per size: the mean of the three instance medians,
with uncertainty the **standard error over those three instances**. That standard
error, not the pair bootstrap, is the error bar of record.

**(ii) Test.** Weighted least squares of `ln rho` against `n` (exponential) and
against `ln n` (power law), compared by weighted chi^2 on three degrees of freedom,
together with the interval-exponent test for a single fixed exponent.

**(iii) Verdict.**

- If the exponential attains the lower chi^2 **and** a fixed exponent is excluded at
  p < 0.05: the claim that the corrugation deepens faster than any fixed power
  stands, now carried by instance-level errors.
- Otherwise — the power law fitting at least as well, **or** a fixed exponent
  surviving at p < 0.05 — that claim is **withdrawn from the abstract** and replaced
  by "the corrugation deepens with n; these data do not determine the functional
  form". The C-shelf conjecture drops its rate clause, and registered prediction 4
  is recorded as falsified as stated and superseded by a deepening of undetermined
  form.

**(iv)** Both fits and the complete per-instance table are reported whichever way
the verdict falls. No statistic is selected after the data are seen.

## Note on the earlier registrations

Each registered prediction that has failed so far was read as strengthening the
conjecture, so no registered outcome has yet cost the paper a claim. This
registration is written so that its failure removes a sentence from the abstract.

## Provenance

- Solutions paired: `results/pq/pq_n{n}_i{k}_sigma0.1.npz`, instances k = 0, 1, 2.
- Runner: `analysis/connectivity.py --num-qubits {n} --instance {k} --outer 150
  --steps 1 --segments 64`.
- Instance 0 keeps its original seed spawn key and file names, so every archive
  committed before this sweep regenerates unchanged.
- The new runs execute on x86 (`n2-highcpu-96`); the architecture of each block is
  recorded with its log, since optimizer trajectories are not bit-portable across
  architectures.
