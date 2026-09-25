# Hypotheses

Committed before the final experimental runs, so the predictions below can be
checked against results rather than written to match them. Where a result has
already been seen, this document says so instead of dressing an observation up
as a forecast — a pre-registration that quietly includes things already
measured is worse than none, because it looks like evidence and is not.

Dated 2026-09-21. Machinery in place at that point: SE(2)/SE(3), factor graph,
Gauss-Newton and Levenberg-Marquardt, selected-inversion covariance, the Monte
Carlo harness, the NEES and coverage statistics, and the robust back-ends.
Still to build: sparse Cholesky, benchmark loaders, the experiment scripts.

---

## The question

Under which operating conditions does the covariance reported by pose-graph
SLAM remain statistically consistent with the true estimation error, and does
it fail conservatively or optimistically?

Only one direction of failure matters. A covariance that is too large makes a
system slow and cautious. One that is too small makes it confident about a
position it does not hold, and that is the failure that causes harm. Every
hypothesis below is therefore stated in terms of *signed* miscalibration, not
its magnitude.

---

## Analysis decisions, fixed in advance

These are the choices that could otherwise be made after seeing results, which
is where most accidental self-deception in this kind of study comes from.

| Decision | Value | Why fixed now |
|---|---|---|
| Statistic | Mean NEES over runs, against the chi-squared acceptance band | Whole distribution also reported; the mean is what the verdict keys on |
| Error definition | `log(estimate⁻¹ ∘ truth)`, the manifold error | A naive parameter difference is a different quantity and would answer a different question |
| Significance level | α = 0.05, two-sided | |
| Multiplicity | Benjamini-Hochberg across all conditions within an experiment | A 20-condition sweep flags one clean condition by chance at α = 0.05 |
| Non-converged runs | Excluded, and the count reported alongside every result | Pooling converged and diverged solutions inflates NEES. This already happened once (see below) |
| Minimum converged fraction | 50%; below that the condition is reported as a convergence failure, not a calibration result | Prevents a broken condition being laundered into an "overconfident" one |
| Gauge | One pose anchored, held at its true value in every run | Otherwise the spread across runs is dominated by gauge freedom |
| Initialization | At ground truth unless stated | Separates calibration from convergence; E4 additionally uses dead reckoning, where convergence is part of the question |
| Runs per condition | ≥ 200 | The band must be narrow enough to detect the effects claimed |
| Interval on every point estimate | Yes | |

A hypothesis is **not** confirmed by a verdict of `OVERCONFIDENT` alone. It
requires the predicted direction, surviving multiplicity correction, with the
converged fraction above threshold.

---

## Predictions

### H1 — Loop-closure density (Q1, experiment E2)

**Prediction.** Overconfidence increases as loop-closure density falls. The
odometry-only end of the sweep is the worst-calibrated, and consistency
improves monotonically as constraints are added.

**Reasoning.** Sparse graphs leave long unconstrained stretches over which
error accumulates non-linearly, and the linearized covariance cannot represent
that growth.

**Falsified by.** Calibration flat across the density sweep, or worse at high
density.

**Confidence.** Moderate. The direction seems forced, but a monotone trend is
a stronger claim than "sparse is bad" and may not hold.

### H2 — Rotational non-linearity (Q2, experiment E3)

**Prediction.** Overconfidence grows with rotational noise, and the onset is
abrupt rather than gradual — a regime where the covariance is fine, then a
narrow band, then rapid collapse.

**Status: already observed in preview, not a prediction.** See below.

### H3 — SE(3) versus SE(2) (Q2, experiment E3)

**Prediction.** At matched rotational uncertainty per degree of freedom, SE(3)
is materially worse calibrated than SE(2).

**Reasoning.** SE(3) rotations do not commute and its exponential map carries
a translation–rotation coupling term with no SE(2) analogue, so the Gaussian
fitted at the mode should be a poorer description of the true posterior.

**Falsified by.** No systematic gap, or SE(2) worse.

**Confidence.** Low to moderate. "Matched uncertainty" across groups of
different dimension is not a clean comparison, and the result may be an
artefact of how the match is defined. If no fair matching can be constructed,
this hypothesis will be withdrawn rather than answered badly, and the
withdrawal recorded here.

> **WITHDRAWN 2026-09-22, before E3 was run.** No fair matching exists, and
> every candidate decides the answer in advance:
>
> - *Same per-axis sigma.* SE(3)'s rotation-error magnitude is then larger by
>   root three, so it meets more non-linearity by construction rather than by
>   anything intrinsic to the group.
> - *Same total rotation-error magnitude.* Fairer on that axis, but SE(3)
>   still carries three noisy degrees of freedom SE(2) does not have. It is a
>   different estimation problem, not the same one in a bigger group.
> - *Same trajectory.* A trajectory that only turns about z makes SE(3)
>   literally the SE(2) subgroup — already verified as an exact identity in
>   the tests — so the comparison is degenerate. Tilting the axis to exercise
>   SE(3) means the two groups no longer share a trajectory.
>
> Withdrawn rather than answered badly, as this section committed to.
>
> Replaced, **as an exploratory question and not a prediction**, by one that
> is well posed: at what rotation-error magnitude does each group lose
> calibration? That is a per-group threshold in radians, so it compares where
> each breaks rather than their values at an arbitrarily matched sigma, and it
> needs no cross-dimensional matching. Any result from it is exploratory and
> must be labelled so.

### H4 — Robust back-ends and calibration (Q3, experiment E4)

**Prediction.** Robust back-ends restore trajectory accuracy under perceptual
aliasing without restoring calibration; the covariance keeps lying while the
estimate recovers.

**Status: partly observed in preview, and the preview contradicts the simple
form of this hypothesis.** See below.

**Refined prediction, still open.** Whether calibration is restored depends on
whether the kernel redescends. Convex kernels (Huber) recover accuracy while
leaving the covariance overconfident; redescending kernels (Cauchy, switchable
constraints, graduated non-convexity) recover both. This is expected to hold
across outlier rates from 0 to 30%, and the gap between convex and redescending
kernels should widen with the rate.

**Falsified by.** Redescending kernels also miscalibrated at higher rates, or
Huber calibrated at rates other than the single one tested, or the separation
disappearing once the outlier rate is swept properly.

### H5 — Real data (experiment E5)

**Prediction.** On public benchmarks, held-out loop closures produce residuals
larger than the optimized graph's covariance predicts — the same overconfidence,
visible without a ground-truth posterior.

**Falsified by.** Held-out residuals consistent with predicted uncertainty, or
conservative.

**Confidence.** Low. This is the least controlled arm; benchmark noise models
are not the ones the data was generated with, so a negative result is
interpretable but a positive one is confounded.

---

## Already observed before this document was written

Recorded so they are not later mistaken for successful predictions. Each was a
single trajectory, single seed, single parameter setting — enough to establish
a direction, not a result.

1. **Rotational-noise onset (H2).** Mean NEES / dof of 0.98, 0.99, 1.73, 12.1,
   88.7 at rotational σ of 0.005, 0.05, 0.15, 0.30, 0.50 on one SE(2)
   trajectory. Onset is abrupt. A per-degree-of-freedom split suggested that
   rotational noise degrades the *translational* covariance while the rotational
   part stays calibrated — a sharper claim than H2 as originally stated, and the
   one E3 should actually test.

2. **The σ = 0.50 figure above was contaminated.** It pooled 27 of 120
   non-converged runs. Restricted to converged runs it is 68.5, not 88.7. The
   direction survived; the number did not. This is why the exclusion rule is
   fixed in the table above.

3. **Robust kernels split by convexity (H4).** At a 20% outlier rate, 80 runs:
   plain least squares RMS 0.567 / NEES-per-dof 341.8; Huber 0.081 / 3.28
   (overconfident); Cauchy 0.058 / 1.00, switchable constraints 0.058 / 1.03,
   graduated non-convexity 0.058 / 0.97 (all consistent). Huber lands in the
   dangerous quadrant — accuracy restored, covariance not.

4. **The linear-Gaussian gate passes.** Where the Laplace covariance is provably
   exact, mean NEES / dof is 0.998 (SE(2), dof 21) and 0.993 (SE(3), dof 30).
   Any miscalibration found elsewhere is therefore a property of the problem
   rather than of this implementation.

---

## Stopping and cut rules

- E1 is the gate. If the linear-Gaussian case ever stops being consistent, no
  other result is reportable until it does again.
- Cut order if time runs short: SE(3) arm of E3 (H3 withdrawn), then a robust
  method from E4, then E5. E5 is on the list but cutting it weakens the work
  noticeably; prefer cutting an E4 kernel first.
- Never cut: the validation gates, the intervals, the limitations section, the
  package API.
- A null result is reportable. "The reported covariance is well calibrated
  across the operating range" would be a useful finding and is not a failed
  project.

---

## Amendment, 2026-09-24: defects found after the first runs

An audit after E2-E4 had been run once found defects in how the conditions
were generated and how convergence was judged. None of them touches a
prediction or an analysis rule above; all of them affected which numbers
those rules were applied to. Every experiment was re-run after the fixes,
and the earlier results are superseded rather than kept alongside.

1. **Sweep labels did not match the conditions.** Rates and densities were
   rounded to whole counts, silently. In E4, ten closures at 5% was zero
   outliers, and 15%, 20% and 25% were all two. In E2, densities 0.05 and 0.1
   on twelve poses were both one closure. Counts are now required to be
   exact, and E2 and E4 use twenty poses so that every swept value is a
   distinct, exact count.

2. **Each condition drew a different graph.** Every level of every sweep
   used its own seed, so neighbouring conditions differed in which closures
   existed, and which were false, as well as in the swept quantity. Closures
   and outliers are now nested prefixes of one fixed random ordering, false
   endpoints are fixed per scenario rather than redrawn per run, and one seed
   is held across each sweep, so conditions are paired run by run.

3. **Slow convergence was recorded as failure.** Gauss-Newton had a budget of
   50 iterations and Levenberg-Marquardt 100, with a damping rule (divide by
   ten on success, multiply by ten on failure) that oscillates in curved
   valleys. Under large rotational noise, or with false closures leaving
   large residuals, convergence is linear and slower than those budgets
   allowed. Every run then reported as non-converged converges when given
   more iterations. Because non-converged runs are excluded (rule above),
   E3 dropped its most non-linear runs, biasing high-noise ratios towards
   calibration, and E4 reported plain least squares as failing to converge
   when it converges to a wrong answer. Levenberg-Marquardt now uses
   Nielsen's gain-ratio damping and a stopping rule on the Newton decrement;
   budgets are sized as a safety net rather than a test.

   Observation 2 above, the 27 non-converged runs pooled into 88.7, may be
   the same artefact. It still shows why pooling is wrong; it does not show
   that those runs had diverged.

4. **Verdict counts ignored multiplicity.** E4's summary counted raw
   `OVERCONFIDENT` verdicts. It now requires surviving Benjamini-Hochberg,
   as the confirmation rule above already said.
