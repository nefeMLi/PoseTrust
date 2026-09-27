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

---

## Amendment, 2026-09-25: scope, and two terms left undefined

**E5 is cut, and H5 is left unanswered.** The cut rules above allow it. E5
needed public benchmarks, file loaders and a sparse solver, none of which
were built: every graph in E1-E4 has at most twenty poses and is solved
densely. H5 is neither confirmed nor refuted, and nothing in the results
speaks to real data.

**"Abrupt" in H2 was never defined.** No criterion was fixed in advance
for what separates an abrupt onset from a steep gradual one, so H2's second
clause cannot be judged against this document. E3 is reported
descriptively instead: calibration holds up to a threshold and then
degrades rapidly and monotonically, with no step change between
neighbouring noise levels.

**"Recovered accuracy" in E4 is an analysis choice, made after the first
runs.** A method counts as having recovered at a rate when its trajectory
error there is below twice its own error with no outliers. The threshold
was not fixed here in advance and is reported as a choice, not a rule.

---

## Amendment, 2026-09-27: E3 across graphs

Written before the new runs. E3 used one graph, and a post-hoc check at
0.15 rad found the size of the effect varied up to five-fold across graphs.
E3 is therefore repeated on eight loop-closure layouts of the same
trajectory (scenario seeds 200, 300, ..., 900; seed 200 is the original
graph), with the same noise levels, 200 runs per condition and
Benjamini-Hochberg within each layout's sweep.

This is exploratory, not a new prediction. For each layout it reports the
first noise level that is overconfident after correction, and the spread of
NEES/dof at each level. If the breaking points differ by more than one grid
step, the README gives them as a range rather than quoting the original
graph's values.

---

## Amendment, 2026-09-27: E4 across graphs, and a naming correction

**"Switchable constraints" above means dynamic covariance scaling.** The
back-end tested in E4 applies the closed-form DCS weight of Agarwal et al.
(2013); it does not optimise switch variables, and its covariance comes from
the weighted information matrix rather than from a system with switches in
it. Every result labelled switchable constraints is a DCS result. The code
and README now say DCS; the text above is left as it was written.

**E4 is repeated on eight loop-closure layouts** (scenario seeds 300, 400,
..., 1000; seed 300 is the original graph), with the same methods, outlier
rates, 200 runs per condition and Benjamini-Hochberg within each layout.
Written before the new runs.

Fixed before seeing them: the README keeps E4's headline, that Huber
recovers accuracy but stays overconfident while DCS stays calibrated, only
if it holds on at least six of the eight layouts, judged at outlier rates of
10% and above. Huber holds on a layout if it is overconfident after
correction at every such rate; DCS holds if it is overconfident after
correction at none of them. Otherwise E4 is reported as layout-dependent, with
the counts.

---

## E5: what covariance should a robust back-end report?

Dated 2026-09-27, before any of the code below exists. The E5 of the
original plan (real data) was cut; this E5 replaces it.

### The question

After IRLS converges, every back-end here reports `(JᵀWJ)⁻¹`, the inverse of
the information matrix with the final robust weights treated as if they were
known constants. E4 shows that this is overconfident for Huber and
conservative for Cauchy and GNC. Is there a covariance that stays calibrated
for the same estimate?

Statistics already has a candidate: the sandwich covariance of an M-estimator
(Huber, 1967; White, 1980). It corrects the variance when the model is wrong
but, as Freedman (2006) points out, it does nothing about bias. A search of
the robust-SLAM literature found no study of the reported covariance's
calibration, only of trajectory accuracy. So the open part is whether
Huber's overconfidence is a variance problem, which a better covariance can
fix, or a bias problem, which it cannot.

### Covariances compared

All are computed from the same converged estimate; only the reported
covariance changes. With whitened residuals `e_i`, whitened Jacobians `J_i`,
`s_i = |e_i|²` and the kernel `ρ(s)`, so that `w_i = ρ'(s_i)`:

| Name | Covariance |
|---|---|
| naive | `(Σ w_i J_iᵀJ_i)⁻¹`, what every back-end reports now |
| sandwich | `A⁻¹ B A⁻¹`, with `A = Σ (w_i J_iᵀJ_i + 2ρ''(s_i) J_iᵀe_i e_iᵀJ_i)` and `B = Σ w_i² J_iᵀe_i e_iᵀJ_i` |
| inlier | `(Σ_{i kept} J_iᵀJ_i)⁻¹`, dropping loop closures with `s_i` above the 95% chi-squared threshold used in E4 |

Odometry factors have `w_i = 1` and `ρ'' = 0` throughout, as in E4. Each is
applied to Huber, Cauchy, DCS and GNC at every E4 outlier rate.

**Gate.** With plain least squares and no outliers the naive covariance is
calibrated (E1, E4), and the sandwich must be too. It may not be: E4's graph
has 57 free parameters against 117 residuals, and the plain sandwich
underestimates the variance when the parameters use that large a share of
the residual degrees of freedom. If it fails the gate on the development
layouts, it is replaced before any test run by the leverage-corrected form,
`B = Σ w_i² J_iᵀ (I − P_i)^(-1/2) e_i e_iᵀ (I − P_i)^(-1/2) J_i`, where
`P_i = w_i J_i A⁻¹ J_iᵀ` is the factor's block of the hat matrix. If that
also fails, H6b and H6d are reported as untestable here. A covariance passes
the gate if it is calibrated on at least six of the eight development
layouts (added 2026-09-27, before the gate was run).

### Bias and variance

Across the Monte Carlo runs of one condition, with mean error `ē` over runs,
the mean NEES splits into a spread term and a bias term
`b = mean_k ēᵀ Σ_k⁻¹ ē`. Reported for every condition, per covariance, as
`b / dof` next to NEES/dof.

### Predictions

**H6a (bias).** For Huber at outlier rates of 10% and above, the bias term
accounts for more than half of the excess NEES (`b > (NEES − dof) / 2`) under
the naive covariance. Confidence low; it could equally be the weights.

**H6b (sandwich).** The sandwich covariance lowers Huber's NEES/dof at every
corrupted rate but leaves it overconfident at 20% and above. Falsified if it
is calibrated there, which would mean the problem was variance after all.

**H6c (inlier).** The inlier covariance is calibrated for Huber wherever the
threshold flags exactly the false closures, and overconfident where it
misses one. Reported with the share of runs where the flagged set is exact.

**H6d (conservative kernels).** The sandwich covariance removes the
conservatism of Cauchy and GNC at 0% outliers.

### Analysis rules, fixed now

- **Development and test layouts.** Everything is developed on the E4
  layouts, seeds 300 to 1000. The final numbers come from eight new layouts,
  seeds 1100 to 1800, run once, after the code and these rules are
  committed. Any change made after seeing development results is recorded
  here as an amendment before the test run.
- **Calibrated** means not overconfident and not conservative after
  Benjamini-Hochberg, within each layout's sweep, as in E4.
- **A covariance fixes Huber** if Huber under it is calibrated at every rate
  of 10% and above on at least six of the eight test layouts, the same form
  as the E4 rule.
- 200 runs per condition, the E4 noise model, trajectory and outlier rates.

### Cut rules

- If the sandwich covariance cannot be made numerically stable (`A` not
  positive definite), that is reported, not patched by adding damping after
  the fact.
- The test layouts are run once. A bug found after that run is fixed, the
  fix recorded here, and both results reported.

### Amendment, 2026-09-27: the sandwich as written cannot exist here

Found in a five-run smoke test of the code, before the gate or any
development run. The sandwich above estimates its middle term from the
residuals, `B = Σ w_i² J_iᵀe_i e_iᵀJ_i`. Each factor adds a rank-one term, and
E4's graph has 39 factors against 57 free parameters, so `B` is singular for
every estimate and the NEES under it is meaningless (the smoke test gave
values near 10¹⁴). This is structural, not a bug, and the leverage-corrected
form has the same rank. Both are dropped.

In their place, the model-based sandwich, `A⁻¹ (Σ w_i² J_iᵀJ_i) A⁻¹`, which
takes the middle term's expectation under the noise model instead of
estimating it. It suits this problem: a false closure here is normal noise
around a wrong endpoint, so its noise covariance is the modelled one, and
its offset shows up in the bias term rather than the variance. For plain
least squares it equals the naive covariance exactly, so the gate is met by
construction; the gate rows are kept as a check of the code. H6b and H6d
refer to this sandwich. Nothing else changes.

### A scalar model, worked out before the development run

One unknown θ, n inliers `y = θ + ε` and k false measurements
`y = θ + Δ + ε`, with `ε ~ N(0, 1)` and `Δ` well beyond the Huber threshold
`δ`. Each outlier then pulls on the estimate with a fixed force `δ`, so

- `θ̂ = ȳ + kδ/n`: variance `1/n`, bias `kδ/n`;
- naive covariance `1/(n + kδ/Δ)`, so `E[NEES] ≈ (1 + k²δ²/n)(1 + kδ/(nΔ))`;
- in Huber's linear region `w + 2ρ''s = 0`, so `A = n`, the sandwich is
  about `1/n`, and `E[NEES] ≈ 1 + k²δ²/n`;
- the inlier covariance is `1/n` as well, with the same `E[NEES]`.

The bias term `k²δ²/n` is what is left once the variance is right, and no
covariance can remove it. A redescending kernel gives a gross outlier almost
no pull, which is why DCS avoids it. The model agrees with H6a and H6b and
disagrees with H6c, which predicted the inlier covariance calibrated. H6c is
left as written and will be judged as written.

### Amendment, 2026-09-27: after the development run, before the test run

**What the development layouts showed.** The gate passes on all eight. For
Huber at 10% outliers and above, the bias term is 98-100% of the excess NEES
on every layout, and no covariance fixes Huber on any of them. The sandwich
makes Cauchy and GNC slightly more conservative at 0% outliers, not less.

**That last result exposed a mistake in the previous amendment.** It called
`Σ w_i² J_iᵀJ_i` the middle term's expectation under the noise model. It is
not: `w_i` depends on the residual, so the expectation is `E[w(s)² s] / d`
with `s ~ χ²(d)`, not the observed `w_i²`. Because `w` falls as `s` grows,
the observed version overstates the middle term, which is why the sandwich
came out conservative. The sandwich stays as committed, and H6b and H6d are
judged on it as written.

**Added, as exploratory.** The textbook form, with both terms replaced by
their expectations under the noise model: `a = E[w + 2ρ''s/d]` and
`c = E[w² s]/d` for robust factors, 1 for odometry, giving
`(Σ a_i J_iᵀJ_i)⁻¹ (Σ c_i J_iᵀJ_i) (Σ a_i J_iᵀJ_i)⁻¹`. It was added after
the development results, so nothing it shows is confirmatory. Seen on
development layout 300 before this amendment was written: at 0% outliers it
gives 0.99 for Cauchy and 1.03 for GNC, where the naive covariance gives 0.94
and 0.93; for Huber at 20% it gives 5.16, against 2.06 naive, because it
treats the false closures as genuine.

**H6c, measured properly.** The development run only recorded how often the
threshold flags exactly the false closures (3-10% of runs). The test run
also records the NEES of those runs and of the rest, pooled over layouts, so
H6c can be judged on the runs it is about.

Nothing else changes. The test layouts have not been run.

---

## E6: report the bias as well as the variance

Dated 2026-09-27, after the E5 test run and before any E6 code exists.

### The question

E5 found that Huber's covariance describes the scatter of its errors well and
misses a shift: the false closures it keeps pull the estimate off, and that
offset is almost all of the excess NEES. If the shift can be estimated from
the solution itself, reporting the mean squared error instead of the variance
should make Huber honest without changing its estimate. Does it?

### Prior work

GNSS integrity monitoring (RAIM and ARAIM, e.g. Blanch, Walter and Enge,
2012) projects the bias of a faulty measurement through the least-squares
solution into a bound on the position error. E6 borrows that projection. It
differs in estimating the pull a robust kernel's down-weighted factors
actually exert at the solution, rather than bounding hypothetical faults, and
in asking for a calibrated covariance rather than a bound. Robust statistics
knows Huber's bias is bounded and non-zero (influence function); robust
pose-graph work removes outliers but does not test the covariance it reports.

### The method

At a converged robust estimate the gradient vanishes, so the pull of the
down-weighted factors P is balanced by the rest T:
`Σ_T J_iᵀe_i = −Σ_P w_i J_iᵀe_i`. Removing P would move the estimate by one
Gauss-Newton step, so the estimated shift is

    b̂ = −H_T⁻¹ Σ_P w_i J_iᵀ e_i,    H_T = Σ_T J_iᵀJ_i,

and the reported covariance becomes `Σ_E6 = Σ_naive + b̂ b̂ᵀ`. In the scalar
model this gives `b̂ = kδ/n`, the bias derived there; a test checks it.

P is the set of loop closures whose squared residual at the solution exceeds
a threshold q. Correct closures above q add noise to `b̂`, so q trades
missed false closures against spurious inflation. q is chosen on the
development layouts from the χ²(3) quantiles {0.95, 0.99, 0.999}, as the one
whose Huber NEES/dof is closest to 1 on average over development layouts and
rates of 10% and above, and fixed before the test run.

### Comparator: reject and refit

What an engineer would try first: drop the closures in P, re-solve with
plain least squares from the robust estimate, and report that solution with
its own covariance. Unlike E6 it changes the estimate.

### Layouts

Development: the E4 layouts, seeds 300 to 1000, trajectory turn 0.25.
Test, run once: eight graphs on trajectories none of the earlier experiments
used, turn 0.10 and 0.40, with scenario seeds 1900, 2000, 2100 and 2200 for
each. Everything else as in E4: 20 poses, 20 closures, the E4 outlier rates,
noise and 200 runs per condition. Huber and DCS are run; DCS is the check
that the method leaves a back-end that is already calibrated alone.

### Predictions

**H7a (primary).** Under `Σ_E6`, Huber's NEES/dof lies within [0.9, 1.1] at
every rate of 10% and above on at least six of the eight test graphs.
Falsified otherwise. Also reported: the strict E5 criterion (calibrated
after Benjamini-Hochberg), which is not expected to hold everywhere, since
at 200 runs it flags errors of a few percent.

**H7b.** At 0% outliers `Σ_E6` does not make Huber conservative: NEES/dof in
[0.9, 1.1] on at least six of eight test graphs.

**H7c.** For DCS, `Σ_E6` changes NEES/dof by less than 0.05 at every rate on
at least six of eight test graphs, because its weights on gross outliers are
near zero.

**H7d (comparator).** Reject-and-refit is also within [0.9, 1.1] at every
rate of 10% and above on at least six of eight test graphs. If it is, E6's
case rests on keeping the estimate and on the explanation, not on being the
only fix.

Confidence: moderate for H7a. `b̂` is a first-order estimate, and large
pulls on these small graphs may be beyond first order; the scalar model says
nothing about how the pull spreads through a graph.

### Rules

- Only q is tuned, only on development layouts, only by the rule above. Any
  other change after development results is recorded here before the test
  run.
- The test graphs are run once. A bug found afterwards is fixed, recorded,
  and both results reported.
- Non-converged runs are excluded and counted, as before.
