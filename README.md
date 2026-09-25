# PoseTrust

**Under which conditions is the covariance that pose-graph SLAM reports
statistically honest, and when it is not, does it fail cautiously or
overconfidently?**

Where the problem is close to linear (small rotational noise, and no false
loop closures left in the graph) the reported covariance is honest. When it
fails, it fails in the dangerous direction, claiming more certainty than it
has: from about 0.06–0.1 rad of rotational noise per measurement, severely
under uncaught false loop closures, and even after a convex robust kernel
(Huber) has restored the trajectory itself. Sparse loop closures matter far
less than either; of the robust back-ends tested, only switchable
constraints kept the covariance honest at every outlier rate.

![E4: trajectory error and calibration against the outlier rate](figures/e4_perceptual_aliasing.svg)

*A calibrated covariance sits at 1 on the lower panel. Huber brings the
trajectory back (top) while its covariance grows steadily more
overconfident (bottom): an evaluation that scored only accuracy would call it
a success.*

---

## How it is measured

A SLAM back-end reports a covariance alongside its estimate: a claim about
how far the estimate could plausibly be from the truth. On real data that
claim cannot be checked, because there is one dataset and one answer. In
simulation it can: the same measurement set is solved under hundreds of
independent noise draws, and the spread of the estimates around the known
truth is compared with the spread the solver claimed.

The comparison is the normalised estimation error squared,
NEES = eᵀ Σ⁻¹ e, with e = log(estimate⁻¹ ∘ truth) the error on the manifold
and Σ the reported covariance. If Σ is honest, NEES follows a chi-squared
distribution with one degree of freedom per estimated coordinate, so its
mean divided by the degrees of freedom is 1. Above 1 the solver is
**overconfident**; below 1 it is **conservative**. Each condition is judged
against the chi-squared acceptance band for its mean, with a bootstrap
interval on the estimate and a Benjamini-Hochberg correction across every
condition in an experiment.

The predictions, analysis rules and every later amendment are in
[HYPOTHESES.md](HYPOTHESES.md), which was committed before the final runs.

## Results

Each experiment runs 200 Monte Carlo runs per condition, over SE(2) and
SE(3) pose graphs of 6–20 poses.

| Experiment | Question | Result | Hypothesis |
|---|---|---|---|
| **E1** validation gate | Where the covariance is provably exact, does the implementation agree? | Mean NEES / dof 0.975 (SE(2)), 0.993 (SE(3)); marginals match their closed form to 4e-15 | Gate passes |
| **E2** loop-closure density | Does thinning loop closures make the covariance overconfident? | Only slightly: at most +4.7% (SE(3), no closures), significant only at the sparsest SE(3) settings | H1: direction supported, "monotone" not |
| **E3** rotational noise | Does rotational non-linearity break calibration? | Yes, from 0.06 rad (SE(3)) and 0.10 rad (SE(2)); at 0.22 rad only 1.5% of SE(3)'s 95% ellipsoids contain the truth | H2: supported; "abrupt" was never defined |
| **E4** false loop closures | Do robust back-ends restore calibration along with accuracy? | Huber: accuracy yes, calibration no (1.35 → 3.25). Switchable: both. Cauchy, GNC: accurate but conservative | H4: half supported |

![E3: calibration against rotational noise](figures/e3_nonlinearity.svg)

Every figure is regenerated from the committed results in `results/`:
[E1](figures/e1_validation_gate.svg),
[E2](figures/e2_loop_closure_density.svg),
[E3](figures/e3_nonlinearity.svg),
[E3 coverage](figures/e3_coverage.svg),
[E4](figures/e4_perceptual_aliasing.svg).

## Limitations

- **Simulation only.** The real-data experiment (E5) was cut; nothing here
  speaks to public benchmarks. See the 2026-09-25 amendment in
  HYPOTHESES.md.
- **One graph per experiment.** Each sweep holds its graph fixed so that
  conditions are compared run by run, which means magnitudes belong to that
  graph. A check made after the runs, over eight random graphs at 0.15 rad,
  found the direction stable (never conservative, 14 of 16 significantly
  overconfident) but the size varying up to five-fold: mean NEES / dof from
  1.00 to 2.71 in SE(2) and 1.10 to 5.18 in SE(3). E3's graph is at the
  severe end.
- **Large ratios are driven by a minority of runs.** At high noise the mean
  NEES is 1.5–2.8 times the median, and the worst 5% of runs carry a quarter
  of the total. The verdicts are robust to this; the magnitudes should be
  read alongside the coverage figure.
- **Cauchy and GNC look calibrated partly for the wrong reason.** Both
  down-weight correct measurements as well as false ones, which inflates the
  covariance they report: they read conservative even with no outliers.
- **Small graphs, dense solves.** At most twenty poses; the normal equations
  are solved densely, which does not scale to real maps.
- **Defects found and fixed after the first runs.** Mislabelled sweep
  conditions, a different random graph per condition, and iteration budgets
  that recorded slow convergence as failure. All results were regenerated;
  the details are in HYPOTHESES.md.

## Reproducing

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest                                      # the test suite, under two minutes
uv run python experiments/e1_validation_gate.py    # E1: about a minute
uv run python experiments/e4_perceptual_aliasing.py --figures-only
```

Each experiment script writes `results/<name>.parquet`, then draws its
figure from that file. `--figures-only` skips the runs and redraws from the
committed results in seconds. On a laptop CPU, E2 takes about ten minutes
and E3 up to an hour; E4 runs its conditions in parallel and takes about
twenty minutes on twelve cores. Figures are vector SVG and need no raster
backend.

## Using the package

```python
import numpy as np

import posetrust
from posetrust.lie import se2
from posetrust.optimize.optimizer import levenberg_marquardt
from posetrust.simulate import NoiseModel, make_scenario, monte_carlo

scenario = make_scenario(se2, n_poses=10, loop_density=0.3, seed=0)
noise = NoiseModel(np.array([0.02, 0.02, 0.15]))  # x, y, heading
result = monte_carlo(se2, scenario, noise, n_runs=200, solver=levenberg_marquardt)

report = posetrust.consistency(result, se2)
print(report.summary())
# consistent     mean NEES    27.979 (dof 27, band [25.99, 28.03], p=0.0617, 200 runs)
```

`report` also exposes `.coverage_curve()`, per-pose results with
`.pose(k)`, and the translation/rotation split with `.by_dof()`.

## Layout

```
src/posetrust/
  lie/            SE(2) and SE(3): exp, log, Jacobians, adjoint
  graph.py        pose graph, residuals, analytic Jacobians
  optimize/       Gauss-Newton, Levenberg-Marquardt, gauge fixing,
                  robust kernels (IRLS, graduated non-convexity)
  covariance.py   marginal and relative covariances by selected inversion
  simulate.py     scenarios, noise, the Monte Carlo harness
  stats.py        NEES, chi-squared bands, coverage, multiplicity
  report.py       the public consistency() API
experiments/      E1-E4, one script each, plus shared analysis rules
results/          per-condition results (parquet)
figures/          the figures, regenerated from results/
tests/            unit tests, and structural checks on every figure
HYPOTHESES.md     predictions and analysis rules, fixed before the runs
```
