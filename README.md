# PoseTrust

[![tests](https://github.com/nefeMLi/PoseTrust/actions/workflows/tests.yml/badge.svg)](https://github.com/nefeMLi/PoseTrust/actions/workflows/tests.yml)

When a pose-graph SLAM solver gives you a trajectory, it also gives you a covariance, which is basically the solver
saying "this is how wrong I might be". A planner or a safety check will usually take that number at face value.
This project checks whether it deserves that trust.

I simulate the pose graphs, so I know the true trajectory. Each graph gets solved 200 times with different noise,
and I compare the real errors with what the covariance claimed. The measure is NEES: per degree of freedom it
should come out around 1. Above 1 means the solver thinks it is more accurate than it really is, which is the
dangerous case.

## What I found

**Rotational noise.** With small heading noise the covariance is fine. Past roughly 0.06 to 0.1 rad per
measurement it becomes overconfident, and it gets bad fast: at 0.22 rad in SE(3), the nominal 95% confidence
region for the whole trajectory contains the truth in only 1.5% of runs. Where exactly it breaks varies a lot
between graphs, but it never goes the cautious way.

![Where calibration breaks depends on the layout](figures/rotational_noise.svg)

**False loop closures.** I added wrong loop closures (a measurement pointing at the wrong pose) and compared four
robust back-ends. Huber keeps the trajectory far closer to the truth than plain least squares does, but its
covariance is overconfident on all 8 layouts, on some by more than a factor of 10. DCS stays honest on 6 of the 8.
Cauchy and GNC are mostly a little too cautious.

![Calibration of each robust back-end on eight layouts](figures/false_loop_closures.svg)

**Why Huber gets it wrong.** Huber turns the weight of a bad closure down but never to zero, so every bad closure
keeps pulling the estimate a bit, and always in the same direction. That makes the error a bias rather than
noise, and a covariance can't describe a shift that is the same in every run. When I split Huber's extra NEES into
spread and bias, the bias term explains roughly 90 to 98% of it (each run has its own covariance, so this is a
close diagnostic rather than an exact split). If I drop the closures Huber flags as suspicious and solve again,
NEES per dof comes back to about 1 on all 8 layouts. I picked the threshold for flagging on these same
simulations, though, so take the refit result as exploratory rather than a clean test.

![Huber's extra NEES is mostly bias, and refitting without the flagged closures fixes it](figures/huber_bias.svg)

## Things that didn't work out the way I expected

- I thought sparse graphs (few loop closures) would be badly calibrated. The difference was only a few percent, so
  I dropped that experiment.
- I thought a better covariance formula would fix Huber, for example the sandwich estimator from robust
  statistics. None of the ones I tried did, and that is how I ended up looking at bias.
- I also tried adding an estimate of the bias to Huber's covariance. It removed most of the problem but not all of
  it, so refitting is the fix I report.

## Limitations

- It's all simulation, with small graphs (at most 20 poses). My false closures are always "right measurement, wrong
  pose", and real outliers can look different.
- The robust back-end results come from one trajectory with eight different loop-closure layouts.
- The refit threshold (the 99.9% chi-squared quantile) was chosen on the same layouts it is reported on.
- The error I use to compare back-ends mixes metres and radians, so it only works as a relative comparison.

## Background

Filters like EKF-SLAM are already known to become overconfident when linearisation breaks down (Bailey et al., IROS
2006; Huang, Mourikis and Roumeliotis, IJRR 2010), so the rotational noise part is really the pose-graph version of
that. The robust back-ends are Huber, Cauchy, dynamic covariance scaling (Agarwal et al., ICRA 2013) and graduated
non-convexity (Yang et al., RA-L 2020). They are normally judged on how good the trajectory is; I was interested in
the covariance they report.

## Running it

Python 3.14:

```sh
pip install -r requirements.txt
python -m experiments.rotational_noise      # about 6 minutes on 12 cores
python -m experiments.false_loop_closures   # a few hours; --huber-only runs just the Huber part (~20 min)
pytest tests.py
```

Results go to `results/` and figures to `figures/`. Add `--figures-only` to redraw the figures from saved results.
The tests check the Jacobians against finite differences, check that the covariance is honest in a case where it
should be, and compare estimates and covariances with GTSAM (that last one only runs on GitHub Actions, since
GTSAM doesn't install on Windows).

The code is in `posetrust/` (SE(2) and SE(3), the pose graph, the solvers, the robust kernels, covariance extraction,
the Monte Carlo loop and the NEES statistics), and the two experiments are in `experiments/`. A quick example:

```python
import numpy as np
from posetrust import se2
from posetrust.optimizer import levenberg_marquardt
from posetrust.simulate import NoiseModel, make_scenario, monte_carlo
from posetrust.stats import consistency

scenario = make_scenario(se2, n_poses=10, loop_density=0.3, seed=0)
result = monte_carlo(se2, scenario, NoiseModel(np.array([0.02, 0.02, 0.15])), n_runs=200, solver=levenberg_marquardt)
print(consistency(result).summary())
# consistent     mean NEES    27.979 (dof 27, band [25.99, 28.03], p=0.0617, 200 runs)
```

## License

MIT, see [LICENSE](LICENSE).
