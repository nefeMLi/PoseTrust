# PoseTrust

[![tests](https://github.com/nefeMLi/PoseTrust/actions/workflows/tests.yml/badge.svg)](https://github.com/nefeMLi/PoseTrust/actions/workflows/tests.yml)

Pose-graph SLAM returns a trajectory and a covariance: the solver's own estimate of how far off that trajectory
might be. I wanted to know whether that estimate can be trusted, and when it can't, whether it errs on the
cautious side or the dangerous one.

Everything is simulated, so the true trajectory is known. Each graph is solved 200 times under fresh measurement
noise, and the actual errors are compared with the reported covariance using NEES (normalised estimation error
squared). If the covariance is honest, NEES per degree of freedom averages 1; above 1 the solver is overconfident.

What I found:

- With small rotational noise the covariance is honest. From about 0.06–0.1 rad of heading noise per measurement it
  becomes overconfident, and quickly: at 0.22 rad only 1.5% of SE(3)'s 95% confidence ellipsoids contain the truth.
- With false loop closures, the Huber kernel keeps the trajectory reasonable but reports far too little uncertainty,
  on all eight graphs I tried. Dynamic covariance scaling (DCS) stays honest on six of the eight.
- Huber's problem is bias. It down-weights a false closure but never ignores it, so every false closure keeps
  pulling the estimate the same way, and no covariance can describe a shift that is the same in every run. Dropping
  the closures Huber flags and re-solving gives an honest covariance on all eight graphs.

![Huber's excess is mostly bias, and removing the flagged closures fixes it](figures/huber_bias.svg)

## Results

| Experiment | Result |
|---|---|
| Rotational noise | Calibration breaks from 0.06 rad in SE(3) and 0.10 rad in SE(2) on the first graph, and between 0.06 and 0.22 rad across eight loop-closure layouts. No layout is ever conservative. |
| False loop closures | From 10% false closures up, Huber is overconfident at every rate on 8 of 8 layouts (NEES/dof up to 15.6), while keeping the trajectory error at 30% outliers to 0.10–0.22 against 0.6–1.7 for plain least squares where it converges. DCS is honest on 6 of 8; Cauchy and GNC are mostly slightly cautious. |
| Why Huber is overconfident | The bias accounts for 91–98% of Huber's excess NEES, pooled over layouts. Re-solving without the flagged closures brings NEES/dof to 1.00–1.01 (median) and within 10% of calibrated on 8 of 8 layouts. |

![Calibration of each robust back-end on eight layouts](figures/false_loop_closures.svg)

![Where calibration breaks depends on the layout](figures/rotational_noise.svg)

## What I expected, and what happened

- I expected sparse graphs to be badly calibrated. Loop-closure density barely mattered (a few percent), so that
  experiment isn't included.
- I expected the robust kernels to fix the trajectory but not the covariance. That holds for Huber; the kernels that
  give gross outliers almost no weight (DCS, Cauchy, GNC) mostly keep the covariance honest or slightly cautious.
- I expected a better covariance formula, such as the sandwich estimator from robust statistics, to fix Huber. None
  of the ones I tried did, which is what pointed to bias.
- Adding an estimate of the bias to Huber's covariance removed most of the excess but not all of it. Re-solving
  without the flagged closures works better, so that is what is reported here.

## Limitations

- Simulation only, on graphs of at most 20 poses, and a false closure is always "the right measurement to the wrong
  pose". Real outliers can look different.
- The robust-kernel results use one trajectory with eight loop-closure layouts.
- The refit uses a fixed threshold (the 99.9% chi-squared quantile) to flag closures.

## Background

Linearised SLAM estimators turning overconfident is known from EKF-SLAM (Bailey et al., IROS 2006; Huang, Mourikis
and Roumeliotis, IJRR 2010); the rotational-noise experiment is the pose-graph version of that. The robust back-ends
are Huber, Cauchy, dynamic covariance scaling (Agarwal et al., ICRA 2013) and graduated non-convexity (Yang et al.,
RA-L 2020). They were designed and evaluated for the trajectory they recover; here the question is the covariance.

## Running

Python 3.14:

```sh
pip install -r requirements.txt
python -m experiments.rotational_noise      # about six minutes on twelve cores
python -m experiments.false_loop_closures   # a few hours; add --huber-only for the Huber part (~20 min)
pytest tests.py
```

Each script writes results to `results/` and figures to `figures/`; `--figures-only` redraws them from the saved
results. `tests.py` checks the Jacobians against finite differences, checks that the covariance is honest where it
should be, and (on GitHub Actions, where GTSAM installs) that estimates and covariances agree with GTSAM.

```
posetrust/     SE(2) and SE(3), the pose graph, Gauss-Newton and Levenberg-Marquardt,
               robust kernels, covariance extraction, the Monte Carlo harness, NEES statistics
experiments/   the two experiments, and common.py for shared statistics and plotting
```

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

MIT. See [LICENSE](LICENSE).
