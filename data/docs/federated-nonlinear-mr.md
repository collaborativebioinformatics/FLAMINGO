# Scenario 1: fitting the non-linear curve

The summary-statistic route and the concatenated route agree on the average
slope in every phenotype model (`federated-summary-mr.md`). This scenario asks
for more: the shape of the dose-response curve. Run from `data/`:

```bash
uv run python scripts/federated_nonlinear_mr.py --shape quadratic
uv run python scripts/federated_nonlinear_mr.py --shape threshold
```

## The two routes

**Concatenated.** Two-stage least squares with `X` and `X^2` as endogenous
regressors, instrumented by the SNP-predicted `X` and its square, with site
intercepts. Any function of the genotypes is a valid instrument, so this is a
proper IV fit rather than plugging the squared prediction into a regression.
It estimates `theta1` and `theta2` jointly, with a covariance that gives a
confidence band for the whole curve.

**Summary statistics.** Each site's per-SNP `beta_x` and `beta_y` are linear
projections of `X` and `Y` on genotype. They carry no information about the
curvature, so the best the coordinator can do is the IVW line: one slope
through the origin. `theta2` is not identifiable from these summaries.

## Results

| set | truth | concatenated quadratic 2SLS | sumstats linear IVW |
|---|---|---|---|
| quadratic | θ1 = 0.30, θ2 = 0.15 | θ1 0.317 (0.014), θ2 0.149 (0.031) | slope 0.318 (0.016) |
| threshold | slope 0.30 below 0.5, flat above | θ1 0.225 (0.014), θ2 -0.053 (0.031) | slope 0.224 (0.015) |

Standard errors in parentheses. Plots: `results/nonlinear.quadratic.png`,
`results/nonlinear.threshold.png`. Each shows the true curve, the pooled fit
with its 95% band, the sumstats line, and the distribution of `X` underneath.

**Quadratic set.** The pooled fit recovers `theta2` to within 0.001 and its
band tracks the true curve across the whole range. The sumstats line has the
right average slope but is wrong almost everywhere: it overstates the effect
below `X = 0` and understates it above, by 0.9 units at `X = 2.5`. A
two-standard-deviation increase in exposure causes roughly three times the
change that the sumstats route reports.

**Threshold set.** Both routes get the population-average slope, but the
pooled fit also detects the curvature: `theta2` is negative at 1.7 standard
errors and the fitted curve flattens above `X = 1`, which is the qualitative
shape of the truth. A quadratic is the wrong family for a kink, so it
overshoots below `X = -1.5`, but it says the effect saturates and the
sumstats line cannot. A piecewise or stratified IV fit on the pooled data
would locate the cutoff.

## What this shows

Federation by summary statistics loses nothing for a linear effect and loses
the shape entirely for a non-linear one. The quadratic set is the cleanest
demonstration, since the pooled fit is correctly specified and recovers both
parameters. For sumstats to compete, sites would have to share more than
per-SNP linear effects, for example per-SNP effects within strata of the
SNP-predicted exposure, which is what stratified non-linear MR methods
consume.
