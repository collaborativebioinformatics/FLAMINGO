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

A middle route, each site fitting the quadratic 2SLS locally and sharing its
two coefficients with their covariance, matches the concatenated fit: see the
"model sumstats" row of `results/sumstats.quadratic.png` and the curved-model
section of `federated-summary-mr.md`.

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

## Federated learning: naive and as MR

The plots carry two more dashed lines from `../federated_learning/`
(`job.py --dataset quadratic --method naive --method 2sri`), an NVFlare
FedAvg MLP trained across the ten sites without pooling individual data.

**Naive (green).** Fits `E[Y | X]` directly. It recovers the *shape* of the
association, quadratic curvature and the threshold kink alike, because a
flexible model sees individual-level data at every site. But it is the
confounded regression: `U` pushes both `X` and `Y`, so it sits above the
truth for large `X` on the threshold set and below it for negative `X` on
the quadratic set.

**2SRI (violet).** The same federation turned into a two-stage MR. Each site
fits its own first stage `X ~ SNPs` by OLS (the instruments are
site-specific, exactly as in the concatenated route), and the federated
second stage is `Y ~ f(X) + c (X - X_hat)`: `f` is a flexible curve head,
and the linear first-stage residual term is the control function that
carries the confounder. `f` then targets the causal curve. On the quadratic
set it lands on the 2SLS fit and the truth. On the threshold set it does
what the quadratic 2SLS cannot: it flattens above the cutoff and tracks the
plateau, because `f` is not restricted to a polynomial. The price is the
usual one for MR with instruments explaining ~9% of `X`: the correction is
noisy and least reliable in the tails, where few people and little
instrument variation sit.

The federated 2SRI is a proper individual-level MR that never moves
individual data between sites. What it needs beyond the summary-statistics
route is many rounds of model-weight exchange rather than one exchange of
per-SNP effects.

## What this shows

Federation by summary statistics loses nothing for a linear effect and loses
the shape entirely for a non-linear one. The quadratic set is the cleanest
demonstration, since the pooled fit is correctly specified and recovers both
parameters. For sumstats to compete, sites have to share more than per-SNP linear
effects. The cheapest option that works is a fitted model per site (its
quadratic coefficients and covariance), which the model-sumstats route shows
recovers `theta2` with the same precision as pooling. Per-SNP effects within
strata of the SNP-predicted exposure, the input of stratified non-linear MR
methods, are the other option.
