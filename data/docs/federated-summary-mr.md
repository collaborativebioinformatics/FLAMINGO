# Federated MR from per-site GWAS summary statistics

`scripts/federated_summary_mr.py` analyses a ten-site federated set without
pooling individual-level data. There is one set per phenotype model under
`simulated_data/federated/<shape>/`, all drawn with the same site sizes,
heritabilities and confounding. Run from `data/`:

```bash
uv run python scripts/simulate_federated_sites.py --shape cox        # regenerate a set
uv run python scripts/federated_summary_mr.py --shape cox            # analyse it
```

`--shape` is one of `linear`, `quadratic`, `threshold`, `cox`. Outputs go to
`results/sumstats.<shape>.{csv,png}`.

## What each site shares

Each site runs a GWAS of X and a GWAS of Y on its own SNPs, using the same
individuals for both (one-sample MR). It shares one row per SNP:

| column | meaning |
|---|---|
| `beta_x`, `se_x` | per-SNP effect on the exposure and its standard error |
| `beta_y`, `se_y` | per-SNP effect on the outcome and its standard error |
| `n` | site sample size |

Written to `simulated_data/federated/<shape>/sumstats/<site>.sumstats.csv`.
Individual genotypes, X, Y and U never leave the site. For the Cox model the
GWAS of Y is a per-SNP Cox regression, so `beta_y` is a log hazard ratio per
allele and every downstream estimate is on the log-HR scale.

## What the coordinator computes

1. **Per-site IVW.** The inverse-variance weighted slope of `beta_y` on
   `beta_x` through the origin, with weights `1 / se_y^2`. This is the
   standard summary-statistic MR estimator.
2. **Meta-analysis.** Inverse-variance combination of the ten per-site IVW
   estimates, with Cochran's Q for heterogeneity.

The SNPs are drawn independently at each site, so `snp0` at one site is a
different variant from `snp0` at another. Per-SNP effects are therefore never
meta-analysed across sites; only the per-site causal estimates are. As a
cross-check, an IVW fit over all 200 site-SNP pairs pooled together gives the
same answer as the meta-analysis, since every pair's ratio estimates the same
slope.

## Results across the four models

Every set uses `theta1 = 0.3`; quadratic has `theta2 = 0.15`, threshold has
cutoff `theta2 = 0.5`. The target is what a linear estimator can recover:
`theta1` for linear and Cox, the n-weighted mean of the sites' average slopes
for the non-linear curves.

| model | target | sumstats (meta of site IVW) | concatenated (one pooled fit) | pooled naive | Q (9 df) |
|---|---|---|---|---|---|
| linear | 0.300 | 0.317 (0.015) | 0.317 (0.014) | 0.391 | 8.8 |
| quadratic | 0.299 | 0.318 (0.016) | 0.318 (0.015) | 0.388 | 9.7 |
| threshold | 0.207 | 0.224 (0.015) | 0.224 (0.015) | 0.299 | 9.1 |
| cox (log HR) | 0.300 | 0.291 (0.018) | 0.293 (0.018) | 0.380 | 6.0 |

Standard errors in parentheses. The pooled fit is 2SLS with per-site first
stages and site intercepts for the three continuous outcomes, and a two-stage
predictor-substitution Cox model stratified by site for the survival outcome.

In all four models the summary-statistic route matches the concatenated fit
to within 0.002, with a standard error at most 0.001 wider. The three
continuous models sit about one standard error above their target, in the
direction of the confounded naive estimate, which is the weak-instrument lean
of one-sample IVW (mean F is between 9 and 21 at seven sites). The Cox model
sits slightly below, which is the non-collapsibility shrinkage seen in the
single-site survival check. Heterogeneity Q is at or below its 9 df in every
case, as expected when every site shares one curve.

### Per-site detail, quadratic set

True `theta1` is 0.30 at every site.

| site | n | mean F | naive OLS | IVW | IVW se |
|---|---|---|---|---|---|
| site01 | 1,426 | 9 | 0.423 | 0.303 | 0.086 |
| site02 | 2,361 | 13 | 0.430 | 0.229 | 0.073 |
| site03 | 3,480 | 14 | 0.437 | 0.384 | 0.072 |
| site04 | 4,556 | 21 | 0.418 | 0.372 | 0.056 |
| site05 | 4,631 | 16 | 0.439 | 0.382 | 0.063 |
| site06 | 5,629 | 12 | 0.388 | 0.277 | 0.076 |
| site07 | 7,141 | 49 | 0.387 | 0.344 | 0.036 |
| site08 | 8,154 | 79 | 0.398 | 0.308 | 0.029 |
| site09 | 8,424 | 17 | 0.335 | 0.385 | 0.061 |
| site10 | 9,380 | 37 | 0.359 | 0.247 | 0.040 |
| **sumstats** (meta of site IVW) | 55,182 | | | **0.318** | **0.016** |
| **concatenated** (one pooled 2SLS) | 55,182 | | 0.388 | **0.318** | **0.015** |

Meta-analysis 95% CI [0.288, 0.349]; pooled 2SLS 95% CI [0.289, 0.347]. Heterogeneity Q = 9.7 on 9 df, so the
sites are consistent with one shared effect, which is how they were built.

Every site's naive OLS sits above 0.30 because of the confounder, and every
site's IVW interval covers 0.30. The combined estimate is 0.02 above the truth,
about one standard error. With mean F between 9 and 21 at seven of the ten
sites, one-sample IVW is expected to lean slightly toward the confounded OLS
value; that weak-instrument bias is the next thing to test, either by
increasing site sizes or by splitting each site into exposure and outcome
halves for a two-sample design.

## The three analysis families on every plot

Each `results/sumstats.<shape>.png` ends with one row per family below the
site rows, coloured by family:

| family | row(s) | what leaves each site | estimate |
|---|---|---|---|
| sumstats (orange) | `sumstats: per-SNP` | per-SNP GWAS effects | inverse-variance meta-analysis, with CI |
| federated: Fed-2SRI (green) | `federated: Fed-2SRI` | model weights each round, via NVFlare FedAvg | the last-round global 2SRI model from `../federated_learning/results/2sri/<dataset>/curves.csv`, summarised into the plot's parameters by least squares on the curve over -2 <= X <= 2; no analytic CI |
| federated: Fed-2SLS (violet) | `federated: Fed-2SLS` | centred cross-product matrices, one round (plus one for the robust SE) | exact federated 2SLS from `results/fed2sls.<dataset>.csv` (`scripts/federated_exact_mr.py`), identical to the concatenated row, with CI; see `federated-exact-mr.md` |
| concatenated (black) | `concatenated` | individual rows | one 2SLS (or stratified 2SPS Cox), with CI |

Grey ticks on any row are the corresponding fit without instruments: naive
OLS or Cox at a site, the pooled naive fit, and the NVFlare `naive` model on
the federated row. The federated row is read from the NVFlare results at
plot time, so it reflects whatever run is on disk; the script says so when
no run exists.

Federated 2SRI values as read from the runs on disk at the time of writing:

| dataset | federated 2SRI | federated naive | truth or target |
|---|---|---|---|
| linear | 0.213 | 0.359 | 0.30 |
| cox (log HR) | 0.504 | 0.374 | 0.30 |
| cox_rare (log HR) | 0.500 | 0.432 | 0.30 |
| quadratic (θ1, θ2) | 0.279, 0.143 | 0.366, 0.150 | 0.30, 0.15 |
| ushape (θ1, θ2) | 0.025, 0.130 | 0.087, 0.132 | 0, 0.15 |
| threshold (θ1, θ2) | 0.148, -0.059 | 0.276, -0.065 | avg slope 0.21 |

The federated MLP recovers the curvature of the quadratic sets well and
removes most of the confounding, but its slopes sit further from the truth
than either the sumstats or concatenated estimates, and on the survival sets
it overshoots the log hazard ratio. Without a standard error it is not
possible to say from one run how much of that is noise; a seed sweep of the
NVFlare job would be the way to find out.

The `federated: Fed-2SLS` row is the other federated route: the same
individual-level 2SLS as the concatenated row, computed from per-site
cross-product matrices and therefore identical to it, with a confidence
interval. On every continuous set the two rows coincide to 1e-16
(`results/fed2sls.<shape>.csv`). Fed-2SLS needs a specified basis, so it does
not replace the Fed-2SRI curve; the two rows answer different questions.

## Curved models: two parameters

For `quadratic` and `threshold` the forest plot has two columns, `theta1`
(slope at X = 0) and `theta2` (curvature in a quadratic basis). The
federation levels it compares:

| level | what leaves each site | can estimate |
|---|---|---|
| per-SNP sumstats | per-SNP `beta_x`, `beta_y` and standard errors | average slope only; `theta2` is not identifiable |
| federated: Fed-2SRI | model weights each round | both, read off the fitted curve, no CI |
| federated: Fed-2SLS | centred cross-product matrices | both, exactly the concatenated fit, with CI |
| concatenated | individual rows | both, in one quadratic 2SLS with site intercepts |

Per-site rows show the site's own local quadratic 2SLS (filled, both
columns) and its per-SNP IVW slope (hollow, first column only). Per-site
`theta2` intervals are wide, since the squared prediction is a weak
instrument at F around 10. Per-SNP summary statistics cannot produce
`theta2` at all, which the plot marks in the second column; the federated
rows recover it. (An earlier version also meta-analysed the sites' own
quadratic fits as a "model sumstats" row; it was dropped because Fed-2SLS
gives that pooled answer exactly.)

## Summary statistics versus concatenated

The last row concatenates all ten sites' individual-level data and fits one
2SLS. Because the SNPs differ by site, the first stage is fitted per site
(each site's SNPs instrument only its own people) and the second stage has
site intercepts. Naively stacking the CSVs and treating `snp0` as one variant
across sites would be wrong.

The sumstats estimate matches the pooled one to three decimals, with a
standard error 0.001 wider. Nothing is lost by sharing summary statistics
instead of data in this linear, homogeneous-effect setting. That equivalence
is expected: IVW on summary statistics is algebraically the same estimator as
2SLS with independent instruments, and inverse-variance meta-analysis of
per-site slopes is the same as one weighted regression over all site-SNP
pairs. It would break under effect heterogeneity across sites, or for the
non-linear curve, where per-site summary statistics carry no information
about `theta2`.

The forest plots are `results/sumstats.<shape>.png` and the per-site tables
`results/sumstats.<shape>.csv`.
