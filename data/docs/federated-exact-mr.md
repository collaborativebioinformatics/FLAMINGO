# FedMR: exact federated MR from sufficient statistics

The summary-statistics route (`federated-summary-mr.md`) recovers the
average slope but not the curve; the NVFlare FedAvg route
(`../../federated_learning/`) recovers a flexible curve but has no analytic
standard error and is an approximation. FedMR is a third federated route:
each site releases a few cross-product matrices, the coordinator sums them
and solves, and the result *is* the pooled two-stage least squares (2SLS)
fit, to machine precision, with its standard errors. Nothing is trained.

Code: the `flamingo_fedmr` package in `../../fedmr/` (numpy only), the
driver `scripts/federated_exact_mr.py`, the NVFlare transport
`../../federated_learning/fedmr_job.py`, the seed sweep
`scripts/fedmr_sweep.py`, and the tests `tests/test_fedmr.py`. Run from
`data/`:

```bash
uv run python scripts/federated_exact_mr.py --all          # every continuous set
uv run pytest -q                                           # identity tests
uv run python scripts/fedmr_sweep.py --seeds 100           # ~15 min
cd ../federated_learning && uv run python fedmr_job.py --all   # the real federation
```

## The estimator

With structural regressors `W = [X, C]` and instruments `Z = [G, C]` (`C`
covariates, which instrument themselves), pooled 2SLS is

```
theta = (B' A^-1 B)^-1 B' A^-1 c       A = Z'Z,  B = Z'W,  c = Z'Y
RSS   = f - 2 theta'e + theta' D theta   D = W'W,  e = W'Y,  f = Y'Y
Var   = RSS / (N - r - K) * (B' A^-1 B)^-1
```

Every matrix is a sum over rows, so a sum over sites: `A = sum_k A_k` and so
on. Site `k` computes its own `A_k, B_k, c_k, D_k, e_k, f_k, n_k` and sends
them; the coordinator adds and solves. `K` site intercepts are absorbed by
centring every column within its site before forming the cross-products,
so no dummy column is transmitted and the estimate is within site.

A second round gives the heteroskedasticity-robust (HC0) covariance: the
coordinator broadcasts `theta`, each site returns `H_k = Z' diag(u^2) Z`
with `u = Y - W theta`, and `Var = M^-1 B'A^-1 H A^-1 B M^-1` with
`M = B'A^-1 B`.

The first-stage partial F and partial R² of the excluded instruments come
from `A`, `B` and `D` alone (nested-model residual sums of squares), so they
need no extra round. With one endogenous regressor this is the usual
first-stage F. With two (the quadratic basis) it is a per-column partial F
and not a conditional (Sanderson-Windmeijer) F; the result flags it.

## Two protocols

| protocol | instruments | what leaves a site (round 1) | rounds |
|---|---|---|---|
| `SharedInstrumentFedMR` | the same harmonized SNPs at every site, one effect allele | centred `G'G` (m x m), `G'X`, `G'Y`, `X'X`, `X'Y`, `Y'Y`, `n` (plus covariate blocks) | 1, + 1 robust |
| `LocalFirstStageFedMR` | each site's SNPs are its own variants (this repo's default simulator) | the site fits `X ~ [1, G]` itself, forms `xhat`, and releases centred `xhat'xhat`, `xhat'X`, `xhat'Y`, `X'X`, `X'Y`, `Y'Y`, `n`: seven numbers | 1, + 1 robust |

The local-first-stage protocol reproduces `pooled_2sls` in
`scripts/federated_summary_mr.py` (per-site first stages, site intercepts),
which is the "concatenated" row of every forest plot. The shared protocol
reproduces one stacked 2SLS with the common `G` and site dummies; on the
`linear_shared` set that differs from the per-site-first-stage fit by
0.002, a reminder that the two are different estimators.

The generated-instrument (local first stage) form uses a *common* `xhat`
column at every site. Giving each site's `xhat` its own column would be a
larger instrument set: identical for one endogenous regressor, not for the
quadratic basis, so the common form is the one implemented.

**Quadratic basis.** `W = [X, X², C]`, `Z = [xhat, xhat², C]`, the form of
`quadratic_2sls`. With a local first stage it stays at one round. With
shared SNPs the first stage is global: round 1 sums the first-stage moments
(centred within site), the coordinator broadcasts `pi`, each site forms
`xhat = a_k + [G, C] pi` with its own intercept recovered locally (needed
because `(a_k + z pi)²` has a cross term centring does not absorb), and
round 2 sums the second-stage statistics.

**Cross-fitting.** The estimating equation is

```
sum_i  xhat_i^(-fold(i)) (Y_i - theta X_i) = 0
```

so the out-of-fold `xhat` is the *instrument* for `X`, not a substituted
regressor. Local folds: one round. Shared SNPs: per-fold first-stage
moments centred within site and fold, `pi_j` from `(A - A_j, b - b_j)`, the
fold's intercept from the site's other folds, then the second stage: two
rounds. Whether cross-fitting reduces the one-sample weak-instrument lean is
a hypothesis; the sweep below tests it.

## Identity checks

`tests/test_fedmr.py` (35 tests) compares both protocols against
independent stacked-data projections written out in the test, not against
the package's own matrix helpers: the full coefficient vector, classical
and HC0 covariance, residual variance, nested-regression first-stage F,
row-partition invariance across transport clients for one logical site,
site-order invariance, transport round-trip, identification errors, and the
cross-fit estimating equation. Tolerance 1e-10; observed differences are at
1e-16.

`scripts/federated_exact_mr.py --all` on the checked-in sets:

| set | protocol | FedMR θ1 (SE, robust SE) | concatenated θ1 (SE) | max diff |
|---|---|---|---|---|
| linear | local first stage | 0.3170 (0.0145, 0.0145) | 0.3170 (0.0145) | 3e-16 |
| quadratic | local first stage | 0.3182 (0.0148, 0.0149) | 0.3182 (0.0148) | 2e-16 |
| threshold | local first stage | 0.2241 (0.0145, 0.0145) | 0.2241 (0.0145) | 2e-16 |
| ushape | local first stage | 0.0182 (0.0148, 0.0149) | 0.0182 (0.0148) | 1e-16 |
| linear_shared | shared instruments | 0.3062 (0.0142, 0.0142) | 0.3062 (0.0142) | 0 |

Quadratic basis on the curved sets (θ1, θ2 with SEs): quadratic 0.3170
(0.0145), 0.1485 (0.0314); ushape 0.0170 (0.0145), 0.1485 (0.0314);
threshold 0.2246 (0.0145), -0.0526 (0.0314). Each equals `quadratic_2sls`
to 1e-15.

## The real federation

`../../federated_learning/fedmr_job.py` runs the protocol in the NVFlare
simulator: one client per site (`src/fedmr_client.py`), a server workflow
that sums and solves (`src/fedmr_controller.py`, a `ModelController` that
never averages), two rounds. After the job it re-runs the package
in-process on the same files and the pooled reference in numpy, and fails
if they disagree by more than 1e-10. On every continuous set the three
agree to 1e-16 in the estimate and the robust SE
(`../../federated_learning/results/fedmr/summary.csv`). The quadratic basis
runs the same way (`--basis quadratic`).

## What is, and is not, protected

This is distributed statistical estimation, not privacy-preserving
estimation. The coordinator sees each site's released matrices. For the
local-first-stage protocol that is seven scalars per site, comparable to
publishing the site's own 2SLS summary. For the shared protocol it is the
site's within-site LD matrix `G'G` and GWAS-level sums `G'X`, `G'Y`, which
is what a cohort releases in a GWAS meta-analysis, but not less. Secure
aggregation could hide per-site values only for cells to which several
sites contribute (the shared protocol's `A`); a cell with one contributor is
that site's number regardless of masking. A threat model, key exchange,
collusion threshold and dropout handling are all needed before any masking
scheme could be called privacy-preserving, and none is claimed here.

## Seed sweep

`scripts/fedmr_sweep.py` moves one design axis at a time away from the
default (10 sites of 500 to 5,000 people, 20 SNPs, `h2_x` 0.10, linear
θ = 0.3) and reports bias, RMSE, mean SE and 95% coverage over 100 seeds
for pooled 2SLS (= FedMR), cross-fitted FedMR, the meta-analysis of site
2SLS fits and per-SNP summary-statistics IVW. The estimand is the
n-weighted mean of the site causal effects, which is what a site-intercept
2SLS targets when effects differ. Results: `results/fedmr_sweep_summary.csv`
and `results/fedmr_sweep.png`; the sweep section below is filled from them.

SWEEP_RESULTS_PLACEHOLDER
