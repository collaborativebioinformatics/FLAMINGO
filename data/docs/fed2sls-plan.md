# Plan: Fed-2SLS, exact federated one-sample MR from sufficient statistics

Source idea: a ChatGPT design note (shared 2026-09-17) proposing that pooled
one-sample 2SLS MR can be computed *exactly* from additive cross-product
matrices, so a federation reproduces the pooled estimate without moving
rows. This page is the working plan, revised on 2026-09-18 after review.
Implementation status is tracked at the end.

## The idea in one paragraph

With structural regressors `W = [X, C]` and instruments `Z = [G, C]`, pooled
2SLS is

```
theta = (B' A^-1 B)^-1 B' A^-1 c,     A = Z'Z,  B = Z'W,  c = Z'Y
RSS   = f - 2 theta'e + theta' D theta,   D = W'W, e = W'Y, f = Y'Y
Var   = RSS / (N - r - absorbed) * (B' A^-1 B)^-1
```

Every matrix is a sum over rows, so a sum over sites. Each site sends its
sums; the coordinator adds them and solves. One round gives the estimate,
classical SE, first-stage F and partial R². A second round (each site returns
`H_k = Z' diag(u²) Z` after receiving theta) gives the HC0 robust covariance.
This is *distributed statistical estimation*: the result is the pooled
estimator, not an approximation of it.

## Verified

On every checked-in continuous set, both protocols below reproduce the
repo's pooled fits (`pooled_2sls`, `quadratic_2sls`) to better than 1e-15
in the estimate and the SE; `data/tests/test_fed2sls.py` holds the checks
against independent stacked-data projections (full coefficient vector,
classical and HC0 covariance, residual variance, first-stage diagnostics).

## Two protocols, not one aggregation

The review made the point that shared and site-specific instruments are
different protocols, not two column layouts. The package exposes them as
such (`flamingo_fedmr.protocols`), and both absorb site intercepts by
within-site centring so no dummy column is ever transmitted:

| protocol | when | second stage | rounds |
|---|---|---|---|
| `SharedInstrumentFedMR` | harmonized SNPs, one effect allele everywhere | `Z = [G, C]`, `W = [X, C]`; every site contributes to every cell of `A` | 1 (+1 robust) |
| `LocalFirstStageFedMR` | each site's SNPs are its own variants (the repo's default simulator) | local first stage `X ~ [1, G, C]`, then `Z = [xhat, C]`, `W = [X, C]`: a generated instrument in a common `(1+q)`-column matrix | 1 (+1 robust) |

The local-first-stage protocol equals pooled 2SLS with per-site first stages
and site intercepts. Giving each site's xhat its own column instead would be
a larger instrument set; identical for one endogenous regressor, not for the
quadratic basis, so the common-column form is the one implemented.

**Quadratic basis** (`quadratic_2sls`'s generated-instrument form): `W = [X,
X², C]`, `Z = [xhat, xhat², C]`. Local first stage: still 1 round. Shared
SNPs: the first stage is global, so round 1 sums the first-stage moments
(centred within site), the coordinator broadcasts `pi`, each site forms
`xhat = a_k + [G, C] pi` with its own intercept `a_k` recovered locally
(needed because `(a_k + z pi)²` has a term centring does not absorb), and
round 2 sums the second-stage statistics: 2 rounds (+1 robust).

**Cross-fitting** (v2). The estimating equation is

```
sum_i  xhat_i^(-fold(i)) (Y_i - theta X_i) = 0
```

with covariates partialled the same way: the out-of-fold xhat is the
*instrument* for X, not a substituted regressor. Local folds: 1 round.
Shared SNPs: per-fold first-stage moments (centred within site and fold),
`pi_j` from `(A - A_j, b - b_j)`, the fold's intercept from the site's other
folds, then the second stage: 2 rounds (+1 robust). Whether this reduces the
one-sample weak-instrument lean is a hypothesis the sweep tests.

## What this is and is not (privacy)

v1 is distributed statistical estimation. Secure aggregation would help only
for cells to which several sites contribute (the shared-instrument `A`);
a cell with one contributor is that site's own number. Site-level
disclosures in the current design: the local-first-stage protocol releases
per site the centred cross-products of `xhat`, `X`, `Y` (a handful of
scalars per site, comparable to a site's own 2SLS summary), and the
shared-instrument protocol releases the site's centred `G'G`, `G'X`, `G'Y`
(a within-site LD matrix and GWAS-level sums). A threat model, key exchange,
collusion threshold and dropout handling would all be needed before any
masking scheme could be called privacy-preserving. None of that is claimed
here; the plots and docs say "federated: Fed-2SLS", not "private".

## Numerical API

No explicit inverse of `A`: every `A^-1(.)` is a linear solve; `M = B'A^-1B`
is `r x r` and its inverse is solved against the identity because the
covariance needs it. The result reports rank and condition number of `A`,
condition number of `M`, the counts of endogenous, excluded-instrument and
exogenous columns and absorbed fixed effects, and raises
`IdentificationError` with the reason for under-identification, collinear
instruments or no residual degrees of freedom. Columns carry roles
(endogenous, exogenous, instrument) in the schema; names alone are not used
for diagnostics. First-stage F and partial R² are per endogenous column;
with more than one endogenous regressor the result flags them as not
conditional (not Sanderson-Windmeijer).

The oracle column `U` is rejected in the CSV loader, not the estimator.

## Code location

`fedmr/` at the repo root is a small package, `flamingo_fedmr` (numpy only):
`schema`, `statistics`, `estimator`, `protocols`, `data`. Both `data/` and
`federated_learning/` depend on it as an editable path dependency, so the
NVFlare client and the analysis scripts run the same arithmetic.

## Work packages and status

1. **Linear statistical core** (`fedmr/`): done. Schema with roles, site
   statistics, aggregation, classical and HC0 covariance, rank and
   conditioning checks, transport helpers.
2. **Independent equivalence tests** (`data/tests/test_fed2sls.py`): done.
   Stacked-data references, full vector and covariance equality, residual
   variance, nested-regression F, row-partition invariance across transport
   clients for one logical site, site-order invariance, transport
   round-trip, identification errors, the cross-fit estimating equation.
3. **Local-first-stage adapter**: done, equals `pooled_2sls` and
   `quadratic_2sls`.
4. **Driver and outputs** (`scripts/federated_exact_mr.py`): done. Four-way
   comparison per set, `results/fed2sls.<shape>.csv`; the forest plots keep
   the NVFlare 2SRI row (`federated: Fed-2SRI`, flexible curve, no analytic CI)
   and add `federated: Fed-2SLS` (specified basis, analytic CI); the
   dose-response plot draws the Fed-2SLS quadratic curve over the concatenated
   one.
5. **Shared-SNP simulator**: done in `simulate_basic._draw_exposure`
   (given `maf`, `beta`, realized `h2_x`), both federated generators
   (`--shared-snps`, `--maf-shift`, `--theta-sd`, `--pleiotropy-mean/sd`),
   manifest records `shared_snps`, shared MAF/beta, effect-allele note and
   per-site realized `h2_x`; `federated/linear_shared` checked in.
6. **NVFlare transport** (`--method fed2sls` of `federated_learning/job.py`,
   `src/fed2sls_engine.py`; formerly a separate `fed2sls_job.py`): transport the
   tested statistics through a one-round (plus robust round) controller
   that sums, and assert equality with the in-process run. No secure
   aggregation claims.
7. **Nonlinear generated-instrument protocol**: done for the quadratic
   basis, both protocols, tested against `quadratic_2sls` and a stacked
   reference.
8. **Cross-fitting and sweeps** (`scripts/fed2sls_sweep.py`): done, 100
   seeds per level; results in `federated-exact-mr.md`. Under effect
   heterogeneity the estimand is the pooled 2SLS limit,
   `theta* = (B'A^-1 sum_k B_k theta_k) / (B'A^-1 B)`, a first-stage-weighted
   mean of the site effects computed per replicate from the realised
   first-stage matrices; it is *not* the n-weighted mean (an earlier draft
   of this plan said so and was wrong).
9. **Docs and manuscript**: done (`data/docs/federated-exact-mr.md`,
   README quick-start steps, both project READMEs, `writing/methods.md` and
   `results.md`, the two existing federated docs).

## Review follow-ups (2026-09-18, second round)

- First-stage F under the local-first-stage protocol is now the F of the
  original SNP set: each site also releases the residual sums of squares of
  its own first stage (additive across sites), and the single-column
  generated-instrument F is kept separately as `generated_instrument_F`.
- The NVFlare job no longer crashes when every requested dataset is skipped.
- `_finish` returns a `Run`; `SiteStats.from_transport` uses keyword
  construction; the controller serialises through `FedMRResult.to_dict`;
  `_union` is a dict. Heterogeneity lives in `scripts/heterogeneity.py`;
  the generalized orchestrator takes one simulator closure per site; the
  driver's rows are a keyword-only dataclass; the oracle check is in the
  loader; compiled caches are ignored and removed.

## Decisions

- Continuous outcomes only. Logistic and Cox IV models have no additive
  sufficient statistics; iterative federated Newton steps are a later
  protocol.
- Site intercepts always absorbed (within-site centring), so every estimate
  is within site.
- Keep the FedAvg MLP route as its own family; it is the only one that fits
  an unspecified curve shape.
