# Plan: FedMR, exact federated one-sample MR from sufficient statistics

Source idea: a ChatGPT design note (shared 2026-09-17) proposing that pooled
one-sample 2SLS MR can be computed *exactly* from additive cross-product
matrices, so a federation can reproduce the pooled estimate without moving
rows. This page turns that note into concrete work for this repo.

## The idea in one paragraph

At site k with structural regressors `W_k = [X_k, C_k]` and instruments
`Z_k = [G_k, C_k]`, the site computes `A_k = Z'Z`, `B_k = Z'W`, `c_k = Z'Y`,
`D_k = W'W`, `e_k = W'Y`, `f_k = Y'Y`, `n_k`. The coordinator sums them and
evaluates

```
theta = (B' A^-1 B)^-1 B' A^-1 c
RSS   = f - 2 theta'e + theta' D theta,   sigma2 = RSS / (N - dim theta)
Var   = sigma2 (B' A^-1 B)^-1
```

Because `Z'Z = sum_k Z_k'Z_k` and so on, this *is* the pooled 2SLS
estimator, not an approximation like FedAvg. One communication round gives
the point estimate, classical SE, first-stage effects, partial R² and F. A
second round (each site returns `H_k = Z' diag(u²) Z` after receiving theta)
gives heteroskedasticity-robust SEs. Cross-fitting (v2) addresses
weak-instrument bias while keeping everything local.

## Verified before planning

On the checked-in `federated/linear` set, assembling block-diagonal
per-site cross-products and evaluating the formula above reproduces
`pooled_2sls()` in `scripts/federated_summary_mr.py`:

```
FedMR  theta=0.316951131310 se=0.014457470143
pooled theta=0.316951131310 se=0.014457470143
|diff| theta=3.33e-16 se=1.73e-18
```

So the repo's existing "concatenated" benchmark is already the target the
federated estimator must hit, and the identity test is the unit test.

## How this fits the repo

The repo currently has three families: per-SNP summary statistics
(`federated_summary_mr.py`), NVFlare FedAvg on an MLP (`federated_learning/`),
and the concatenated pooled fit. FedMR is a fourth: a real federation (one
round through NVFlare) that matches the concatenated fit exactly, with
standard errors, which the FedAvg route lacks. It also closes the gap the
non-linear doc points out: sharing per-SNP effects loses `theta2`, sharing
sufficient statistics keeps it.

Two instrument layouts matter here:

| layout | when | `Z` structure | `A` |
|---|---|---|---|
| site-specific SNPs (current simulator) | every site draws its own SNPs | block-diagonal: `[1, G_k]` per site, site dummies in `W` | direct sum of `A_k` |
| shared, harmonized SNPs (new `--shared-snps`) | same variants at every site, allele-aligned | columns aligned across sites, `Z = [G, site dummies]`, site dummies also in `W` | dense sum of `A_k` |

The estimator code should take a column layout per site and handle both;
the first one reproduces today's `pooled_2sls` and `quadratic_2sls`, the
second is the setting the design note assumes (its "Round 0" harmonization).

## Work packages

### WP1. Estimator library (pure numpy, `data/scripts/fedmr.py`)

- `site_stats(Z, W, Y) -> SiteStats` with fields `A, B, c, D, e, f, n` and a
  column layout (names of Z and W columns) so the coordinator can align.
- `aggregate(list[SiteStats]) -> Stats`: sums aligned columns; site-specific
  instrument columns are placed in their own block; site dummies appended.
- `fit(Stats) -> FedMRResult`: theta, classical covariance, RSS, N, and
  first-stage diagnostics per endogenous column: `pi = A^-1 b_X`, partial R²,
  partial F (reduced model on the covariate submatrices of A and B).
- `robust_cov(Stats, H)` for the sandwich `M^-1 B'A^-1 H A^-1 B M^-1`, with
  `site_robust_stats(Z, W, Y, theta) -> H_k` as the round-2 site function.
- Covariate handling: covariates appear in both W and Z. `U` is an oracle
  column and must never be used; assert it is excluded.
- Non-linear basis: allow `W = [X, X², C]` with instruments
  `[xhat, xhat², C]` where `xhat` is site-local (matches `quadratic_2sls`).
  With shared SNPs the first stage is global, so the protocol becomes two
  rounds: round 1 sums `G'G, G'X`; round 2 sums the second-stage moments.
- Global standardization helper: sum `G`, `G²`, `N` first, then every site
  applies the same transform (the note's section 7); never standardize
  per site.
- Tests (`data/tests/test_fedmr.py`, pytest): identity against
  `pooled_2sls` and `quadratic_2sls` on every checked-in continuous set
  (`|diff| < 1e-10`); classical SE equality; robust SE equals an in-memory
  HC0 sandwich on stacked data; partial F equals a direct nested-model F;
  invariance to site order and to splitting one site into two.

### WP2. Driver and results (`data/scripts/federated_exact_mr.py`)

- Runs FedMR on `simulated_data/federated/<shape>` for `linear`, `quadratic`,
  `ushape`, `threshold` (continuous outcomes only in v1; Cox and binary have
  no additive sufficient statistics and stay with the existing routes).
- Prints a four-family table per set: pooled 2SLS, FedMR (classical and
  robust SE), meta-analysis of site 2SLS, per-SNP sumstats IVW, plus
  `|FedMR - pooled|`, first-stage F and partial R².
- Writes `results/fedmr.<shape>.csv` and adds a `fedmr` row to the forest
  plots in `federated_summary_mr.py` (a filled marker with a CI on the
  federated row, replacing the "no CI" caveat) and a curve with a band on
  `federated_nonlinear_mr.py`.

### WP3. Simulator option: shared SNPs and heterogeneity knobs

- `simulate_federated_sites.py --shared-snps`: draw MAF and `beta` once from
  the base seed; each site draws its own individuals. Option
  `--maf-shift sd` perturbs MAFs per site (logit scale) to model allele
  frequency differences; `--theta-sd` gives site-specific causal effects for
  the heterogeneity scenario; `--pleiotropy-sd` adds direct `G -> Y` effects.
- Manifest records `shared_snps`, the shared `beta`, and per-site MAFs.
- Check in one `federated/linear_shared` set so the dense-A path has data.

### WP4. Federation through NVFlare (`federated_learning/`)

- New job `fedmr_job.py` with a client script that computes `SiteStats` and
  sends them once as an `FLModel` (arrays keyed by name), and a small
  server-side controller that *sums* rather than averages and then calls
  `fedmr.fit`. Round 2 broadcasts theta and collects `H_k`. Reuse the
  existing `simulator_run` helper and `--engine local` pattern for a fast
  in-process version that runs the same functions.
- Assert at the end of the job that the NVFlare result equals the in-process
  result and the pooled result to `1e-10`; print that line as the headline.
- Secure aggregation: v1 sends `A_k` in the clear. Add an additive-masking
  filter (pairwise seeded masks that cancel in the sum) so the server sees
  only the aggregate; document it as "privacy-preserving FedMR" versus
  "statistical FedMR". NVFlare's homomorphic-encryption filters are the
  stretch goal, not v1.
- Write `results/fedmr/<dataset>/estimates.json` in the same layout the
  data scripts read for their plots.

### WP5. Cross-fitted FedMR (v2, `fedmr.py` and the driver)

- Each site splits into `k` folds; for fold j it fits the first stage on the
  other folds (locally, or via summed moments minus the fold's moments in the
  shared-SNP case) and predicts `xhat` for fold j. The second stage then
  uses only out-of-fold `xhat`, so the one-sample weak-instrument lean seen
  in `federated-summary-mr.md` (about one SE above truth at F 9 to 21)
  should shrink.
- The shared-SNP version is the interesting one: out-of-fold moments are
  `A - A_fold`, so it still needs no rows to move, only one extra round.
- Compare bias across seeds against pooled 2SLS, FedMR and sumstats.

### WP6. Benchmark sweep (`data/scripts/fedmr_sweep.py`)

Follow `sweep_seeds.py` and `rare_events_sweep.py`. Report bias, RMSE and
coverage per estimator over 200 seeds for these axes, one at a time from the
default ten-site setting:

| axis | values |
|---|---|
| sites | 2, 5, 10, 20 |
| instrument strength | mean `h2_x` 0.02, 0.05, 0.10, 0.20 |
| SNP count | 5, 20, 100 |
| size imbalance | equal, current spread, 90/10 |
| allele frequencies | shared, shifted |
| causal effect | shared, site-specific `theta` |
| pleiotropy | none, balanced, directional |

The first row of every table is the identity check `max |FedMR - pooled|`.
Expect FedMR and pooled to differ from the meta-analysis of site 2SLS mainly
under imbalance and small sites, and from sumstats IVW under non-linearity.

### WP7. Docs and manuscript

- `data/docs/federated-exact-mr.md`: protocol, what leaves a site (a table
  of matrix sizes: with 20 SNPs and 10 sites the block-diagonal `A` has
  ten 21x21 blocks), the identity result, the sweep results.
- README: add step 4b to the quick start and a fourth family to the pipeline
  figure text; `writing/methods.md` and `results.md` get a FedMR section.
- Update `federated-summary-mr.md` and `federated-nonlinear-mr.md` where they
  say the federated row has no CI.

## Order and effort

1. WP1 with tests (half a day; the algebra is already verified).
2. WP2 driver and plot rows (half a day).
3. WP4 NVFlare job with the in-process engine first, NVFlare second (one day).
4. WP3 shared-SNP simulator, then rerun WP2 and WP4 on it (half a day).
5. WP5 cross-fitting and WP6 sweep (one to two days).
6. WP7 as each piece lands.

## Decisions taken in this plan

- Continuous outcomes only in v1. Logistic and Cox IV models have no
  additive sufficient statistics; they would need iterative federated
  Newton steps (still exact, but a different protocol) and belong to v3.
- Site intercepts always in `W` and `Z`, so the estimate is within-site, as
  the note recommends.
- Keep the FedAvg MLP route; it remains the only route that fits an
  unspecified curve shape (threshold). FedMR fits a chosen basis exactly.
- Dependencies: numpy and polars only in `data/`; pytest added as a dev
  dependency. No new runtime packages in `federated_learning/`.
