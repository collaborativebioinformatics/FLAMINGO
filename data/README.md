# Simulated MR data (team10-data)

Simulated data for testing Mendelian randomization (MR) methods.

This repo collects the design notes, the survey of existing simulators, and
the parameter files used to generate benchmark datasets.

## Contents

| Path | What |
|---|---|
| `docs/simulation-models.md` | Every model in one place: exposure model, causal curves, outcome types, generators, checked-in datasets |
| `docs/mr-simulation-model.md` | The data-generating model and the knobs an MR benchmark should vary |
| `docs/package-survey.md` | Existing simulators (simmrd, simulateGP, GWASBrewer) and the within-family gap |
| `docs/basic-simulator-parameters.md` | Parameter table for the Python simulator, fixed quantities, outputs |
| `docs/federated-summary-mr.md` | Federated MR: per-site GWAS summary statistics, per-site IVW, meta-analysis |
| `docs/federated-nonlinear-mr.md` | Scenario 1: pooled quadratic 2SLS recovers the curve, summary statistics only its average slope |
| `docs/federated-rare-events.md` | Scenario 2, negative result: few events per site does not separate sumstats from pooled Cox |
| `docs/federated-exact-mr.md` | Fed-2SLS: exact federated 2SLS from summed sufficient statistics; protocols, identity checks, NVFlare run, seed sweep |
| `docs/fed2sls-plan.md` | The Fed-2SLS plan as revised after review, with status per work package |
| `docs/fed2sls-branch-notes.html` | Self-contained page explaining what the `fed2sls` branch adds: background, variables and shapes, protocols, verification, sweep, review history, references |
| `docs/simmrd-review.md` | Source-level review of simmrd: API, outputs, limits |
| `scripts/simulate_basic.py` | Pure-Python generator: linear, quadratic, threshold, or Cox survival outcome; no pleiotropy, no LD |
| `scripts/simulate_binary.py` | Same exposure model with a 0/1 outcome from a liability or logistic link at a target prevalence |
| `scripts/simulators.py`, `scripts/simulate_federated.py` | Registry of outcome simulators and the generic ten-site orchestrator (`--outcome continuous|binary|survival`) |
| `scripts/check_survival.py` | Naive, 2SPS, 2SRI and oracle Cox fits against the true log hazard ratio |
| `scripts/federated_summary_mr.py` | Per model: GWAS summary stats per site, IVW MR per site, meta-analysis vs the NVFlare federated model vs one pooled fit on concatenated data, forest plot; for curved models a two-column plot with each site's own quadratic fit |
| `scripts/federated_nonlinear_mr.py` | Quadratic 2SLS on concatenated sites vs the sumstats line; dose-response plot |
| `scripts/rare_events_sweep.py` | Seed sweep of the rare-event Cox setting: bias, RMSE and coverage for both routes |
| `scripts/simulate_federated_sites.py` | Ten sites sharing one causal curve (`--shape`), per-site heritability/confounding sampled from a distribution; `--shared-snps`, `--maf-shift`, `--theta-sd`, `--pleiotropy-*` for heterogeneity scenarios |
| `scripts/federated_exact_mr.py` | Fed-2SLS on every continuous set: exact federated 2SLS (classical and robust SE, first-stage F, cross-fitted variant) against the concatenated, site-meta and sumstats estimators; `results/fed2sls.<shape>.csv` |
| `scripts/fed2sls_sweep.py` | Seed sweep of Fed-2SLS, cross-fitted Fed-2SLS, site meta-analysis and sumstats IVW over sites, instrument strength, SNP count, imbalance, MAF shift, effect heterogeneity and pleiotropy |
| `tests/test_fed2sls.py` | Fed-2SLS identity tests against stacked-data references and the pooled fits (`uv run pytest -q`) |
| `simulated_data/single/` | Single-site draws: `basic`, `basic_seed2`, `quadratic`, `threshold`, `cox` as `{csv,truth.json}` |
| `simulated_data/federated/<shape>/` | Ten-site federated draws, one per phenotype model plus `cox_rare` (small sites, short follow-up), `ushape` (θ1 = 0, zero average slope), `binary_quadratic_logistic` (0/1 outcome, no summary-statistics analysis yet) and `linear_shared` (the same 20 SNPs at every site, for the shared-instrument Fed-2SLS protocol): `site01..site10.{csv,truth.json}`, `manifest.{csv,json}`, and `sumstats/` from the analysis |
| `simmrd/params/*.yaml` | Parameter files for the simmrd CLI, one per scenario |
| `simmrd/README.md` | How to run the simmrd CLI against these parameter files |

## Quick start (Python, uv)

```bash
uv run python scripts/simulate_basic.py            # defaults: n=10000, 20 SNPs, theta=0.3
uv run python scripts/simulate_basic.py --help     # all knobs
```

Writes `simulated_data/single/basic.csv` (id, snp0..snpJ, U, X, Y) and `simulated_data/single/basic.truth.json`
(theta, MAFs, per-SNP betas, confounder strengths, seed). Both are checked in;
the CSV can be regenerated from the seed in the truth file. Sanity check on the
default seed: naive OLS 0.39, 2SLS through the SNPs 0.29, true theta 0.30.

```bash
uv run python scripts/simulate_federated_sites.py --shape quadratic  # 10 sites, one shared curve
uv run python scripts/federated_summary_mr.py --shape quadratic      # sumstats MR vs pooled fit
```

`--shape` is `linear`, `quadratic`, `threshold` or `cox`; all four sets are
checked in. Writes `simulated_data/federated/<shape>/site01..site10.{csv,truth.json}`
plus a `manifest.{csv,json}`. Population size (1,000-10,000), SNP heritability
`h2_x`, and confounder strengths `gamma_x`/`gamma_y` are drawn per site from
distributions meant to mimic real variation between countries; `theta1`,
`theta2` (the causal curve) are the same at every site. See
`docs/basic-simulator-parameters.md` for the exact distributions.

## simmrd scenarios (R, optional)

```bash
git clone https://github.com/noahlorinczcomi/simmrd
cd simmrd/cli
pixi install && pixi run setup
pixi run simulate --params simmrd/params/uhp_chp.yaml \
  --output uhp_chp.rds --iterations 500 --seed 42
```

## Scope

- **In scope now:** individual-level base model in Python (uv); summary-statistic MR under pleiotropy, weak instruments,
  sample overlap and winner's curse, via simmrd.
- **Not covered by any package found:** within-family MR (dynastic effects,
  assortative mating, stratification). That needs a pedigree simulator; see
  the "within-family" section of `docs/package-survey.md` for what simACE
  would need.
