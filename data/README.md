# Simulated MR data (team10-data)

Simulated data for testing Mendelian randomization (MR) methods.

This repo collects the design notes, the survey of existing simulators, and
the parameter files used to generate benchmark datasets.

## Contents

| Path | What |
|---|---|
| `docs/mr-simulation-model.md` | The data-generating model and the knobs an MR benchmark should vary |
| `docs/package-survey.md` | Existing simulators (simmrd, simulateGP, GWASBrewer) and the within-family gap |
| `docs/basic-simulator-parameters.md` | Parameter table for the Python simulator, fixed quantities, outputs |
| `docs/simmrd-review.md` | Source-level review of simmrd: API, outputs, limits |
| `scripts/simulate_basic.py` | Pure-Python generator: linear, quadratic, threshold, or Cox survival outcome; no pleiotropy, no LD |
| `scripts/check_survival.py` | Naive, 2SPS, 2SRI and oracle Cox fits against the true log hazard ratio |
| `scripts/simulate_federated_sites.py` | Ten sites sharing one quadratic causal curve, per-site heritability/confounding sampled from a distribution |
| `simulated_data/basic.truth.json` | True parameters of the checked-in baseline draw |
| `simulated_data/federated/` | Ten-site federated draw: `site01..site10.{csv,truth.json}` + `manifest.{csv,json}` |
| `simmrd/params/*.yaml` | Parameter files for the simmrd CLI, one per scenario |
| `simmrd/README.md` | How to run the simmrd CLI against these parameter files |

## Quick start (Python, uv)

```bash
uv run python scripts/simulate_basic.py            # defaults: n=10000, 20 SNPs, theta=0.3
uv run python scripts/simulate_basic.py --help     # all knobs
```

Writes `simulated_data/basic.csv` (id, snp0..snpJ, U, X, Y) and `simulated_data/basic.truth.json`
(theta, MAFs, per-SNP betas, confounder strengths, seed). Both are checked in;
the CSV can be regenerated from the seed in the truth file. Sanity check on the
default seed: naive OLS 0.39, 2SLS through the SNPs 0.29, true theta 0.30.

```bash
uv run python scripts/simulate_federated_sites.py  # 10 sites, one shared quadratic curve
```

Writes `simulated_data/federated/site01..site10.{csv,truth.json}` plus a
`manifest.{csv,json}`. Population size (1,000-10,000), SNP heritability
`h2_x`, and confounder strengths `gamma_x`/`gamma_y` are drawn per site from
distributions meant to mimic real variation between countries; `theta1`,
`theta2` (the quadratic causal curve) are the same at every site. See
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
