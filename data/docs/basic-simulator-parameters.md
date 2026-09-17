# Parameters of the basic simulator

`scripts/simulate_basic.py` implements the base model in
`mr-simulation-model.md`. The same parameters drive `scripts/sweep_seeds.py`,
which adds `--seeds` for the number of replicates.

| Parameter | CLI flag | Default | Controls |
|---|---|---|---|
| `n` | `--n` | 10000 | Number of individuals |
| `n_snps` | `--n-snps` | 20 | Number of SNP instruments |
| `theta` | `--theta` | 0.3 | Causal effect of X on Y. The target quantity. |
| `h2_x` | `--h2-x` | 0.10 | Share of Var(X) explained by the SNPs. Sets instrument strength. |
| `gamma_x` | `--gamma-x` | 0.3 | Effect of the confounder U on X |
| `gamma_y` | `--gamma-y` | 0.3 | Effect of the confounder U on Y |
| `seed` | `--seed` | 1 | Random seed |

## Fixed inside the function

| Quantity | Value | Note |
|---|---|---|
| Minor allele frequency | Uniform(0.05, 0.5) per SNP | Independent SNPs, no LD |
| Var(X) | 1 | Residual variance is set to `1 - h2_x - gamma_x^2` so theta and R² read on a standard scale |
| Var(e_y) | 1 | Outcome noise |
| Pleiotropy | none | No SNP has a direct path to Y |

## Outputs

| File | Contents |
|---|---|
| `simulated_data/single/<name>.csv` | One row per individual: `id`, `snp0..snpJ`, `U`, `X`, `Y` |
| `simulated_data/single/<name>.truth.json` | The seven parameters above plus the drawn MAFs and per-SNP betas |

`U` is written to the CSV as an oracle for checks. Neither the naive OLS
nor the 2SLS estimator in `sweep_seeds.py` uses it, and it must not be given
to an MR method under test.

## Non-linear link: `simulate_nonlinear()`

Same SNPs, confounder and exposure as `simulate()`. Only the X to Y link
changes, selected by `--shape`.

| Shape | f(X) | `theta1` | `theta2` |
|---|---|---|---|
| `quadratic` | `theta1 X + theta2 X²` | linear slope at X = 0 | curvature |
| `threshold` | `theta1 min(X, theta2)` | slope below the cutoff | cutoff; effect is flat above it |

On the CLI `--theta` is `theta1` and `--theta2` is the second parameter. The
other seven parameters are unchanged.

The truth file adds `avg_slope`, the population-average derivative E[f'(X)].
That is what a linear MR estimator targets when the true curve is not a line:
`theta1 + 2 theta2 E[X]` for quadratic (so `theta1`, since X is centred) and
`theta1 P(X < theta2)` for threshold.

Single-seed check (n = 10000, default confounding):

| shape | theta1 | theta2 | avg_slope | naive OLS | 2SLS |
|---|---|---|---|---|---|
| quadratic | 0.3 | 0.15 | 0.302 | 0.387 | 0.282 |
| threshold | 0.3 | 0.5 | 0.206 | 0.296 | 0.197 |

Linear 2SLS recovers the average slope in both cases. It says nothing about
the curvature or the cutoff, which is what a non-linear MR method has to add.

## Survival outcome: `simulate_survival()` (`--shape cox`)

Same SNPs, confounder and exposure. The outcome is an event time under a Cox
proportional-hazards model with a Weibull baseline:

```
h(t | X, U) = h0(t) exp(theta X + gamma_y U)
T = scale * (-log V / exp(theta X + gamma_y U))^(1/k),   V ~ Uniform(0, 1)
```

`theta` is the log hazard ratio per unit of X. Columns `time` and `event`
replace `Y`.

| Parameter | CLI flag | Default | Controls |
|---|---|---|---|
| `weibull_k` | fixed | 1.5 | Baseline shape; > 1 means hazard rises with time |
| `weibull_scale` | fixed | 10 | Baseline time scale |
| `censor_frac` | `--censor-frac` | 0.3 | Target share censored by an exponential censoring time |
| `followup` | `--followup` | 15 | Administrative end of follow-up |

The truth file adds `hazard_ratio` and the realised `event_rate`.

`scripts/check_survival.py` fits four Cox models to a dataset: naive (X only),
2SPS (SNP-predicted X), 2SRI (X plus first-stage residual), and oracle (X and
U). Over 40 seeds at the defaults (event rate about 0.6):

| estimator | mean | sd | bias |
|---|---|---|---|
| naive | 0.374 | 0.014 | +0.074 |
| 2SPS | 0.271 | 0.040 | -0.029 |
| 2SRI | 0.287 | 0.040 | -0.013 |
| oracle | 0.300 | 0.014 | 0.000 |

The oracle is unbiased, which confirms the simulation. The two MR estimators
remove most of the confounding but sit below the truth. That is the known
non-collapsibility of the hazard ratio: conditioning on a noisy proxy for X
(2SPS) or leaving part of X's variation unexplained shrinks the coefficient
toward zero even without confounding. The bias grows with the exposure
variance the instruments do not capture, so it is a property of Cox MR, not
of this simulator.
## Multiple sites, one causal curve: `simulate_federated_sites.py`

For federation experiments, `scripts/simulate_federated_sites.py` calls
`simulate_nonlinear()` once per site with a fixed `theta1`, `theta2` (the
quadratic curve is biology, shared across every site) but per-site draws of
the nuisance parameters, mimicking biobanks in different countries:

| Parameter | Distribution | Rationale |
|---|---|---|
| `n` | one draw per equal-width bin of `[--pop-min, --pop-max]` (default 1,000-10,000) | ten distinct population sizes spread across the full range, not clustered near the mean |
| `h2_x` | `Beta(mean * kappa, (1 - mean) * kappa)`, default mean 0.10, kappa 40 | SNP heritability of a trait genuinely varies across ancestries/environments; Beta keeps draws in (0, 1) and concentration `kappa` sets how tight the spread is around the mean |
| `gamma_x`, `gamma_y` | independent `Beta(mean * kappa, (1 - mean) * kappa)` draws, default mean 0.3, kappa 20 | confounding strength (e.g. SES-driven) plausibly differs by country; `gamma_x` is drawn independently from `gamma_y` |

`gamma_x` is clipped to `sqrt(0.95 - h2_x)` per site so `Var(e_x) = 1 - h2_x -
gamma_x^2` in `simulate_basic.py` never goes negative.

Each site gets its own seed (`--seed + site index`), so SNP MAFs/betas and
individuals differ across sites even though `theta1`/`theta2` don't. Writes
`simulated_data/<out>/site01.csv` .. `site10.csv` (+ matching
`.truth.json`) and a `manifest.{csv,json}` with the sampled `n`, `h2_x`,
`gamma_x`, `gamma_y`, `avg_slope` per site.

```bash
uv run python scripts/simulate_federated_sites.py            # 10 sites, defaults above
uv run python scripts/simulate_federated_sites.py --help     # all knobs
```
