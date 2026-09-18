# Simulation models

Every dataset in this repo comes from one exposure model combined with one
causal curve and one outcome type. This page lists the options, where each is
implemented, and which checked-in datasets use them. Parameter tables with
defaults are in `basic-simulator-parameters.md`.

## Exposure model (shared by everything)

```
G_j ~ Binomial(2, p_j)      j = 1..20 independent SNPs, MAF uniform on (0.05, 0.5)
U   ~ N(0, 1)               unmeasured confounder
X   = sum_j beta_j G_j + gamma_x U + e_x        scaled to Var(X) = 1
```

The SNPs explain `h2_x` of Var(X) (default 0.10) and the confounder enters
with `gamma_x` (default 0.3). Implemented once in `_draw_exposure()` in
`scripts/simulate_basic.py`; every simulator below calls it.

## Causal curves f(X)

| shape | f(X) | parameters | average slope a linear MR targets |
|---|---|---|---|
| `linear` | θ·X | θ | θ |
| `quadratic` | θ1·X + θ2·X² | θ1 slope at X = 0, θ2 curvature | θ1 + 2θ2·E[X], so θ1 |
| `threshold` | θ1·min(X, θ2) | θ1 slope below the cutoff θ2, flat above | θ1·P(X < θ2) |

Defined in `causal_curve()` in `scripts/simulate_basic.py`. The quadratic
with θ1 = 0 is the U-shape: zero average slope, so linear MR reports no
effect (`simulated_data/federated/ushape/`).

## Outcome types

| outcome | model | curves | function |
|---|---|---|---|
| continuous | Y = f(X) + γ_y·U + e_y | all three | `simulate()`, `simulate_nonlinear()` in `simulate_basic.py` |
| survival | Cox proportional hazards, h(t) = h0(t)·exp(θX + γ_y·U), Weibull baseline (k = 1.5, scale 10), exponential censoring at a target fraction plus an administrative cutoff | linear only; θ is the log hazard ratio | `simulate_survival()` in `simulate_basic.py` |
| binary | liability L = f(X) + γ_y·U + e_y, standardised, then turned into 0/1 at a target prevalence (default 0.3) | all three | `simulate_binary()` in `simulate_binary.py` |

Binary outcomes have two links. `liability` is a deterministic threshold at
the prevalence quantile of the standardised liability, the classic
liability-threshold model. `logistic` solves for an intercept so that the
mean of expit(intercept + L) equals the prevalence, then draws
Y ~ Bernoulli, which adds sampling noise on top of the liability.

Output columns are `id`, `snp0..snp19`, `U`, `X`, then `Y` for continuous
and binary or `time`, `event` for survival. `U` is written as an oracle and
must not be given to a method under test. Every dataset has a `.truth.json`
beside it with the parameters, the drawn MAFs and per-SNP betas, and the
average slope.

## Single-site generators

| script | outcome | `--shape` | default output |
|---|---|---|---|
| `simulate_basic.py` | continuous or survival | `linear`, `quadratic`, `threshold`, `cox` | `simulated_data/single/basic` |
| `simulate_binary.py` | binary | `linear`, `quadratic`, `threshold`, plus `--link` and `--prevalence` | `simulated_data/binary` |

## Federated generators

Both draw ten sites around one shared curve. Population size is spread
across 1,000 to 10,000; `h2_x`, `gamma_x` and `gamma_y` are drawn per site
from Beta distributions around the single-site defaults, so sites differ in
instrument strength and confounding but not in biology. Each site has its
own seed, so its SNPs are distinct variants from every other site's.

| script | outcomes | how the model is chosen | output |
|---|---|---|---|
| `simulate_federated_sites.py` | continuous, survival | `--shape linear|quadratic|threshold|cox`, with `--censor-frac`, `--followup` for cox | `simulated_data/federated/<shape>` |
| `simulate_federated.py` | continuous, binary, survival | `--outcome` plus `--shape`, `--link`, `--prevalence`, and the Weibull and censoring knobs, through the `make_simulator()` registry in `simulators.py` | `simulated_data/federated/continuous_<shape>`, `binary_<shape>_<link>`, or `survival` |

The registry is the extension point: a new outcome type is one branch in
`simulators.py`, and the federated orchestrator does not change.

Both generators take the same heterogeneity knobs. `--shared-snps` is on
by default (`--no-shared-snps` restores site-specific variants, which is
how every checked-in set except `linear_shared` was made): it draws one
MAF vector and one `beta` vector from the base seed and reuses them at
every site, with
each site's realized `h2_x` computed from those effects at its own allele
frequencies (recorded in the manifest, replacing the drawn value);
`--maf-shift SD` perturbs the shared MAFs per site on the logit scale;
`--theta-sd SD` gives each site its own `theta1`; `--pleiotropy-mean` and
`--pleiotropy-sd` add direct SNP -> outcome effects (shared across sites
when the SNPs are shared, drawn per site otherwise). The survival simulator
accepts none of these.

## Checked-in datasets

| path | model | notes |
|---|---|---|
| `single/basic`, `single/basic_seed2` | continuous, linear, θ = 0.3 | seeds 1 and 2 |
| `single/quadratic` | continuous, quadratic, θ1 = 0.3, θ2 = 0.15 | |
| `single/threshold` | continuous, threshold, θ1 = 0.3, cutoff 0.5 | |
| `single/cox` | survival, θ = 0.3 | event rate 0.61 |
| `federated/linear` | continuous, linear, θ = 0.3 | 10 sites, n = 55,182 |
| `federated/quadratic` | continuous, quadratic, θ1 = 0.3, θ2 = 0.15 | same sites |
| `federated/threshold` | continuous, threshold, θ1 = 0.3, cutoff 0.5 | same sites |
| `federated/ushape` | continuous, quadratic, θ1 = 0, θ2 = 0.15 | zero average slope |
| `federated/cox` | survival, θ = 0.3 | same sites |
| `federated/cox_rare` | survival, θ = 0.3, sites of 300 to 3,000, follow-up 1.2 | 24 to 128 events per site |
| `federated/binary_quadratic_logistic` | binary, quadratic, θ1 = 0.3, θ2 = 0.15, logistic link, prevalence 0.3 | same sites |
| `federated/linear_shared` | continuous, linear, θ = 0.3, `--shared-snps` (now the default) | same sites and sizes, the same 20 SNPs everywhere; for the shared-instrument Fed-2SLS protocol |

Each federated folder holds `site01..site10.{csv,truth.json}` and a
`manifest.{csv,json}` with the per-site draws. Folders that have been through
`federated_summary_mr.py` also hold a `sumstats/` directory of per-site GWAS
summary statistics.
