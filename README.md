# FLAMINGO - Federated Non-Linear Mendelian Randomization
# Purpose

The purpose of this work is to demonstrate the utility of federated learning for non-linear mendelian randomization. Here we created our own synthetic dataset to simulate learning across bio-banks and compare it with conventional MR.

# Quick Start

Requires [uv](https://docs.astral.sh/uv/) (Python >=3.12 for `data/`, 3.10-3.12 for `federated_learning/`).

```bash
# 1. Clone the repo
git clone git@github.com:collaborativebioinformatics/FLAMINGO.git
cd FLAMINGO

# 2. Simulate ten biobank sites sharing one causal curve

cd data
uv sync
uv run python scripts/simulate_federated_sites.py --shape quadratic

# 3. Conventional MR: per-site summary statistics + IVW meta-analysis vs a pooled fit
uv run python scripts/federated_summary_mr.py --shape quadratic

# 4. Non-linear MR: pooled quadratic 2SLS vs the summary-statistics line
uv run python scripts/federated_nonlinear_mr.py --shape quadratic

# 5. Federated learning: NVFlare FedAvg, one simulated client per site
cd ../federated_learning
uv sync
uv run python job.py --dataset quadratic
```

Results are written to `data/results/` and `federated_learning/results/<dataset>/`.


# Intro

Mendelian randomization (MR) uses genetic variants as natural experiments to estimate whether an exposure (such as BMI) causes an outcome (such as heart disease). It faces the same problem as other multi-site analyses: individual-level data usually cannot leave the bio-bank that holds it. Unlike many other fields, MR has largely worked around this without federated learning (FL). The standard MR methods (inverse-variance weighted, MR-Egger and weighted median) do not need individual-level data. For each genetic variant they need only two numbers: its estimated effect on the exposure and its effect on the outcome, each with a standard error. Cohorts routinely publish these summary statistics. As a result, "two-sample" MR can take the exposure effects from one cohort and the outcome effects from another. When several sites each report their own estimate, combining them with a fixed-effects meta-analysis loses essentially no precision compared with pooling all the raw data, at least for linear models in large samples. Summary statistics also make differences between sites easy to measure. Random-effects models and heterogeneity statistics show how much the estimates disagree across sites, for example because of differences in ancestry, in how participants were recruited, or in how the disease was defined.

# Pipeline

![FLAMINGO pipeline: shared causal model, simulation across 10 biobank sites, summary-statistics MR, non-linear MR and NVFlare FedAvg, overlaid on one dose-response plot](flamingo_pipeline.png)

# Further Reading

- [In-depth reasoning](indepth_reasoning.md): more detail on the methods and the reasoning behind them.
