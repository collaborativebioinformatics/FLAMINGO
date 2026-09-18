<p align="center">
  <img src="images/flamingo-logo-dark-rounded.png" alt="FLAMINGO logo" width="200">
</p>

# FLAMINGO - Federated Non-Linear Mendelian Randomization
# Purpose

The purpose of this work is to demonstrate the utility of federated learning for non-linear mendelian randomization. Here we created our own synthetic dataset to simulate learning across bio-banks and compare it with conventional MR.

# Quick Start

Requires [uv](https://docs.astral.sh/uv/) and Python >=3.12. One `uv sync` at the
root builds a single environment for the whole repository — `data/`,
`federated_learning/`, `fedmr/` and the dashboard all share it.

```bash
# 1. Clone the repo and build the environment (once)
git clone git@github.com:collaborativebioinformatics/FLAMINGO.git
cd FLAMINGO
uv sync

# 2. Simulate ten biobank sites sharing one causal curve
cd data
uv run python scripts/simulate_federated_sites.py --shape quadratic

# 3. Conventional MR: per-site summary statistics + IVW meta-analysis vs a pooled fit
uv run python scripts/federated_summary_mr.py --shape quadratic

# 4. Non-linear MR: pooled quadratic 2SLS vs the summary-statistics line
uv run python scripts/federated_nonlinear_mr.py --shape quadratic

# 5. FedMR: exact federated 2SLS from summed sufficient statistics (equals the pooled fit)
uv run python scripts/federated_exact_mr.py --all
uv run pytest -q                                   # identity tests against the pooled fits

# 6. Federated learning: NVFlare FedAvg, one simulated client per site
cd ../federated_learning
uv run python job.py --dataset quadratic

# 7. FedMR through NVFlare: two rounds, no training, checked against the pooled fit
uv run python fedmr_job.py --all
```

Results are written to `data/results/` and `federated_learning/results/<dataset>/`.
FedMR is described in [data/docs/federated-exact-mr.md](data/docs/federated-exact-mr.md).

# Interactive dashboard

To explore the whole pipeline without the command line — set the simulation
parameters, generate a federation, then run the federated learning and MR
workflows over it and compare every estimator side by side:

```bash
cd dashboard
uv run streamlit run app.py
```

See [dashboard/README.md](dashboard/README.md).


# Intro

Mendelian randomization (MR) uses genetic variants as natural experiments to estimate whether an exposure (such as BMI) causes an outcome (such as heart disease). It faces the same problem as other multi-site analyses: individual-level data usually cannot leave the bio-bank that holds it. Unlike many other fields, MR has largely worked around this without federated learning (FL). The standard MR methods (inverse-variance weighted, MR-Egger and weighted median) do not need individual-level data. For each genetic variant they need only two numbers: its estimated effect on the exposure and its effect on the outcome, each with a standard error. Cohorts routinely publish these summary statistics. As a result, "two-sample" MR can take the exposure effects from one cohort and the outcome effects from another. When several sites each report their own estimate, combining them with a fixed-effects meta-analysis loses essentially no precision compared with pooling all the raw data, at least for linear models in large samples. Summary statistics also make differences between sites easy to measure. Random-effects models and heterogeneity statistics show how much the estimates disagree across sites, for example because of differences in ancestry, in how participants were recruited, or in how the disease was defined.

# Pipeline

![FLAMINGO pipeline: shared causal model, simulation across 10 biobank sites, summary-statistics MR, non-linear MR and NVFlare FedAvg, overlaid on one dose-response plot](flamingo_pipeline.png)

# Further Reading

- [In-depth reasoning](indepth_reasoning.md): more detail on the methods and the reasoning behind them.
- [FedMR](data/docs/federated-exact-mr.md): exact federated one-sample MR from sufficient statistics, the protocols, the identity checks and the seed sweep; plan and review history in [data/docs/fedmr-plan.md](data/docs/fedmr-plan.md).
- [Robust and privacy-preserving federation](federated_learning/ROBUST_PRIVATE.md): malicious sites, what the server learns from the updates, differential privacy and secure aggregation for the federated MR second stage.
