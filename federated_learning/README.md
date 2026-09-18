# Federated learning and federated MR with NVFlare

A minimal NVFlare (2.9) FedAvg setup on the simulated multi-site data in
`../data/simulated_data/federated/<dataset>/`, one simulated client per
`siteNN.csv`. The same small network and federation run under every outcome
model the simulator produces, and under three estimation methods: a naive
predictive fit and two Mendelian randomization (MR) two-stage fits that use
the SNPs as instruments.

## Methods

| Method | Second stage (federated) | What `f` estimates |
|---|---|---|
| `naive` | `outcome ~ f(X)` | `E[outcome | X]`: the association, confounded by `U` |
| `2sri` (default) | `outcome ~ f(X) + c (X - X_hat)` | the causal curve; the first-stage residual is a control function that carries the confounder |
| `2sps` | `outcome ~ f(X_hat)` | the causal curve over the range of `X_hat` (predictor substitution) |

`X_hat` comes from a **site-local** first stage, ordinary least squares of X on
the 20 SNPs on the site's training split. It has to be local: every simulated
site has its own SNP effects and allele frequencies, so there is no shared
instrument for the federation to learn. This mirrors the concatenated route in
`../data/scripts/federated_nonlinear_mr.py`, which also fits a per-site first
stage. Only the second stage is federated. The first-stage test R² is reported
as `fs_r2` (instrument strength, about 0.09 here, matching the simulated
heritability).

The control-function term in `2sri` is linear on purpose. With a flexible `h`
the two heads `f(X)` and `h(X - X_hat)` are nearly collinear when the
instruments explain 9% of X, and the split between them is arbitrary. A linear
term is the classic 2SRI and is identified here because the confounder enters
additively.

## Outcome families

| Dataset | Outcome | Task | Loss | Metrics |
|---|---|---|---|---|
| `linear`, `quadratic`, `threshold` | `Y` real-valued | continuous | MSE | mse, mae, r2 |
| `binary_quadratic_logistic` | `Y` in {0, 1} | binary | BCE | accuracy, precision, recall, f1, auc |
| `cox`, `cox_rare` | `time`, `event` | survival | Cox partial likelihood (Breslow) | c_index |

The task is detected from the columns and `manifest.json`, see `src/tasks.py`.
For binary and survival outcomes the two-stage fits are the usual 2SRI/2SPS
approximations on the logit and log-hazard scales.

## Files

| File | What |
|---|---|
| `job.py` | Builds the `FedAvgJob`, attaches `src/client.py` to every site, runs the chosen engine, prints per-round tables, writes results and plots |
| `src/model.py` | `MLP` (naive) and `MRModel` (2sri / 2sps second stage). Output is predicted Y, a logit, or a log relative hazard |
| `src/tasks.py` | The three outcome families: targets, loss (Cox partial likelihood with Breslow ties), metrics, and the true causal curve from the manifest |
| `src/fedsite.py` | One site: 80/20 split, local first stage, local training, evaluation, fitted-curve recording. Shared by both engines |
| `src/client.py` | NVFlare Client API script: receives the global weights, runs the site's round, sends the weights back |
| `src/local_engine.py` | In-process FedAvg over the same `Site` objects, no NVFlare processes |
| `src/plots.py` | Per-run metric and fitted-curve plots plus the all-datasets overview |
| `fedmr_job.py`, `src/fedmr_controller.py`, `src/fedmr_client.py` | FedMR through NVFlare: sufficient-statistics client, summing server workflow, identity check against the in-process and pooled fits |
| `results/fedmr/<dataset>/estimates.<basis>.json`, `results/fedmr/summary.csv` | FedMR estimates, SEs, robust SEs, diagnostics, and the identity-check differences |
| `results/<method>/<dataset>/` | `metrics.csv`, `curves.csv`, `metrics_by_round.{global,local}.png`, `fitted_curve.png` |
| `results/fitted_curves_all.png`, `results/summary.csv` | Overview across datasets and methods |
| `workspace/` | Simulator output and per-job logs (git-ignored) |

## Run

```bash
uv sync                                        # nvflare, torch (CPU), pandas, scikit-learn, lifelines, matplotlib
uv run python job.py --dataset quadratic       # 2SRI on one dataset in the NVFlare simulator (~30 s)
uv run python job.py --all --method naive --method 2sri --method 2sps --engine local --jobs 6   # full sweep, ~1 min
uv run python job.py --dataset cox --method 2sps --rounds 10 --epochs 3 --lr 0.005
uv run python src/plots.py                     # re-render every plot from results/ without training
```

## FedMR: the exact federated 2SLS, no training

`fedmr_job.py` runs a different kind of federation on the same files: each
client computes the sufficient statistics of a two-stage least squares MR
(`flamingo_fedmr`, in `../fedmr/`), the server (`src/fedmr_controller.py`,
a `ModelController` that sums rather than averages) adds them and solves,
and a second round collects the robust-covariance term. Two rounds, no
epochs, and the result equals the pooled 2SLS to machine precision, with
standard errors. Only continuous outcomes.

```bash
uv run python fedmr_job.py --dataset linear                 # local first stage, 2 rounds
uv run python fedmr_job.py --dataset linear_shared          # shared SNPs across sites
uv run python fedmr_job.py --dataset quadratic --basis quadratic
uv run python fedmr_job.py --all
```

After each job the script re-runs the protocol in-process on the same
files and the pooled fit in numpy, and fails unless all three agree to
1e-10. `results/fedmr/summary.csv` has the estimates; on every set the
observed differences are at 1e-16. The FedAvg 2SRI model above and FedMR
answer different questions (a flexible curve without an analytic CI versus
a specified basis with one), and the forest plots in `../data/results/`
keep both rows. See `../data/docs/federated-exact-mr.md`.

## Engines and speed

`--engine nvflare` (default) is the real federation: the NVFlare simulator
starts a server and one client per site and exchanges weights through
NVFlare. `--engine local` runs the same `Site` code and the same
train-size-weighted FedAvg in one process. The two agree up to the random
initial weights; use `local` for sweeps and `nvflare` to show the federation.

The model is tiny, so wall time is overhead, not training. What was found and
what job.py does about it, per job of 10 sites, 5 rounds, 2 local epochs:

| Cost | Cause | Fix |
|---|---|---|
| 7x slower training | torch's default intra-op threads on small batches | `fedsite.py` pins torch to one thread |
| 2 s idle per round | NVFlare's server tells clients to wait `task_request_interval` (default 2 s) between task requests | `--task_interval 0.05`, written into the exported job config |
| ~10 s client start-up, ~4 s imports, ~2 s plots | NVFlare simulator and Python | unavoidable per job; `--jobs N` runs N jobs concurrently |
| ~1 ms per optimizer step in eager PyTorch | batch 64 meant 690 steps per epoch | default batch is now 256 (512 for survival); test metrics and the 2SRI curve are unchanged or better |

Measured on a 12-core CPU: one NVFlare job 38 s before, 32 s after; one local
job about 10 s; the full sweep of 7 datasets x 3 methods took 7 min before
(NVFlare, 6 jobs at a time) and 52 s now (local, 6 jobs at a time).

## What happens each round

1. The server sends the global second-stage model to every client.
2. Each client evaluates it on its own 20% test split and prints the task's
   metrics (`[siteNN] round r global-model test: ...`), and records the
   causal-curve head `f` on a fixed grid of X values.
3. Each client trains for `--epochs` epochs on its 80% train split (Adam).
4. Each client evaluates the updated model on the same test split and prints it.
5. Weights go back to the server, which averages them weighted by train size.

After the last training round every client evaluates the final aggregate
once more (an evaluation-only round, so `--rounds 5` reports rounds 0 to 5,
where round 5 has only the "global" stage). `job.py` prints one table per
round and stage with every client's metrics plus a test-size-weighted mean,
then writes `results/summary.csv` and `results/fitted_curves_all.png` over
every run present in `results/`.

## Reading the fitted curves

`fitted_curve.png` overlays the last-round `f(X)` on the true causal curve
from the manifest and on the pooled binned mean of the outcome, all relative
to X = 0. The naive curve tracks the binned means, which include the
confounder's path. The 2SRI curve moves away from them toward the causal
curve: below the naive fit where confounding inflates the association and
above it where it deflates it. The correction is only as strong as the
instruments, so the MR curves are noisier than the naive one and least
reliable in the tails of X, where few people and little instrument variation
sit. For 2SPS the plot is solid only within two standard deviations of
`X_hat`, which is where `f(X_hat)` is identified, and dotted beyond. Binary
outcomes are drawn on the logit scale, which is where the model's `f` lives;
the binned data points are converted to logits to match. `../data/scripts/federated_nonlinear_mr.py` draws the naive and 2SRI
curves next to the concatenated 2SLS fit and the summary-statistics IVW line.

## Robust and privacy-preserving federation (optional)

`src/fedsec/` adds defenses against malicious sites and against anyone who
watches the updates: robust aggregation rules, simulated attacks, site-level
differential privacy with exact accounting, secure aggregation, and leakage
probes. Every component is off unless a YAML config turns it on, and with
everything off the runs above are byte-identical to before. Threat model,
every assumption, and results: [ROBUST_PRIVATE.md](ROBUST_PRIVATE.md).

```bash
uv run python job.py --dataset quadratic --secure_config configs/secure/robust_median.yaml
uv run python job.py --dataset cox --engine local --secure_config configs/secure/dp_distributed_secagg.yaml
uv run python secure_experiments.py --suite all --jobs 12     # robustness, privacy, combined sweeps
uv run python tests/test_fedsec.py
```

Secure runs write to `results/secure/`; `results/<method>/` is never touched by them.

| File | What |
|---|---|
| `src/fedsec/` | config, attacks, aggregators, dp, secagg, leakage, protocol (shared by both engines), NVFlare aggregator, report figures |
| `configs/secure/*.yaml` | one scenario per file; `experiments.yaml` defines the sweeps |
| `secure_experiments.py` | runs the sweeps with the local engine, writes `results/secure/experiments/<suite>/` |
| `tests/test_fedsec.py` | unit checks of the building blocks |

## Notes

- Last-round weighted test metrics for every run are in `results/summary.csv`, written by `job.py` and by `src/plots.py`.
  Test metrics are similar across methods because all three predict the
  outcome from the same information; the methods differ in what `f` means,
  not in predictive accuracy.
- `X` alone explains little of the outcome by design (R² near 0.1, AUC near
  0.58, c-index near 0.6), so the binary model rarely predicts a case and its
  accuracy sits at the majority-class rate.
- `cox_rare` has 5 to 26 events per test split, so its per-site c-index is noisy.
