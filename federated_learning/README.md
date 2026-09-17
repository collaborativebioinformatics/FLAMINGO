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
| `job.py` | Builds the `FedAvgJob`, attaches `src/client.py` to every site, runs the simulator, prints per-round tables, writes results and plots |
| `src/model.py` | `MLP` (naive) and `MRModel` (2sri / 2sps second stage). Output is predicted Y, a logit, or a log relative hazard |
| `src/tasks.py` | The three outcome families: targets, loss, metrics, and the true causal curve from the manifest |
| `src/client.py` | Client API script: 80/20 split, local first stage, local training, evaluation, fitted-curve recording |
| `src/plots.py` | Per-run metric and fitted-curve plots plus the all-datasets overview |
| `results/<method>/<dataset>/` | `metrics.csv`, `curves.csv`, `metrics_by_round.{global,local}.png`, `fitted_curve.png` |
| `results/fitted_curves_all.png`, `results/summary.csv` | Overview across datasets and methods |
| `workspace/` | Simulator output (git-ignored) |

## Run

```bash
uv sync                                                # nvflare, torch (CPU), pandas, scikit-learn, lifelines, matplotlib
uv run python job.py --dataset quadratic               # 2SRI on one dataset: 10 sites, 5 rounds, 2 local epochs
uv run python job.py --all --method naive --method 2sri --method 2sps
uv run python job.py --dataset cox --method 2sps --rounds 10 --epochs 3 --lr 0.005
uv run python src/plots.py                             # re-render every plot from results/ without training
```

## What happens each round

1. The server sends the global second-stage model to every client.
2. Each client evaluates it on its own 20% test split and prints the task's
   metrics (`[siteNN] round r global-model test: ...`), and records the
   causal-curve head `f` on a fixed grid of X values.
3. Each client trains for `--epochs` epochs on its 80% train split (Adam).
4. Each client evaluates the updated model on the same test split and prints it.
5. Weights go back to the server, which averages them weighted by train size.

After the simulator finishes, `job.py` prints one table per round and stage
with every client's metrics plus a test-size-weighted mean.

## Reading the fitted curves

`fitted_curve.png` overlays the last-round `f(X)` on the true causal curve
from the manifest and on the pooled binned mean of the outcome, all relative
to X = 0. The naive curve tracks the binned means, which include the
confounder's path. The 2SRI curve moves away from them toward the causal
curve: below the naive fit where confounding inflates the association and
above it where it deflates it. The correction is only as strong as the
instruments, so the MR curves are noisier than the naive one and least
reliable in the tails of X, where few people and little instrument variation
sit. `../data/scripts/federated_nonlinear_mr.py` draws the naive and 2SRI
curves next to the concatenated 2SLS fit and the summary-statistics IVW line.

## Notes

- Last-round weighted test metrics for every run are in `results/summary.csv`.
  Test metrics are similar across methods because all three predict the
  outcome from the same information; the methods differ in what `f` means,
  not in predictive accuracy.
- `X` alone explains little of the outcome by design (R² near 0.1, AUC near
  0.58, c-index near 0.6), so the binary model rarely predicts a case and its
  accuracy sits at the majority-class rate.
- `cox_rare` has 5 to 26 events per test split, so its per-site c-index is noisy.
