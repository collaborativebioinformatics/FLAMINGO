## 1

Aim: estimate a shared non-linear exposure–outcome causal relationship across biobank sites using genetic instruments, without transferring individual records. The diagram is conceptual; ten sites reflect the committed synthetic federation. No formal privacy guarantee is implied by keeping records local. Sources: README.md; writing/methods.md; data/docs/mr-simulation-model.md.

## 2

G: genetic instruments; X: exposure; Y: outcome; U: unmeasured confounding. IVW combines per-SNP summary associations into an average slope. FedAvg trains federated neural estimators (including two-stage residual inclusion, 2SRI) via model updates. FedMR aggregates site-centred sufficient statistics and solves two-stage least squares, reproducing the corresponding pooled estimator for its specified basis. FedMR may require one or two rounds depending on basis, first-stage protocol and covariance choice. The route diagrams are conceptual, not fitted results. Source: writing/methods.md; federated_learning/README.md; dashboard/README.md.

## 3

Real saved quadratic simulation results, not clinical data. Curves show the zero-intercept structural component θ1*x + θ2*x², with truth θ1=0.30, θ2=0.15. FedMR quadratic: θ1=0.31696352932430816, θ2=0.14851304929760675. Pooled quadratic: θ1=0.3169635293243077, θ2=0.14851304929760703. The maximum coefficient difference reported in the CSV is 4.440892098500626e-16. This is numerical identity to the corresponding pooled estimator, not error versus truth. IVW represents a single slope and does not recover curvature. N=55,182 across ten sites per writing/results.md and the committed quadratic manifest. Curves do not display uncertainty. Sources: data/results/fedmr.quadratic.csv; data/results/fedmr.quadratic.truth.json; data/simulated_data/federated/quadratic/manifest.csv; writing/results.md.

## 4

Actual screenshot of the local FLAMINGO Streamlit dashboard from this branch, showing the Sensitivity & invariance area using committed simulation data. Three workflows: simulate heterogeneous biobanks; compare conventional MR, federated learning and non-linear estimators; stress-test assumptions using site exclusion, regularisation, robustness, invariance and what-if perturbations. No new experiment was run for this screenshot. Source: dashboard/app.py; dashboard/sensitivity_tab.py; dashboard/README.md. Launch: cd dashboard && uv run streamlit run app.py.
