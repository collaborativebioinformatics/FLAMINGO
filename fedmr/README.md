# flamingo-fedmr

Exact federated one-sample Mendelian randomization from summed sufficient
statistics. Pure numpy, no training loop. Shared by `../data` (analysis
scripts and tests) and `../federated_learning` (NVFlare transport); both
depend on it as an editable path dependency.

Modules: `schema` (column roles and the per-site design), `statistics` (what a
site releases and how the coordinator sums it), `estimator` (the 2SLS solve,
covariances, diagnostics), `protocols` (shared-instrument and local-first-stage
protocols, quadratic basis, cross-fitting), `data` (CSV loading for the
simulated sites). See `../data/docs/federated-exact-mr.md`.
