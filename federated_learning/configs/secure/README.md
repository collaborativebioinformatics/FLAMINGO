# fedsec scenario configs

One YAML per scenario for `job.py --secure_config`. Missing keys take the
defaults in `src/fedsec/config.py`; every section is off unless `enabled: true`.
Results go to `results/secure/<name>/<method>/<dataset>/`. Threat model and
assumptions: `../../ROBUST_PRIVATE.md`. The sweep in `secure_experiments.py` builds
its scenarios from `experiments.yaml` instead.

| File | Question it answers |
|---|---|
| `leakage_probe.yaml` | Plain FedAvg, measured: what does the server learn from one site's update? |
| `attack_signflip.yaml` | One malicious site (the largest) flips and boosts its update. Undefended FedAvg. |
| `robust_median.yaml` | Same attack, weighted coordinate-wise median |
| `robust_geomedian.yaml` | Same attack, weighted geometric median (RFA) |
| `robust_normbound.yaml` | Same attack, FedAvg after adaptive update-norm bounding |
| `attack_shuffle_trimmed.yaml` | Two sites train on shuffled outcomes ("hide the effect"), trimmed mean |
| `secagg.yaml` | Secure aggregation: the server sees only the sum |
| `dp_central.yaml` | Central DP: trusted server clips and adds noise |
| `dp_local.yaml` | Local DP: each site adds all the noise itself |
| `dp_distributed_secagg.yaml` | Distributed DP behind secure aggregation: untrusted server, central-DP noise level |
| `private_and_bounded.yaml` | Central DP with server-side clipping against a scaling attack: clipping doubles as a robustness bound |
