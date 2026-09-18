"""Robust and privacy-preserving extensions to the federated MR second stage.

Every component is off by default and switched on from a YAML config
(configs/secure/*.yaml, loaded by config.load_config). With every section off,
job.py and local_engine.py take their original code path unchanged.

    config.py       the config schema, defaults, validation, and the assumption each switch carries
    attacks.py      malicious-site behaviour: model poisoning of the update, data poisoning of training
    aggregators.py  server-side robust aggregation rules and update-norm bounding
    dp.py           update clipping, Gaussian noise, and the privacy accountant (Gaussian DP)
    secagg.py       pairwise-mask secure aggregation over fixed-point integers mod 2^64
    leakage.py      what an observer of one site's message can learn (update similarity, membership inference)
    protocol.py     client and server message pipelines shared by both engines
    nvflare_aggregator.py   the server pipeline as an NVFlare ModelAggregator
    report.py       figures for secure_experiments.py

See ../../ROBUST_PRIVATE.md for the threat model and the full list of assumptions.
"""
