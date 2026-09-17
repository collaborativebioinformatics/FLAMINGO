"""FedMR: exact federated one-sample MR from summed sufficient statistics."""

from .data import SiteData, load_site_csv, load_sites
from .estimator import Diagnostics, FedMRResult, IdentificationError, fit, robust_cov
from .protocols import LocalFirstStageFedMR, Run, SharedInstrumentFedMR, fedmr, protocol_for
from .schema import Design, Layout, Role, centre_within
from .statistics import SiteStats, Stats, aggregate, aggregate_robust, site_robust_stats, site_stats

__all__ = [
    "SiteData", "load_site_csv", "load_sites",
    "Diagnostics", "FedMRResult", "IdentificationError", "fit", "robust_cov",
    "LocalFirstStageFedMR", "SharedInstrumentFedMR", "Run", "fedmr", "protocol_for",
    "Design", "Layout", "Role", "centre_within",
    "SiteStats", "Stats", "aggregate", "aggregate_robust", "site_robust_stats", "site_stats",
]
