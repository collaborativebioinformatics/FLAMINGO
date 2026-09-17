"""Individual-level data for one site, and loading of the repo's simulated CSVs.

The simulated files carry a column `U`, the unmeasured confounder, written as
an oracle for checks. It must never reach a design matrix. That rule lives
here, in the loader, because a matrix estimator cannot tell an oracle from
a covariate by its contents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORACLE_COLUMNS = ("U",)


@dataclass
class SiteData:
    """G: (n, m) dosages for one effect allele per SNP, X: (n,), Y: (n,), C: (n, q) covariates."""
    name: str
    G: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    C: np.ndarray = None
    cov_names: tuple = ()

    def __post_init__(self):
        self.G = np.asarray(self.G, dtype=np.float64)
        self.X = np.asarray(self.X, dtype=np.float64)
        self.Y = np.asarray(self.Y, dtype=np.float64)
        if self.C is None:
            self.C = np.empty((len(self.X), 0))
        self.C = np.asarray(self.C, dtype=np.float64)
        assert self.G.shape[0] == len(self.X) == len(self.Y) == self.C.shape[0]
        assert self.C.shape[1] == len(self.cov_names)

    @property
    def n(self):
        return len(self.X)


def _read_csv(path):
    try:
        import polars as pl
        df = pl.read_csv(path)
        return list(df.columns), (lambda c: df[c].to_numpy())
    except ImportError:
        import pandas as pd
        df = pd.read_csv(path)
        return list(df.columns), (lambda c: df[c].to_numpy())


def load_site_csv(path, name=None, covariates=()) -> SiteData:
    for c in covariates:
        if c in ORACLE_COLUMNS:
            raise ValueError(f"{c!r} is a simulator oracle and may not be used as a covariate")
    path = Path(path)
    cols, get = _read_csv(path)
    snp_cols = [c for c in cols if c.startswith("snp")]
    G = np.column_stack([get(c) for c in snp_cols])
    C = np.column_stack([get(c) for c in covariates]) if covariates else None
    return SiteData(name or path.stem, G, get("X"), get("Y"), C, tuple(covariates))


def load_sites(folder, covariates=()) -> list[SiteData]:
    return [load_site_csv(p, covariates=covariates) for p in sorted(Path(folder).glob("site*.csv"))]
