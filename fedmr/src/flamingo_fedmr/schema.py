"""Column roles and the per-site design that every protocol produces.

A site's second-stage problem is Y = W theta + error with instruments Z.
Every column of Z and W has a name (for alignment across sites) and a role:

    ENDOGENOUS   structural regressor instrumented by Z          (in W only)
    EXOGENOUS    covariate that instruments itself                (in W and Z, same name)
    INSTRUMENT   excluded instrument                              (in Z only)

Site intercepts are never columns: every protocol centres each column within
its site before forming cross-products (the Frisch-Waugh-Lovell step), and
records how many fixed effects were absorbed so degrees of freedom stay
right. That keeps the transmitted matrices small and site-agnostic, and it
means the coordinator never receives a per-site dummy column.

Names alone do not carry enough information for first-stage diagnostics;
the roles do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Role(str, Enum):
    ENDOGENOUS = "endogenous"
    EXOGENOUS = "exogenous"
    INSTRUMENT = "instrument"


@dataclass
class Design:
    """Centred design matrices for one logical site.

    Z: (n, dz) instruments, W: (n, dw) structural regressors, Y: (n,) outcome.
    z_roles / w_roles: Role per column. Exogenous columns appear in both Z and
    W under the same name. `absorbed` counts fixed effects removed by centring
    (one site intercept per logical site, so 1 here)."""
    site: str
    Z: np.ndarray
    W: np.ndarray
    Y: np.ndarray
    z_names: list
    w_names: list
    z_roles: list
    w_roles: list
    absorbed: int = 1

    def __post_init__(self):
        assert self.Z.shape[0] == self.W.shape[0] == len(self.Y)
        assert len(self.z_names) == self.Z.shape[1] and len(self.w_names) == self.W.shape[1]
        assert len(self.z_roles) == self.Z.shape[1] and len(self.w_roles) == self.W.shape[1]
        exog_z = {n for n, r in zip(self.z_names, self.z_roles) if r == Role.EXOGENOUS}
        exog_w = {n for n, r in zip(self.w_names, self.w_roles) if r == Role.EXOGENOUS}
        if exog_z != exog_w:
            raise ValueError(f"exogenous columns must appear in both Z and W: {exog_z ^ exog_w}")

    @property
    def n(self):
        return len(self.Y)

    @property
    def endogenous(self):
        return [n for n, r in zip(self.w_names, self.w_roles) if r == Role.ENDOGENOUS]

    @property
    def instruments(self):
        return [n for n, r in zip(self.z_names, self.z_roles) if r == Role.INSTRUMENT]

    @property
    def exogenous(self):
        return [n for n, r in zip(self.w_names, self.w_roles) if r == Role.EXOGENOUS]


@dataclass
class Layout:
    """The coordinator's global column layout, the union of the sites' names with their roles."""
    z_names: list
    w_names: list
    z_roles: list
    w_roles: list
    absorbed: int = 0
    sites: list = field(default_factory=list)

    @property
    def endogenous(self):
        return [n for n, r in zip(self.w_names, self.w_roles) if r == Role.ENDOGENOUS]

    @property
    def instruments(self):
        return [n for n, r in zip(self.z_names, self.z_roles) if r == Role.INSTRUMENT]

    @property
    def exogenous(self):
        return [n for n, r in zip(self.w_names, self.w_roles) if r == Role.EXOGENOUS]

    def z_index(self, names):
        idx = {n: i for i, n in enumerate(self.z_names)}
        return np.array([idx[n] for n in names], dtype=int)

    def w_index(self, names):
        idx = {n: i for i, n in enumerate(self.w_names)}
        return np.array([idx[n] for n in names], dtype=int)


def centre_within(*arrays):
    """Subtract each column's mean: absorbs one intercept per call (one logical site)."""
    out = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        out.append(a - a.mean(axis=0, keepdims=True) if a.ndim == 2 else a - a.mean())
    return out
