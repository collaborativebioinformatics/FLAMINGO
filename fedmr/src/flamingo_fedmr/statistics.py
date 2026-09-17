"""What a site releases, and how the coordinator sums it.

Round 1, from a centred Design (Z, W, Y):

    A = Z'Z   B = Z'W   c = Z'Y   D = W'W   e = W'Y   f = Y'Y   n

Round 2, after the coordinator broadcasts theta:

    H = Z' diag(u^2) Z   with u = Y - W theta

Sums over sites of these are the pooled matrices, so the coordinator's fit
is the pooled fit. Alignment is by column name; a name present at one site
only gets its own global column (that is how site-local generated
instruments enter). Note what this does and does not hide: a cell of the
global sum to which only one site contributes is that site's own value.
This module is distributed statistical estimation, not privacy-preserving
estimation; see the docs for the disclosure discussion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Design, Layout, Role


@dataclass
class SiteStats:
    site: str
    n: int
    absorbed: int
    z_names: list
    w_names: list
    z_roles: list
    w_roles: list
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    D: np.ndarray
    e: np.ndarray
    f: float

    # --- transport: arrays for the payload, plain python for the metadata (e.g. NVFlare FLModel)
    def arrays(self):
        return {"A": self.A, "B": self.B, "c": self.c, "D": self.D, "e": self.e,
                "f": np.array([self.f]), "n": np.array([self.n], dtype=np.float64),
                "absorbed": np.array([self.absorbed], dtype=np.float64)}

    def meta(self):
        return {"site": self.site, "z_names": list(self.z_names), "w_names": list(self.w_names),
                "z_roles": [Role(r).value for r in self.z_roles], "w_roles": [Role(r).value for r in self.w_roles]}

    @classmethod
    def from_transport(cls, arrays, meta):
        g = lambda k: np.asarray(arrays[k], dtype=np.float64)
        return cls(meta["site"], int(round(float(g("n")[0]))), int(round(float(g("absorbed")[0]))),
                   list(meta["z_names"]), list(meta["w_names"]),
                   [Role(r) for r in meta["z_roles"]], [Role(r) for r in meta["w_roles"]],
                   g("A"), g("B"), g("c"), g("D"), g("e"), float(g("f").ravel()[0]))


def site_stats(d: Design) -> SiteStats:
    Z, W, Y = d.Z, d.W, d.Y
    return SiteStats(d.site, d.n, d.absorbed, list(d.z_names), list(d.w_names), list(d.z_roles), list(d.w_roles),
                     Z.T @ Z, Z.T @ W, Z.T @ Y, W.T @ W, W.T @ Y, float(Y @ Y))


def site_robust_stats(d: Design, theta_by_name: dict) -> np.ndarray:
    """H = Z' diag(u^2) Z with u = Y - W theta, theta aligned to this site's W columns."""
    theta = np.array([theta_by_name[nm] for nm in d.w_names])
    u = d.Y - d.W @ theta
    return d.Z.T @ (d.Z * (u**2)[:, None])


@dataclass
class Stats:
    """The coordinator's sums in the global layout."""
    layout: Layout
    N: int
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    D: np.ndarray
    e: np.ndarray
    f: float


def _union(named_roles):
    names, roles = [], []
    for ns, rs in named_roles:
        for n, r in zip(ns, rs):
            if n not in names:
                names.append(n); roles.append(Role(r))
            elif roles[names.index(n)] != Role(r):
                raise ValueError(f"column {n!r} has role {roles[names.index(n)]} at one site and {r} at another")
    return names, roles


def build_layout(parts: list[SiteStats]) -> Layout:
    z_names, z_roles = _union((p.z_names, p.z_roles) for p in parts)
    w_names, w_roles = _union((p.w_names, p.w_roles) for p in parts)
    return Layout(z_names, w_names, z_roles, w_roles, sum(p.absorbed for p in parts), [p.site for p in parts])


def aggregate(parts: list[SiteStats]) -> Stats:
    """Sum the sites' statistics into the global layout (union of column names, first-use order)."""
    if not parts:
        raise ValueError("no site statistics to aggregate")
    layout = build_layout(parts)
    dz, dw = len(layout.z_names), len(layout.w_names)
    A, B, c = np.zeros((dz, dz)), np.zeros((dz, dw)), np.zeros(dz)
    D, e = np.zeros((dw, dw)), np.zeros(dw)
    f, N = 0.0, 0
    for p in parts:
        zr, wr = layout.z_index(p.z_names), layout.w_index(p.w_names)
        A[np.ix_(zr, zr)] += p.A
        B[np.ix_(zr, wr)] += p.B
        c[zr] += p.c
        D[np.ix_(wr, wr)] += p.D
        e[wr] += p.e
        f += p.f
        N += p.n
    return Stats(layout, N, A, B, c, D, e, f)


def aggregate_robust(parts: list[tuple[list, np.ndarray]], layout: Layout) -> np.ndarray:
    """Sum the sites' H into the global Z layout. parts: [(site z_names, H), ...]."""
    H = np.zeros((len(layout.z_names), len(layout.z_names)))
    for names, Hk in parts:
        zr = layout.z_index(names)
        H[np.ix_(zr, zr)] += Hk
    return H
