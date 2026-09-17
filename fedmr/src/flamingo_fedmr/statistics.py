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
    # optional: the site's own first stage on its original SNPs, so the coordinator can report
    # an F for the SNP set rather than for the single generated instrument
    fs_rss_full: float = None
    fs_rss_reduced: float = None
    fs_n_instruments: int = 0
    fs_n_params: int = 0

    # --- transport: arrays for the payload, plain python for the metadata (e.g. NVFlare FLModel)
    def arrays(self) -> dict:
        scalars = np.array([self.f, self.n, self.absorbed,
                            np.nan if self.fs_rss_full is None else self.fs_rss_full,
                            np.nan if self.fs_rss_reduced is None else self.fs_rss_reduced,
                            self.fs_n_instruments, self.fs_n_params], dtype=np.float64)
        return {"A": self.A, "B": self.B, "c": self.c, "D": self.D, "e": self.e, "scalars": scalars}

    def meta(self) -> dict:
        return {"site": self.site, "z_names": list(self.z_names), "w_names": list(self.w_names),
                "z_roles": [Role(r).value for r in self.z_roles], "w_roles": [Role(r).value for r in self.w_roles]}

    @classmethod
    def from_transport(cls, arrays: dict, meta: dict) -> "SiteStats":
        arr = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}
        f, n, absorbed, rss_full, rss_red, n_inst, n_par = arr["scalars"].tolist()
        return cls(site=meta["site"], n=int(round(n)), absorbed=int(round(absorbed)),
                   z_names=list(meta["z_names"]), w_names=list(meta["w_names"]),
                   z_roles=[Role(r) for r in meta["z_roles"]], w_roles=[Role(r) for r in meta["w_roles"]],
                   A=arr["A"], B=arr["B"], c=arr["c"], D=arr["D"], e=arr["e"], f=float(f),
                   fs_rss_full=None if np.isnan(rss_full) else float(rss_full),
                   fs_rss_reduced=None if np.isnan(rss_red) else float(rss_red),
                   fs_n_instruments=int(round(n_inst)), fs_n_params=int(round(n_par)))


def site_stats(d: Design) -> SiteStats:
    Z, W, Y = d.Z, d.W, d.Y
    fs = d.first_stage_local or {}
    return SiteStats(site=d.site, n=d.n, absorbed=d.absorbed, z_names=list(d.z_names), w_names=list(d.w_names),
                     z_roles=list(d.z_roles), w_roles=list(d.w_roles),
                     A=Z.T @ Z, B=Z.T @ W, c=Z.T @ Y, D=W.T @ W, e=W.T @ Y, f=float(Y @ Y),
                     fs_rss_full=fs.get("rss_full"), fs_rss_reduced=fs.get("rss_reduced"),
                     fs_n_instruments=fs.get("n_instruments", 0), fs_n_params=fs.get("n_params", 0))


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
    first_stage_local: dict = None   # summed local first-stage diagnostics, or None


def _union(named_roles) -> tuple[list, list]:
    """Insertion-ordered union of (name, role) pairs; a name may not change role between sites."""
    roles: dict = {}
    for names, rs in named_roles:
        for n, r in zip(names, rs):
            r = Role(r)
            if roles.setdefault(n, r) != r:
                raise ValueError(f"column {n!r} has role {roles[n]} at one site and {r} at another")
    return list(roles), list(roles.values())


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
    fs = None
    if all(p.fs_rss_full is not None for p in parts):
        fs = {"rss_full": sum(p.fs_rss_full for p in parts), "rss_reduced": sum(p.fs_rss_reduced for p in parts),
              "n_instruments": sum(p.fs_n_instruments for p in parts), "n_params": sum(p.fs_n_params for p in parts)}
    return Stats(layout, N, A, B, c, D, e, f, fs)


def aggregate_robust(parts: list[tuple[list, np.ndarray]], layout: Layout) -> np.ndarray:
    """Sum the sites' H into the global Z layout. parts: [(site z_names, H), ...]."""
    H = np.zeros((len(layout.z_names), len(layout.z_names)))
    for names, Hk in parts:
        zr = layout.z_index(names)
        H[np.ix_(zr, zr)] += Hk
    return H
