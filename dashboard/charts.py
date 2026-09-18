"""Plotly figures for the sensitivity tab.

One palette for the whole tab, matching the figures data/scripts/ already produce:
blue is the instrumented (IV) family, orange the meta-analysis of site estimates,
grey the naive fit without instruments, ink the truth. Robust estimators get their
own fixed slots so a colour never changes meaning between charts.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, MUTED, GRID, SURFACE, MID = "#1f1f1e", "#6b6a63", "#e6e5df", "#ffffff", "#f0efec"
IV, META, NAIVE, TRUTH = BLUE, ORANGE, MUTED, INK
ANCHOR, MINIMAX, VREX, PULSE = VIOLET, MAGENTA, GREEN, YELLOW
DIVERGING = [[0.0, BLUE], [0.5, MID], [1.0, RED]]
SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]

FONT = dict(family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif", size=12, color=INK)


def layout(fig: go.Figure, title: str = "", height: int = 420, xlab: str = "", ylab: str = "",
           legend: bool = True) -> go.Figure:
    # subplot titles sit just above the plotting area, the legend above them and the title
    # at the top of the container, so the three never overlap
    has_subtitles = any(a.yref == "paper" and (a.y or 0) >= 0.99 for a in fig.layout.annotations)
    top = 40 + (30 if title else 0) + (26 if legend else 0) + (22 if has_subtitles else 0)
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=1.0, yanchor="top", yref="container", pad=dict(t=10),
                   font=dict(size=14, color=INK)),
        height=height + top - 40, font=FONT, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=60, r=20, t=top, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0 + (0.08 if has_subtitles else 0.01), xanchor="left", x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        showlegend=legend, hoverlabel=dict(bgcolor=SURFACE, font=dict(color=INK, size=12), bordercolor=GRID),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID, automargin=True,
                     tickfont=dict(color=MUTED), title_text=xlab, title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID, automargin=True,
                     tickfont=dict(color=MUTED), title_text=ylab, title_font=dict(color=MUTED, size=12))
    return fig


def _vline(fig, x, color, dash="dash", name=None, row=None, col=None, width=1.5):
    if x is None or not np.isfinite(x):
        return
    kw = dict(row=row, col=col) if row else {}
    fig.add_vline(x=float(x), line=dict(color=color, dash=dash, width=width), **kw)
    if name:
        fig.add_annotation(x=float(x), y=1.0, yref="paper" if not row else f"y{'' if row == 1 and (col or 1) == 1 else ''} domain",
                           text=name, showarrow=False, font=dict(size=10, color=color), yanchor="bottom",
                           xanchor="left", xshift=3, **kw)


def _hline(fig, y, color, dash="dash", name=None, row=None, col=None):
    if y is None or not np.isfinite(y):
        return
    kw = dict(row=row, col=col) if row else {}
    fig.add_hline(y=float(y), line=dict(color=color, dash=dash, width=1.5),
                  annotation_text=name, annotation_position="top left",
                  annotation_font=dict(size=10, color=color), **kw)


# ------------------------------------------------------------------ forests


def forest(rows: list[dict], names: tuple[str, ...], truth=None, ref=None, title: str = "",
           height: int | None = None, ref_label: str = "all sites") -> go.Figure:
    """Point + 95% CI per row and per parameter. rows: {label, theta, se, color?, symbol?, group?, hover?}."""
    p = len(names)
    fig = make_subplots(rows=1, cols=p, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=[str(n) for n in names] if p > 1 else None)
    labels = [r["label"] for r in rows][::-1]
    groups_seen = set()
    for j in range(p):
        for r in rows:
            g = r.get("group", "")
            show = g not in groups_seen and j == 0 and bool(g)
            groups_seen.add(g) if j == 0 else None
            th, se = float(r["theta"][j]), float(r["se"][j])
            hover = r.get("hover", "")
            fig.add_trace(go.Scatter(
                x=[th], y=[r["label"]], mode="markers", name=g or r["label"], legendgroup=g, showlegend=show,
                marker=dict(color=r.get("color", IV), size=r.get("size", 9), symbol=r.get("symbol", "circle"),
                            line=dict(color=SURFACE, width=1.5)),
                error_x=dict(type="data", array=[1.96 * se], color=r.get("color", IV), thickness=2, width=0),
                hovertemplate=f"<b>{r['label']}</b><br>{names[j]} = {th:.3f}  [{th - 1.96 * se:.3f}, {th + 1.96 * se:.3f}]"
                              + (f"<br>{hover}" if hover else "") + "<extra></extra>",
            ), row=1, col=j + 1)
        if truth is not None and j < len(truth) and truth[j] is not None and np.isfinite(truth[j]):
            fig.add_vline(x=float(truth[j]), line=dict(color=TRUTH, dash="dash", width=1.5), row=1, col=j + 1)
        if ref is not None and j < len(ref) and np.isfinite(ref[j]):
            fig.add_vline(x=float(ref[j]), line=dict(color=META, dash="dot", width=1.5), row=1, col=j + 1)
    fig.update_yaxes(categoryorder="array", categoryarray=labels, showgrid=False)
    layout(fig, title, height or max(300, 26 * len(rows) + 120))
    fig.update_xaxes(title_text="estimate (95% CI); dashed = truth" + (", dotted = " + ref_label if ref is not None else ""),
                     row=1, col=1)
    return fig


def influence(sites: list[str], share: np.ndarray, shift_iv: np.ndarray, shift_ols: np.ndarray, n: np.ndarray,
              F: np.ndarray, title: str) -> go.Figure:
    size = 10 + 30 * np.sqrt(n / n.max())
    fig = go.Figure()
    for shift, color, name in ((shift_ols, NAIVE, "naive fit (no instruments)"), (shift_iv, IV, "2SLS")):
        fig.add_trace(go.Scatter(
            x=share * 100, y=shift, mode="markers", name=name,
            marker=dict(size=size, color=color, opacity=0.85, line=dict(color=SURFACE, width=2)),
            customdata=np.column_stack([sites, n, F]),
            hovertemplate="<b>%{customdata[0]}</b><br>n = %{customdata[1]:,}, mean F = %{customdata[2]:.1f}"
                          "<br>precision share %{x:.1f}%<br>shift when dropped: %{y:.2f} se<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line=dict(color=GRID, width=1.5))
    for y, txt in ((1, "+1 se"), (-1, "−1 se")):
        fig.add_hline(y=y, line=dict(color=MUTED, dash="dot", width=1), annotation_text=txt,
                      annotation_position="top left", annotation_font=dict(color=MUTED, size=10))
    return layout(fig, title, 400, "site's share of the total precision (%)",
                  "change in the estimate when the site is dropped (in full-data se)")


def leave_k(results: dict[int, list[dict]], component: int, name: str, truth, full, title: str) -> go.Figure:
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for k, res in results.items():
        y = np.array([r["theta"][component] for r in res])
        x = k + rng.uniform(-0.18, 0.18, len(y))
        fig.add_trace(go.Box(x=[k] * len(y), y=y, name=f"drop {k}", marker_color=IV, line=dict(color=IV, width=1.5),
                             fillcolor="rgba(42,120,214,0.08)", boxpoints=False, showlegend=False, width=0.5))
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=f"drop {k}", showlegend=False,
                                 marker=dict(color=IV, size=6, opacity=0.6, line=dict(color=SURFACE, width=1)),
                                 customdata=[", ".join(r["dropped"]) for r in res],
                                 hovertemplate="dropped %{customdata}<br>" + name + " = %{y:.3f}<extra></extra>"))
    _hline(fig, truth, TRUTH, "dash", "truth")
    _hline(fig, full, META, "dot", "all sites")
    fig.update_xaxes(tickmode="array", tickvals=list(results), ticktext=[f"drop {k}" for k in results], showgrid=False)
    return layout(fig, title, 380, "", name, legend=False)


def cumulative(labels: list[str], theta: np.ndarray, se: np.ndarray, truth, name: str, title: str) -> go.Figure:
    x = list(range(len(labels)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(theta + 1.96 * se) + list(theta - 1.96 * se)[::-1], fill="toself",
                             fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=theta, mode="lines+markers", name="running 2SLS (95% band)",
                             line=dict(color=IV, width=2), marker=dict(size=8, line=dict(color=SURFACE, width=1.5)),
                             customdata=labels, hovertemplate="+ %{customdata}<br>" + name + " = %{y:.3f}<extra></extra>"))
    _hline(fig, truth, TRUTH, "dash", "truth")
    fig.update_xaxes(tickmode="array", tickvals=x, ticktext=labels, tickangle=-45, showgrid=False)
    return layout(fig, title, 380, "sites added, in this order", name)


def funnel(sites: list[str], theta: np.ndarray, se: np.ndarray, fe: float, truth, name: str, title: str) -> go.Figure:
    prec = 1 / se
    fig = go.Figure()
    s_grid = np.linspace(se.min() * 0.8, se.max() * 1.15, 50)
    for sign in (-1, 1):
        fig.add_trace(go.Scatter(x=fe + sign * 1.96 * s_grid, y=1 / s_grid, mode="lines", showlegend=False,
                                 line=dict(color=MUTED, dash="dot", width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=theta, y=prec, mode="markers", name="site 2SLS",
                             marker=dict(color=IV, size=11, line=dict(color=SURFACE, width=2)), customdata=sites,
                             hovertemplate="<b>%{customdata}</b><br>" + name + " = %{x:.3f}<br>precision %{y:.1f}<extra></extra>"))
    _vline(fig, fe, META, "dot", "fixed effect")
    _vline(fig, truth, TRUTH, "dash", "truth")
    return layout(fig, title, 380, name, "precision (1 / se)")


def heatmap(z: np.ndarray, x: list[str], y: list[str], title: str, diverging: bool = True, zmax: float | None = None,
            hover: str = "%{y} · %{x}<br>z = %{z:.2f}", height: int | None = None, colorbar: str = "",
            text: np.ndarray | None = None) -> go.Figure:
    if diverging:
        zmax = zmax or float(np.nanmax(np.abs(z))) or 1.0
        scale, zmin = DIVERGING, -zmax
    else:
        zmax = zmax or float(np.nanmax(z)) or 1.0
        scale, zmin = SEQUENTIAL, 0.0
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=scale, zmin=zmin, zmax=zmax, xgap=2, ygap=2,
                               hovertemplate=hover + "<extra></extra>", colorbar=dict(title=colorbar, thickness=10, len=0.8),
                               text=text, texttemplate="%{text}" if text is not None else None, textfont=dict(size=10)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    return layout(fig, title, height or max(260, 22 * len(y) + 120), legend=False)


# ------------------------------------------------------------ regularisation


def kclass(kp: dict, names: tuple[str, ...], truth) -> go.Figure:
    p = len(names)
    fig = make_subplots(rows=p + 1, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[1.0] * p + [0.7],
                        subplot_titles=[f"{n}(κ)" for n in names] + ["instrument test statistic along the path"])
    k = kp["kappa"]
    colors = {"OLS": NAIVE, "PULSE": PULSE, "2SLS": IV, "Fuller(1)": AQUA, "LIML": VIOLET}
    for j in range(p):
        fig.add_trace(go.Scatter(x=k, y=kp["theta"][:, j], mode="lines", line=dict(color=IV, width=2), showlegend=False,
                                 hovertemplate="κ = %{x:.3f}<br>" + names[j] + " = %{y:.3f}<extra></extra>"), row=j + 1, col=1)
        for name, (kv, th, _) in kp["marks"].items():
            fig.add_trace(go.Scatter(x=[kv], y=[th[j]], mode="markers", name=name, legendgroup=name, showlegend=(j == 0),
                                     marker=dict(color=colors[name], size=11, symbol="diamond" if name == "PULSE" else "circle",
                                                 line=dict(color=SURFACE, width=1.5)),
                                     hovertemplate=f"<b>{name}</b> κ = {kv:.3f}<br>{names[j]} = {th[j]:.3f}<extra></extra>"),
                          row=j + 1, col=1)
        if truth is not None and j < len(truth) and np.isfinite(truth[j]):
            fig.add_hline(y=float(truth[j]), line=dict(color=TRUTH, dash="dash", width=1.5), row=j + 1, col=1)
    fig.add_trace(go.Scatter(x=k, y=kp["T"], mode="lines", line=dict(color=MUTED, width=2), showlegend=False,
                             hovertemplate="κ = %{x:.3f}<br>T = %{y:.1f}<extra></extra>"), row=p + 1, col=1)
    fig.add_hline(y=kp["crit"], line=dict(color=RED, dash="dot", width=1.5), row=p + 1, col=1,
                  annotation_text=f"χ²({kp['df']}) {100 * (1 - kp['alpha']):.0f}% = {kp['crit']:.1f}",
                  annotation_position="top right", annotation_font=dict(size=10, color=RED))
    layout(fig, "", 200 * p + 200)
    fig.update_xaxes(title_text="κ  (0 = OLS, 1 = 2SLS, κ_LIML = LIML)", row=p + 1, col=1)
    fig.update_yaxes(title_text="n·R² of residual on instruments", row=p + 1, col=1)
    return fig


def path(x: np.ndarray, thetas: np.ndarray, names: tuple[str, ...], truth, xlab: str, title: str, color=IV,
         logx: bool = True, marks: dict | None = None, extra: dict | None = None, ref: np.ndarray | None = None,
         ref_label: str = "") -> go.Figure:
    """One panel per parameter: theta_j as a function of a tuning parameter, with optional
    marker points {name: (x, theta)} and a second path `extra` {name: (color, thetas)}."""
    p = len(names)
    fig = make_subplots(rows=1, cols=p, horizontal_spacing=0.08, subplot_titles=list(names) if p > 1 else None)
    for j in range(p):
        fig.add_trace(go.Scatter(x=x, y=thetas[:, j], mode="lines", name="path", legendgroup="path", showlegend=(j == 0),
                                 line=dict(color=color, width=2),
                                 hovertemplate=xlab + " = %{x:.3g}<br>" + names[j] + " = %{y:.3f}<extra></extra>"), row=1, col=j + 1)
        if extra:
            for name, (c, th) in extra.items():
                fig.add_trace(go.Scatter(x=x, y=th[:, j], mode="lines", name=name, legendgroup=name, showlegend=(j == 0),
                                         line=dict(color=c, width=2, dash="dot"),
                                         hovertemplate=xlab + " = %{x:.3g}<br>" + names[j] + " = %{y:.3f}<extra>" + name + "</extra>"),
                              row=1, col=j + 1)
        if marks:
            for name, (mx, mth, mcolor) in marks.items():
                fig.add_trace(go.Scatter(x=[mx], y=[mth[j]], mode="markers", name=name, legendgroup=name, showlegend=(j == 0),
                                         marker=dict(color=mcolor, size=11, line=dict(color=SURFACE, width=1.5)),
                                         hovertemplate=f"<b>{name}</b><br>{names[j]} = {mth[j]:.3f}<extra></extra>"), row=1, col=j + 1)
        if truth is not None and j < len(truth) and np.isfinite(truth[j]):
            fig.add_hline(y=float(truth[j]), line=dict(color=TRUTH, dash="dash", width=1.5), row=1, col=j + 1)
        if ref is not None and np.isfinite(ref[j]):
            fig.add_hline(y=float(ref[j]), line=dict(color=META, dash="dot", width=1.5), row=1, col=j + 1,
                          annotation_text=ref_label, annotation_position="bottom right", annotation_font=dict(size=10, color=META))
        if logx:
            fig.update_xaxes(type="log", row=1, col=j + 1)
        fig.update_xaxes(title_text=xlab, row=1, col=j + 1)
    return layout(fig, title, 380)


# ------------------------------------------------------- robustness across sites


def surprise_fan(grid: np.ndarray, curves: np.ndarray, sites: list[str], minimax_x: float, pooled_x: float, truth,
                 name: str) -> go.Figure:
    fig = go.Figure()
    for i, s in enumerate(sites):
        fig.add_trace(go.Scatter(x=grid, y=curves[i], mode="lines", name=s, showlegend=False,
                                 line=dict(color=IV, width=1), opacity=0.35,
                                 hovertemplate=f"<b>{s}</b><br>{name} = %{{x:.3f}}<br>z² = %{{y:.2f}}<extra></extra>"))
    env = curves.max(axis=0)
    fig.add_trace(go.Scatter(x=grid, y=env, mode="lines", name="most surprised site (max over sites)",
                             line=dict(color=INK, width=2.5), hovertemplate=name + " = %{x:.3f}<br>max z² = %{y:.2f}<extra></extra>"))
    fig.add_hline(y=3.84, line=dict(color=RED, dash="dot", width=1.5), annotation_text="z² = 3.84 (5%)",
                  annotation_position="top right", annotation_font=dict(size=10, color=RED))
    _vline(fig, minimax_x, MINIMAX, "solid", "minimax")
    _vline(fig, pooled_x, IV, "dot", "pooled 2SLS")
    _vline(fig, truth, TRUTH, "dash", "truth")
    fig.update_yaxes(range=[0, min(float(env.max()), 12)])
    return layout(fig, "How surprised would each site be by a candidate value? (Wald z² per site)", 400, name, "z² of the site against the candidate")


def surprise_contour(t1: np.ndarray, t2: np.ndarray, Z: np.ndarray, points: dict, site_pts: np.ndarray, sites: list[str],
                     names: tuple[str, str]) -> go.Figure:
    fig = go.Figure()
    cap = 12.0
    fig.add_trace(go.Contour(x=t1, y=t2, z=np.minimum(Z, cap), colorscale=SEQUENTIAL, zmin=0, zmax=cap,
                             contours=dict(start=0, end=cap, size=1, coloring="heatmap"),
                             line=dict(width=0.5, color=SURFACE), colorbar=dict(title="max z² (capped)", thickness=10, len=0.8),
                             hovertemplate=names[0] + " = %{x:.3f}<br>" + names[1] + " = %{y:.3f}<br>max z² = %{z:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=site_pts[:, 0], y=site_pts[:, 1], mode="markers", name="site estimates",
                             marker=dict(color=SURFACE, size=7, line=dict(color=IV, width=2)), customdata=sites,
                             hovertemplate="<b>%{customdata}</b><br>" + names[0] + " = %{x:.3f}, " + names[1] + " = %{y:.3f}<extra></extra>"))
    for name, (pt, color, symbol) in points.items():
        if pt is None or not np.all(np.isfinite(pt)):
            continue
        fig.add_trace(go.Scatter(x=[pt[0]], y=[pt[1]], mode="markers", name=name,
                                 marker=dict(color=color, size=13, symbol=symbol, line=dict(color=SURFACE, width=1.5)),
                                 hovertemplate=f"<b>{name}</b><br>{names[0]} = {pt[0]:.3f}, {names[1]} = {pt[1]:.3f}<extra></extra>"))
    return layout(fig, "Largest site Wald statistic over the parameter plane", 460, names[0], names[1])


# ------------------------------------------------------------- certificate


def certificate(cert: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.32, 0.68], horizontal_spacing=0.1,
                        subplot_titles=["J decomposition: observed vs expected", "per-site over-identification (Sargan) statistic"])
    parts = [("within sites (instrument validity)", cert["within"], cert["df_within"], BLUE, cert["p_within"]),
             ("between sites (same θ everywhere)", cert["between"], cert["df_between"], ORANGE, cert["p_between"])]
    for name, val, df, color, pv in parts:
        fig.add_trace(go.Bar(x=["observed J"], y=[val], name=name, marker_color=color, marker_line=dict(color=SURFACE, width=2),
                             hovertemplate=f"<b>{name}</b><br>J = {val:.1f} on {df} df<br>p = {pv:.3g}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=["expected (df)"], y=[df], name=name, showlegend=False, marker_color=color, opacity=0.35,
                             marker_line=dict(color=SURFACE, width=2), hovertemplate=f"{name}<br>df = {df}<extra></extra>"), row=1, col=1)
    rows = cert["sites"]
    fig.add_trace(go.Bar(x=[r["site"] for r in rows], y=[r["J"] for r in rows], name=parts[0][0], showlegend=False,
                         marker_color=BLUE, marker_line=dict(color=SURFACE, width=2),
                         customdata=np.array([[r["df"], r["p"], r["between"]] for r in rows]),
                         hovertemplate="<b>%{x}</b><br>Sargan J = %{y:.1f} on %{customdata[0]} df<br>p = %{customdata[1]:.3f}"
                                       "<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Bar(x=[r["site"] for r in rows], y=[r["between"] for r in rows], name=parts[1][0], showlegend=False,
                         marker_color=ORANGE, marker_line=dict(color=SURFACE, width=2),
                         hovertemplate="<b>%{x}</b><br>(θ_e − θ)' V⁻¹ (θ_e − θ) = %{y:.2f}<extra></extra>"), row=1, col=2)
    if rows and rows[0]["df"] > 0:
        from scipy import stats as sps
        crit = sps.chi2.ppf(0.95, rows[0]["df"])
        fig.add_hline(y=crit, line=dict(color=RED, dash="dot", width=1.5), row=1, col=2,
                      annotation_text=f"χ²({rows[0]['df']}) 95%", annotation_position="top right", annotation_font=dict(size=10, color=RED))
    fig.update_layout(barmode="stack")
    return layout(fig, "", 400)


def snp_bars(T: np.ndarray, crit: float, removed: list[int], injected: list[int], title: str) -> go.Figure:
    L = len(T)
    colors = [ORANGE if j in removed else IV for j in range(L)]
    fig = go.Figure(go.Bar(x=[f"snp{j}" for j in range(L)], y=T, marker_color=colors, marker_line=dict(color=SURFACE, width=2),
                           customdata=[("removed by the search" if j in removed else "kept") + (" · pleiotropy injected" if j in injected else "")
                                       for j in range(L)],
                           hovertemplate="<b>%{x}</b><br>T = %{y:.1f}<br>%{customdata}<extra></extra>"))
    fig.add_hline(y=crit, line=dict(color=RED, dash="dot", width=1.5), annotation_text="χ²(K) 95%",
                  annotation_position="top right", annotation_font=dict(size=10, color=RED))
    for j in injected:
        fig.add_annotation(x=f"snp{j}", y=T[j], text="▼", showarrow=False, yshift=12, font=dict(color=RED, size=11))
    return layout(fig, title, 340, "", "T_j = Σ_sites z²", legend=False)


def search_path(path_rows: list[dict], truth, name: str) -> go.Figure:
    x = [r["size"] for r in path_rows]
    th = np.array([r["theta"] for r in path_rows]); se = np.array([r["se"] for r in path_rows])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.08,
                        subplot_titles=[f"{name} as the sites' disagreements are removed", "J-test p-value of the remaining set"])
    fig.add_trace(go.Scatter(x=x, y=th, mode="lines+markers", name=name, line=dict(color=IV, width=2),
                             marker=dict(size=9, line=dict(color=SURFACE, width=1.5)),
                             error_y=dict(type="data", array=1.96 * se, color=IV, thickness=1.5, width=0),
                             customdata=[("removed snp%d" % r["removed"]) if r["removed"] is not None else "all SNPs" for r in path_rows],
                             hovertemplate="%{customdata}<br>set size %{x}<br>" + name + " = %{y:.3f}<extra></extra>"), row=1, col=1)
    if truth is not None and np.isfinite(truth):
        fig.add_hline(y=float(truth), line=dict(color=TRUTH, dash="dash", width=1.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["p"] for r in path_rows], mode="lines+markers", name="p (J)", line=dict(color=MUTED, width=2),
                             marker=dict(size=8), hovertemplate="set size %{x}<br>p = %{y:.3g}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0.05, line=dict(color=RED, dash="dot", width=1.5), row=2, col=1)
    fig.update_xaxes(autorange="reversed", title_text="number of SNPs kept", row=2, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    return layout(fig, "", 420, legend=False)


def icp_heatmap(results: list[dict], sites: list[str], title: str, key: str = "p") -> go.Figure:
    z = np.array([[-np.log10(max(r[key], 1e-6)) for r in res["sites"]] for res in results])
    ylabels = [("✓ " if res["accepted"] else "✗ ") + ("{ " + ", ".join(res["set"]) + " }" if res["set"] else "∅") for res in results]
    fig = heatmap(z, sites, ylabels, title, diverging=False, zmax=6, colorbar="−log10 p",
                  hover="%{y} vs %{x}<br>−log10 p = %{z:.2f}", height=max(260, 26 * len(results) + 130))
    fig.add_vline(x=-0.5, line=dict(width=0))
    return fig


def dumbbell(sites: list[str], ols: np.ndarray, iv: np.ndarray, truth, hover_extra: list[str], name: str) -> go.Figure:
    fig = go.Figure()
    for s, o, v in zip(sites, ols, iv):
        fig.add_trace(go.Scatter(x=[o, v], y=[s, s], mode="lines", line=dict(color=GRID, width=3), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ols, y=sites, mode="markers", name="naive slope (no instruments)",
                             marker=dict(color=NAIVE, size=10, line=dict(color=SURFACE, width=1.5)), customdata=hover_extra,
                             hovertemplate="<b>%{y}</b><br>naive = %{x:.3f}<br>%{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=iv, y=sites, mode="markers", name="site 2SLS",
                             marker=dict(color=IV, size=10, line=dict(color=SURFACE, width=1.5)), customdata=hover_extra,
                             hovertemplate="<b>%{y}</b><br>2SLS = %{x:.3f}<br>%{customdata}<extra></extra>"))
    _vline(fig, truth, TRUTH, "dash", "truth")
    fig.update_yaxes(categoryorder="array", categoryarray=sites[::-1], showgrid=False)
    return layout(fig, "Which estimator is invariant across sites? Naive slope vs 2SLS, per site", max(300, 26 * len(sites) + 120), name)
