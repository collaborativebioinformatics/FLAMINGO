"""The sensitivity tab: leave-sites-out, an overview of the estimators and robustness across
sites, all computed live from per-site sufficient statistics (sensitivity.py) and drawn with
Plotly (charts.py). Nothing here writes to disk; what-if perturbations live in memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st
from scipy import stats as sps

import charts as ch
import runner
import sensitivity as S

FALLBACK = runner.DATA_DIR / "simulated_data" / "federated"


@st.cache_data(show_spinner="Computing per-site sufficient statistics…")
def load_stats(folder: str, stamp: float, pert_key: str) -> list[S.SiteStats]:
    d = json.loads(pert_key)
    d["pleio_snps"] = tuple(d["pleio_snps"])
    d["pleio_sites"] = d["pleio_sites"] if isinstance(d["pleio_sites"], str) else tuple(d["pleio_sites"])
    return S.load_sites(Path(folder), S.Perturbation(**d))


def _fmt(theta: np.ndarray, se: np.ndarray | None = None) -> str:
    if se is None:
        return ", ".join(f"{t:.3f}" for t in theta)
    return ", ".join(f"{t:.3f} ± {1.96 * s:.3f}" for t, s in zip(theta, se))


def _plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]})


def _pvalue(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.3f}"


def perturbation_controls(site_ids: list[str], n_snps: int) -> S.Perturbation:
    with st.expander("What-if perturbations — change the data in memory and watch every analysis react"):
        st.caption("Nothing on disk changes. Use these to see which analysis catches which violation: "
                   "pleiotropy (invalid instruments), a site with a different causal effect, a direct site effect "
                   "on the outcome level, or one site with extra confounding.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pleiotropic SNPs** — a direct effect on Y, treating SNP *j* as the same variant at every site")
            pleio = st.multiselect("SNP indices", list(range(n_snps)), default=[], key="wi_pleio")
            alpha = st.slider("Direct effect α per allele", 0.0, 0.5, 0.15, 0.01, key="wi_alpha")
            where = st.selectbox("At which sites", ["all"] + site_ids, key="wi_pleio_sites")
            st.markdown("**Deviant site** — its causal slope is θ + δ")
            dev = st.selectbox("Site", ["none"] + site_ids, key="wi_dev")
            delta = st.slider("δ", -0.6, 0.6, 0.3, 0.05, key="wi_delta")
        with c2:
            st.markdown("**Outcome level shift** — a direct site effect on Y, absorbed by the site intercept")
            sh = st.selectbox("Site", ["none"] + site_ids, key="wi_shift")
            c = st.slider("Shift", -2.0, 2.0, 1.0, 0.1, key="wi_c")
            st.markdown("**Extra confounding** — one site's outcome gets an extra c·U")
            cf = st.selectbox("Site", ["none"] + site_ids, key="wi_conf")
            cc = st.slider("c", 0.0, 1.5, 0.6, 0.05, key="wi_cc")
    return S.Perturbation(
        pleio_snps=tuple(pleio), pleio_alpha=alpha, pleio_sites="all" if where == "all" else (where,),
        deviant_site=None if dev == "none" else dev, deviant_delta=delta,
        shift_site=None if sh == "none" else sh, shift_c=c,
        confound_site=None if cf == "none" else cf, confound_c=cc)


def render(params: dict, paths: runner.RunPaths, stat_card, stat_row) -> None:
    shape = params["shape"]
    if shape not in S.SUPPORTED_SHAPES:
        st.info("The sufficient-statistics analyses need a continuous outcome; the Cox shape is not covered yet. "
                "Switch the causal shape to linear, quadratic or threshold.")
        return
    folder = paths.sites_for(shape)
    if not (folder / "manifest.json").exists():
        folder = FALLBACK / shape
        if not (folder / "manifest.json").exists():
            st.info("No sites for this parameter set yet — generate them in tab 1, or run the experiments.")
            return
        st.caption(f"No sites generated for this parameter set yet; showing the committed federation in "
                   f"`{folder.relative_to(runner.REPO)}` instead.")
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["shape"] = shape
    site_ids = [s["site_id"] for s in manifest["sites"]]
    n_snps = int(manifest.get("n_snps", 20))

    st.markdown("Every estimate on this tab is a function of one small bundle per site — n, column means and the "
                "site-centred cross products of (Y, X, X², x̂, x̂²) plus G′G and G′(Y, X) — so leaving sites out, "
                "re-weighting and comparing sites never touches the rows again. That bundle is what "
                "a federation exchanges.")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    model = c1.radio("Causal curve fitted", ["linear", "quadratic"], horizontal=True,
                     index=1 if shape in runner.CURVED_SHAPES else 0,
                     help="linear: Y = θX. quadratic: Y = θ1 X + θ2 X², instrumented by (x̂, x̂²).")
    alpha = c2.select_slider("Test level α", [0.01, 0.05, 0.10], value=0.05)
    n_show = c3.slider("Leave-k-out: largest k", 1, 3, 2, help="All subsets up to this size (random 400 when there are more).")
    pert = perturbation_controls(site_ids, n_snps)
    if pert.active():
        st.warning("What-if perturbation active — the data below differ from the simulated federation.", icon="⚠️")

    stamp = (folder / "manifest.json").stat().st_mtime
    sites = load_stats(str(folder), stamp, pert.key())
    names = S.MODELS[model]["names"]
    truth = S.truth_for(sites, manifest, model)
    ids = [s.site for s in sites]
    n_arr = np.array([s.n for s in sites], float)
    F_arr = np.array([s.mean_F for s in sites])

    A = S.aggregate(sites, model)
    full = S.two_sls(A, "pooled 2SLS")
    naive = S.ols(A)
    ests = S.per_site(sites, model)
    ests_ols = S.per_site_ols(sites, model)
    het = S.heterogeneity(ests, 0)
    ivw = S.ivw_meta(ests)
    equal = S.ivw_meta(ests, 0.0)
    mm = S.minimax(ests, full.theta)
    cert_instr = "snps" if model == "linear" else "basis"

    drop1 = S.leave_out(sites, model, 1)
    shifts = np.array([(full.theta[0] - d["theta"][0]) / full.se[0] for d in drop1])
    worst = int(np.argmax(np.abs(shifts)))

    tabs = st.tabs(["Leave sites out", "Overview", "Robust across sites"])

    # ------------------------------------------------------------------ overview
    with tabs[1]:
        cert = S.certificate(sites, model, cert_instr)
        cards = [stat_card("Pooled 2SLS " + names[0], f"{full.theta[0]:.3f}", f"95% CI ± {1.96 * full.se[0]:.3f}", "pink"),
                 stat_card("Truth " + names[0], f"{truth[0]:.3f}" if truth is not None and np.isfinite(truth[0]) else "—",
                           "what the fit targets", "neutral"),
                 stat_card("Largest drop-1 shift", f"{shifts[worst]:+.2f} se", f"dropping {drop1[worst]['dropped'][0]}",
                           "coral" if abs(shifts[worst]) > 1 else "neutral"),
                 stat_card("Between-site test", _pvalue(cert["p_between"]), f"same θ everywhere · Q {het['Q']:.1f} on {het['df']} df",
                           "coral" if cert["p_between"] < alpha else "neutral")]
        if cert["df_within"] > 0:
            cards.append(stat_card("Within-site test", _pvalue(cert["p_within"]), f"instrument validity · {cert['df_within']} df",
                                   "coral" if cert["p_within"] < alpha else "neutral"))
        cards.append(stat_card("I² across sites", f"{100 * het['I2']:.0f}%", f"τ = {het['tau']:.3f}", "neutral"))
        stat_row(cards)

        kp = S.kclass_path(A, alpha)
        rows = [
            {"label": "pooled 2SLS (site fixed effects)", "theta": full.theta, "se": full.se, "color": ch.IV, "group": "instrumented"},
            {"label": "IVW meta-analysis of site 2SLS", "theta": ivw.theta, "se": ivw.se, "color": ch.META, "group": "meta-analysis",
             "symbol": "diamond"},
            {"label": "random effects (DerSimonian–Laird)", "theta": np.array([het["re"]] + list(ivw.theta[1:])),
             "se": np.array([het["re_se"]] + list(ivw.se[1:])), "color": ch.META, "group": "meta-analysis", "symbol": "diamond",
             "hover": "θ2 shown as the fixed-effect value" if len(names) > 1 else ""},
            {"label": "equal-weight meta-analysis", "theta": equal.theta, "se": equal.se, "color": ch.META, "group": "meta-analysis",
             "symbol": "diamond"},
            {"label": "minimax over sites (Chebyshev centre)", "theta": mm, "se": np.zeros_like(mm), "color": ch.MINIMAX,
             "group": "robust", "symbol": "square"},
            {"label": "PULSE" + (" (test rejects even 2SLS)" if kp["pulse_rejected"] else ""), "theta": kp["marks"]["PULSE"][1],
             "se": np.zeros(len(names)), "color": ch.PULSE, "group": "K-class", "symbol": "triangle-up"},
            {"label": "LIML", "theta": kp["marks"]["LIML"][1], "se": np.zeros(len(names)), "color": ch.VIOLET, "group": "K-class",
             "symbol": "triangle-up"},
            {"label": "naive fit (no instruments)", "theta": naive.theta, "se": naive.se, "color": ch.NAIVE, "group": "confounded"},
        ]
        _plot(ch.forest(rows, names, truth, None, "Where the estimators land — one row per way of combining the sites",
                        height=380))
        st.caption("Squares and triangles have no standard error on this chart (the robust and K-class point estimates); "
                   "the dashed line is the truth.")
        with st.expander("Table"):
            st.dataframe(pl.DataFrame([{"estimator": r["label"], **{n: float(r["theta"][j]) for j, n in enumerate(names)},
                                        **{f"{n} se": float(r["se"][j]) for j, n in enumerate(names)}} for r in rows]),
                         width="stretch", hide_index=True)
        _plot(ch.dumbbell(ids, np.array([e.theta[0] for e in ests_ols]), np.array([e.theta[0] for e in ests]),
                          truth[0] if truth is not None else None,
                          [f"γx = {s.truth.get('gamma_x', float('nan')):.2f}, γy = {s.truth.get('gamma_y', float('nan')):.2f}, "
                           f"h²x = {s.truth.get('h2_x', float('nan')):.3f}" for s in sites], names[0]))
        st.caption("The naive slope moves with each site's confounding (γx·γy) while the instrumented slope stays put: "
                   "invariance across environments is what the instruments buy.")

    # --------------------------------------------------------------- leave out
    with tabs[0]:
        st.markdown("#### Drop one site")
        rows = [{"label": f"− {d['dropped'][0]}", "theta": d["theta"], "se": d["se"], "color": ch.IV,
                 "hover": f"n left {d['n']:,}"} for d in drop1]
        _plot(ch.forest(rows, names, truth, full.theta, "Estimate with one site removed (dotted = all sites)", ref_label="all sites"))
        share = np.array([1 / e.cov[0, 0] for e in ests]); share = share / share.sum()
        drop1_ols = S.leave_out(sites, model, 1, estimator=S.ols)
        shift_ols = np.array([(naive.theta[0] - d["theta"][0]) / naive.se[0] for d in drop1_ols])
        _plot(ch.influence(ids, share, shifts, shift_ols, n_arr, F_arr,
                           f"Influence of each site on {names[0]}: bubble area is site size"))
        st.caption("A site far from zero moves the answer when it leaves. Under confounding that differs by site, "
                   "the naive fit's influence points scatter while the instrumented ones stay near zero.")

        st.markdown("#### Drop several sites")
        results = {k: S.leave_out(sites, model, k) for k in range(1, n_show + 1)}
        cols = st.columns(len(names))
        for j, nm in enumerate(names):
            with cols[j]:
                _plot(ch.leave_k(results, j, nm, truth[j] if truth is not None else None, full.theta[j],
                                 f"{nm}: every subset with k sites removed"))

        st.markdown("#### Add sites one at a time")
        order_by = st.radio("Order", ["site id", "smallest first", "largest first", "weakest instruments first",
                                      "strongest instruments first"], horizontal=True)
        key = {"site id": np.arange(len(sites)), "smallest first": np.argsort(n_arr), "largest first": np.argsort(-n_arr),
               "weakest instruments first": np.argsort(F_arr), "strongest instruments first": np.argsort(-F_arr)}[order_by]
        cum = S.cumulative(sites, model, list(key))
        cols = st.columns(len(names))
        for j, nm in enumerate(names):
            with cols[j]:
                _plot(ch.cumulative([e.label for e in cum], np.array([e.theta[j] for e in cum]), np.array([e.se[j] for e in cum]),
                                    truth[j] if truth is not None else None, nm, f"Cumulative estimate of {nm}"))

        st.markdown("#### Sites as a meta-analysis")
        stat_row([stat_card("Cochran Q", f"{het['Q']:.1f}", f"on {het['df']} df · p = {_pvalue(het['p'])}",
                            "coral" if het["p"] < alpha else "neutral"),
                  stat_card("I²", f"{100 * het['I2']:.0f}%", "share of variation beyond chance", "neutral"),
                  stat_card("τ (between-site sd)", f"{het['tau']:.3f}", "DerSimonian–Laird", "neutral"),
                  stat_card("Prediction interval", f"[{het['pi_lo']:.2f}, {het['pi_hi']:.2f}]", "a new site's " + names[0], "pink")])
        rows = [{"label": s.site, "theta": e.theta, "se": e.se, "color": ch.IV, "group": "site 2SLS",
                 "hover": f"n = {s.n:,}, mean F = {s.mean_F:.1f}"} for s, e in zip(sites, ests)]
        rows += [{"label": "fixed effect (IVW)", "theta": ivw.theta, "se": ivw.se, "color": ch.META, "symbol": "diamond", "size": 12,
                  "group": "combined"},
                 {"label": "random effects", "theta": np.array([het["re"]] + list(ivw.theta[1:])),
                  "se": np.array([het["re_se"]] + list(ivw.se[1:])), "color": ch.META, "symbol": "diamond", "size": 12, "group": "combined"},
                 {"label": "pooled 2SLS", "theta": full.theta, "se": full.se, "color": ch.INK, "symbol": "square", "size": 11, "group": "combined"}]
        _plot(ch.forest(rows, names, truth, None, "Per-site estimates and the combined rows"))
        _plot(ch.funnel(ids, np.array([e.theta[0] for e in ests]), np.array([e.se[0] for e in ests]), het["fe"],
                        truth[0] if truth is not None else None, names[0], "Funnel: precision against estimate"))
        st.caption("Which sites disagree with which, and with the joint model, is on the Robust across sites tab.")

        st.markdown("#### Weak-instrument filter")
        f_min = st.slider("Keep sites with mean per-SNP F at least", 0.0, float(np.ceil(F_arr.max())), 0.0, 1.0)
        keep = [s for s, f in zip(sites, F_arr) if f >= f_min]
        if len(keep) >= 1:
            e_keep = S.two_sls(S.aggregate(keep, model))
            stat_row([stat_card("Sites kept", f"{len(keep)} / {len(sites)}", f"n = {sum(s.n for s in keep):,}", "pink"),
                      stat_card(names[0], f"{e_keep.theta[0]:.3f}", f"95% CI ± {1.96 * e_keep.se[0]:.3f}", "pink")]
                     + ([stat_card(names[1], f"{e_keep.theta[1]:.3f}", f"95% CI ± {1.96 * e_keep.se[1]:.3f}", "pink")] if len(names) > 1 else []))
        else:
            st.info("No site passes that threshold.")

    # ------------------------------------------------------------- robust
    with tabs[2]:
        st.markdown("#### Which sites disagree, and with whom")
        st.markdown("Fit the **joint model** to the whole federation at once: site intercepts, one causal slope per site and, "
                    "by default, one error variance for everyone. It puts every site's slope on a common noise scale and gives, "
                    "for each site, the Wald test of its slope against the slope fitted on all the *other* sites — the site × X "
                    "interaction. That test is the last column; the cells to its left compare pairs of sites on the same scale.")
        scale_choice = st.radio("Noise scale", ["joint model (one σ² for the federation)", "each site's own σ² (meta-analysis view)"],
                                horizontal=True, key="rob_scale")
        dis = S.site_disagreement(sites, model, scale="site" if scale_choice.startswith("each") else "joint")
        K = len(sites)
        crit = float(sps.norm.ppf(1 - alpha / (2 * K)))
        zj = dis["z_joint"][:, 0]
        top = int(np.nanargmax(np.abs(zj)))
        flagged = int(np.sum(np.abs(dis["z_joint"]) > crit))
        sd_site = np.sqrt(dis["sigma2_site"])
        stat_row([stat_card("Most deviant site", ids[top], f"z = {zj[top]:+.2f} against the joint fit of the others",
                            "coral" if abs(zj[top]) > crit else "neutral"),
                  stat_card("Sites flagged", f"{flagged} / {K}", f"|z| > {crit:.2f} · α = {alpha:g}, Bonferroni over sites",
                            "coral" if flagged else "neutral"),
                  stat_card("Equal slopes test", _pvalue(dis["p"]), f"joint model · Q {dis['Q']:.1f} on {dis['df']} df",
                            "coral" if dis["p"] < alpha else "neutral"),
                  stat_card("Residual sd", f"{np.sqrt(dis['sigma2']):.3f}", f"joint · sites range {sd_site.min():.3f}–{sd_site.max():.3f}",
                            "neutral")])
        for j, nm in enumerate(names):
            z = np.concatenate([dis["z_pair"][:, :, j], dis["z_joint"][:, [j]]], axis=1)
            z[np.arange(K), np.arange(K)] = np.nan
            text = np.where(np.isfinite(z), np.vectorize(lambda v: f"{v:+.1f}")(np.nan_to_num(z)), "")
            _plot(ch.heatmap(z, ids + ["joint model"], ids, f"Disagreement in {nm}: z of the row site minus the column", zmax=4,
                             hover="%{y} − %{x}<br>z = %{z:.2f}", colorbar="z", text=text, height=max(420, 40 * K + 160),
                             vsep=K - 0.5))
        st.caption("A row coloured all the way across is a site the federation disagrees with; one off-colour cell is a pair. "
                   "Give one site a deviant slope (what-if) and its row lights up, ending in the joint-model column. Give it extra "
                   "confounding instead and its own σ² inflates: on its own scale the site looks agreeable, on the joint scale it does not "
                   "get to hide behind its noise — switch the scale to compare.")

        st.markdown("#### The estimate no site objects to")
        st.markdown("Each site's Wald statistic z²(θ) says how surprised it is by a candidate θ. The pooled 2SLS minimises the "
                    "precision-weighted sum; the **minimax** estimate minimises the largest one, so the most sceptical site is as "
                    "content as possible. That is group-DRO with each site's own scale, and the Chebyshev centre of the site "
                    "confidence regions.")
        if len(names) == 1:
            lo = min(e.theta[0] - 3 * e.se[0] for e in ests); hi = max(e.theta[0] + 3 * e.se[0] for e in ests)
            grid = np.linspace(lo, hi, 400)
            curves = np.array([((grid - e.theta[0]) / e.se[0]) ** 2 for e in ests])
            _plot(ch.surprise_fan(grid, curves, ids, float(mm[0]), float(full.theta[0]), truth[0] if truth is not None else None, names[0]))
        else:
            pts = np.array([e.theta for e in ests])
            t1 = np.linspace(pts[:, 0].min() - 0.1, pts[:, 0].max() + 0.1, 90)
            t2 = np.linspace(pts[:, 1].min() - 0.15, pts[:, 1].max() + 0.15, 90)
            T1, T2 = np.meshgrid(t1, t2)
            Z = np.zeros_like(T1)
            for e in ests:
                Vi = np.linalg.inv(e.cov)
                d1, d2 = T1 - e.theta[0], T2 - e.theta[1]
                Z = np.maximum(Z, Vi[0, 0] * d1**2 + 2 * Vi[0, 1] * d1 * d2 + Vi[1, 1] * d2**2)
            _plot(ch.surprise_contour(t1, t2, Z, {"pooled 2SLS": (full.theta, ch.IV, "circle"), "minimax": (mm, ch.MINIMAX, "square"),
                                                   "truth": (truth, ch.INK, "x")}, pts, ids, names))
        surprise = S.wald_surprise(ests, full.theta)
        surprise_mm = S.wald_surprise(ests, mm)
        st.dataframe(pl.DataFrame({"site": ids, "z² at pooled 2SLS": surprise.round(2), "z² at minimax": surprise_mm.round(2)}),
                     width="stretch", hide_index=True)
        st.caption("Robust across sites is not the same as robust to a bad site: give one site a deviant effect (what-if) and the "
                   "minimax estimate moves towards it, because leaving that site unhappy is exactly what it refuses to do. "
                   "The joint-model heatmap above is the tool that names the site instead.")

        st.markdown("#### V-REx: equalise the sites' excess risk")
        st.markdown("Krueger et al. (2021): mean excess risk across sites plus λ times its variance. The excess risk is each "
                    "site's second-stage loss above its own minimum, so the penalty targets how badly a shared θ fits each site, "
                    "not the site's noise level.")
        lams = np.logspace(-1, 4, 26)
        vp = S.vrex_path(sites, model, ests, full.theta, lams)
        _plot(ch.path(lams, vp, names, truth, "λ", "V-REx path from the pooled fit (λ → 0) towards equal excess risk",
                      color=ch.VREX, ref=full.theta, ref_label="2SLS"))
        st.caption("For the quadratic curve λ → 0 is the stacked second stage (one instrument block per site), which sits a little "
                   "off the pooled 2SLS. Like the minimax estimate, V-REx accommodates a deviant site rather than exposing it.")
