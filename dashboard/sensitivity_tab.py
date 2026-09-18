"""The sensitivity tab: leave-sites-out, regularisation paths, robustness across sites and
invariance tests, all computed live from per-site sufficient statistics (sensitivity.py) and
drawn with Plotly (charts.py). Nothing here writes to disk; what-if perturbations live in memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st

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
                "regularising, re-weighting and testing invariance never touches the rows again. That bundle is what "
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

    tabs = st.tabs(["Overview", "Leave sites out", "Regularisation", "Robust across sites", "Invariance & ICP"])

    # ------------------------------------------------------------------ overview
    with tabs[0]:
        cert = S.certificate(sites, model, cert_instr)
        drop1 = S.leave_out(sites, model, 1)
        shifts = np.array([(full.theta[0] - d["theta"][0]) / full.se[0] for d in drop1])
        worst = int(np.argmax(np.abs(shifts)))
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

        gam = np.array([np.inf])
        anchor_inf = S.anchor_path(sites, model, gam)[0]
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
            {"label": "anchor γ→∞ (between-site regression)", "theta": anchor_inf, "se": np.zeros_like(anchor_inf), "color": ch.ANCHOR,
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
    with tabs[1]:
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
        c1, c2 = st.columns(2)
        with c1:
            _plot(ch.funnel(ids, np.array([e.theta[0] for e in ests]), np.array([e.se[0] for e in ests]), het["fe"],
                            truth[0] if truth is not None else None, names[0], "Funnel: precision against estimate"))
        with c2:
            z = S.pairwise_z(ests, 0)
            _plot(ch.heatmap(z, ids, ids, f"Pairwise disagreement in {names[0]} (z of the difference)", zmax=4,
                             hover="%{y} − %{x}<br>z = %{z:.2f}", colorbar="z"))

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

    # ----------------------------------------------------------- regularisation
    with tabs[2]:
        st.markdown("#### K-class path: OLS → PULSE → 2SLS → LIML")
        st.markdown("The K-class family interpolates the confounded OLS fit (κ = 0) and 2SLS (κ = 1) and continues to LIML. "
                    "The **PULSE** (Jakobsen & Peters 2022) stops at the smallest κ whose residuals still pass the instrument test: "
                    "the least instrumenting the data can justify. Everything comes from the pooled cross products.")
        instr_opts = ["basis (x̂, x̂²)" if model == "quadratic" else "basis (x̂)"] + (["all SNPs (over-identified)"] if model == "linear" else [])
        instr_choice = st.radio("Instrument set", instr_opts, horizontal=True, key="kc_instr")
        A_k = S.aggregate(sites, model, "snps") if instr_choice.startswith("all") else A
        kp = S.kclass_path(A_k, alpha)
        _plot(ch.kclass(kp, names, truth))
        marks = kp["marks"]
        st.dataframe(pl.DataFrame([{"estimator": k, "κ": float(v[0]), **{n: float(v[1][j]) for j, n in enumerate(names)},
                                    "test statistic": float(v[2])} for k, v in marks.items()]), width="stretch", hide_index=True)
        if kp["pulse_rejected"]:
            st.caption(f"The instrument test rejects along the whole path (critical value {kp['crit']:.1f} on {kp['df']} df), "
                       "so the PULSE falls back to 2SLS. With SNP instruments this is the pooled Sargan test; see the certificate tab.")

        st.markdown("#### Ridge on the second stage")
        lambdas = np.logspace(-3, 1.5, 60)
        rp = S.ridge_path(A, lambdas)
        _plot(ch.path(lambdas, rp, names, truth, "λ (relative to the instrument signal)",
                      "Ridge-penalised 2SLS: shrinking θ towards 0", ref=full.theta, ref_label="2SLS"))
        st.caption("With a quadratic curve the curvature θ2 is the first to go: it is the less well identified direction.")

        st.markdown("#### Re-weighting the sites")
        powers = np.linspace(0, 1.5, 31)
        pw = np.array([S.ivw_meta(ests, float(a)).theta for a in powers])
        pw_se = np.array([S.ivw_meta(ests, float(a)).se for a in powers])
        _plot(ch.path(powers, pw, names, truth, "weight exponent a (0 = equal sites, 1 = inverse variance)",
                      "Meta-analysis with weights ∝ precision^a", logx=False, ref=ivw.theta, ref_label="IVW"))
        st.caption("Inverse-variance weighting lets the biggest, best-instrumented sites dominate; a = 0 gives every biobank one vote.")

    # ------------------------------------------------------------- robust
    with tabs[3]:
        st.markdown("#### Anchor regression with the site as anchor")
        st.markdown("Rothenhäusler et al. (2021): minimise the within-site residual sum of squares plus γ times the between-site "
                    "part. γ = 0 is the site-fixed-effects fit (the pooled 2SLS above), γ = 1 pools everyone with one intercept, "
                    "γ → ∞ regresses site means on site means and is the estimate most protected against shift interventions on the site. "
                    "Shown for the instrumented second stage and for the naive regression.")
        gammas = np.concatenate([[1e-3], np.logspace(-2, 3, 60)])
        an_iv = S.anchor_path(sites, model, gammas, iv=True)
        an_ols = S.anchor_path(sites, model, gammas, iv=False)
        inf_iv = S.anchor_path(sites, model, np.array([np.inf]), iv=True)[0]
        _plot(ch.path(gammas, an_iv, names, truth, "γ", "Anchored 2SLS (solid) and anchored naive regression (dotted)",
                      color=ch.ANCHOR, extra={"naive, anchored": (ch.NAIVE, an_ols)},
                      marks={"γ = 1 (one intercept)": (1.0, an_iv[np.argmin(np.abs(gammas - 1))], ch.ANCHOR),
                             "γ → ∞ (between sites)": (gammas[-1], inf_iv, ch.INK)}))
        st.caption("In this simulation the sites share E[X] = 0, so the between-site regression rests on a handful of noisy site "
                   "means and the path only moves at large γ. Inject an outcome level shift at one site (what-if) and the path bends.")

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
                   "The between-site certificate on the invariance tab is the tool that names the site instead.")

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

    # ----------------------------------------------------------- invariance
    with tabs[4]:
        st.markdown("#### The cross-site validity certificate")
        st.markdown("Stack every site's IV moment conditions and fit one θ by GMM. The over-identification statistic J splits exactly "
                    "into a **within-site** part (each site's own Sargan test: are its instruments valid?) and a **between-site** part "
                    "(do all sites agree on θ?). Both are functions of the per-site statistics only.")
        cert_choice = st.radio("Instruments", ["all SNPs at each site"] + (["basis (x̂, x̂²): between-site part only"] if model == "quadratic" else []),
                               horizontal=True, key="cert_instr")
        cert = S.certificate(sites, model, "snps" if cert_choice.startswith("all") else "basis")
        stat_row([stat_card("Overall J", f"{cert['J']:.1f}", f"{cert['df']} df · p = {_pvalue(cert['p'])}",
                            "coral" if cert["p"] < alpha else "pink"),
                  stat_card("Within sites", f"{cert['within']:.1f}", f"{cert['df_within']} df · p = {_pvalue(cert['p_within'])}",
                            "coral" if cert["p_within"] < alpha else "neutral"),
                  stat_card("Between sites", f"{cert['between']:.1f}", f"{cert['df_between']} df · p = {_pvalue(cert['p_between'])}",
                            "coral" if cert["p_between"] < alpha else "neutral"),
                  stat_card("GMM θ", _fmt(cert["theta"], np.sqrt(np.diag(cert["cov"]))), "precision-weighted site estimates", "pink")])
        _plot(ch.certificate(cert))
        if model == "quadratic" and cert_choice.startswith("all"):
            st.caption("X² is instrumented by the SNPs' linear effects here, which is weak; the (x̂, x̂²) basis is just-identified and "
                       "leaves only the between-site part.")

        st.markdown("#### Invariant instruments: sites as environments")
        st.markdown("A valid SNP has the same causal ratio βy/βx at every site. A pleiotropic SNP with direct effect α has ratio "
                    "θ + α/βx, which moves with the site's βx — so disagreement between sites flags it even when no single site can. "
                    "Below, z is each (site, SNP) deviation from βy = θ·βx, and T_j sums z² over sites. The greedy search removes the "
                    "SNP the sites disagree about most until the stacked J test accepts: the ICP-style largest accepted instrument set. "
                    "SNP *j* is treated as the same variant at every site, as it would be with real rsIDs.")
        A_lin = S.aggregate(sites, "linear", "snps")
        theta_lin = float(S.two_sls(A_lin).theta[0])
        inv = S.snp_invariance(sites, theta_lin)
        search = S.invariant_instrument_search(sites, alpha)
        injected = list(pert.pleio_snps) if pert.pleio_alpha else []
        stat_row([stat_card("SNPs kept", f"{len(search['accepted'])} / {inv['z'].shape[1]}",
                            "largest accepted set (greedy)" if search["is_accepted"] else "search stopped, set not accepted",
                            "pink" if search["is_accepted"] else "coral"),
                  stat_card("Removed", ", ".join(f"snp{j}" for j in search["removed"]) or "none",
                            ("injected: " + ", ".join(f"snp{j}" for j in injected)) if injected else "no pleiotropy injected",
                            "coral" if search["removed"] else "neutral"),
                  stat_card("θ on the final set", f"{search['path'][-1]['theta']:.3f}", f"± {1.96 * search['path'][-1]['se']:.3f} · p = {_pvalue(search['path'][-1]['p'])}", "pink")])
        if search["reason"]:
            st.info(f"Search stopped: {search['reason']}. A rejection that singles out no SNP is also what a chance rejection "
                    f"looks like: on a clean federation the stacked J test rejects at the nominal rate, and this draw of the "
                    f"committed data happens to sit at p ≈ 0.01 within sites.", icon="ℹ️")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            _plot(ch.heatmap(inv["z"], [f"snp{j}" for j in range(inv["z"].shape[1])], ids,
                             f"Deviation from βy = θ·βx per site and SNP (θ = {theta_lin:.3f})", zmax=4,
                             hover="%{y} · %{x}<br>z = %{z:.2f}", colorbar="z"))
        with c2:
            _plot(ch.snp_bars(inv["T"], inv["crit"], search["removed"], injected, "Summed disagreement per SNP"))
        _plot(ch.search_path(search["path"], truth_for_linear(sites, manifest), "θ (linear, SNP instruments)"))

        st.markdown("#### Invariant causal prediction with sites as environments")
        st.markdown("**Classic ICP** (Peters, Bühlmann & Meinshausen 2016) regresses Y on candidate predictor sets and keeps the sets "
                    "whose residuals are invariant across sites (a Chow test on the coefficients and an F test on the variances, "
                    "per site against the rest, Bonferroni over sites). Here the candidates include the oracle confounder U. "
                    "**IV-ICP** replaces the regression with the pooled 2SLS and tests, per site, that the structural residual is "
                    "orthogonal to (x̂, x̂²): hidden confounding may differ between sites, only the causal curve must be invariant.")
        icp = S.classic_icp(sites, alpha=alpha)
        ivicp = S.iv_icp(sites, alpha=alpha)
        c1, c2 = st.columns(2)
        with c1:
            stat_row([stat_card("Classic ICP output", _set_label(icp["parents"]),
                                "intersection of accepted sets" if icp["parents"] is not None else "no set is invariant", "coral" if not icp["parents"] else "pink")])
            _plot(ch.icp_heatmap(icp["results"], ids, "Classic ICP: −log10 p per candidate set and site"))
            st.caption("The confounder's effect on Y (γy) differs between sites in this simulation, so even Y | X, U is not invariant "
                       "and classic ICP finds nothing — sites are the wrong environments for a regression that cannot see the instruments.")
        with c2:
            stat_row([stat_card("IV-ICP output", _set_label(ivicp["parents"]),
                                "intersection of accepted sets" if ivicp["parents"] is not None else "no set is invariant",
                                "pink" if ivicp["parents"] else "coral")])
            _plot(ch.icp_heatmap(ivicp["results"], ids, "IV-ICP: −log10 p per candidate curve and site"))
            st.dataframe(pl.DataFrame([{"curve": " + ".join(r["set"]), "θ": _fmt(r["theta"], r["se"]), "p (Bonferroni)": r["p"],
                                        "accepted": r["accepted"]} for r in ivicp["results"]]), width="stretch", hide_index=True)
            st.caption("With a quadratic truth the linear curve leaves curvature in the residual, which the x̂² moment picks up at "
                       "every site; the quadratic curve is accepted. Under a linear truth both are accepted and the intersection is {X}.")


def truth_for_linear(sites, manifest):
    t = S.truth_for(sites, manifest, "linear")
    return float(t[0]) if t is not None else None


def _set_label(parents) -> str:
    if parents is None:
        return "no accepted set"
    return "{ " + ", ".join(sorted(parents)) + " }" if parents else "∅"
