"""FLAMINGO dashboard: choose the simulation parameters, run the MR chain, read the results.

Streamlit entry point. All of the science lives in data/scripts/; this module is
parameter widgets, run orchestration (see runner.py) and presentation.
"""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import polars as pl
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner  # noqa: E402

HERE = Path(__file__).resolve().parent
LOGO = HERE / "assets" / "flamingo-logo-128.png"

st.set_page_config(page_title="FLAMINGO dashboard", page_icon=str(LOGO), layout="wide")

# Colours come from the pipeline figure in the README (flamingo_pipeline.png);
# the rest of the theme lives in .streamlit/config.toml. The figure's legend is
# load-bearing rather than decorative: pink is the causal pathway and what
# travels between sites, coral is what differs per site and the confounding, and
# a dashed edge marks a benchmark that needs pooled individual rows.
CSS = """
<style>
:root {
  --fl-rose: #d4537e;
  --fl-coral: #d85a30;
  --fl-maroon: #72243e;
  --fl-border: #f4c0d1;
  --fl-coral-tint: #faece7;
  --fl-ink: #2c2c2a;
  --fl-muted: #6b6a65;
}
.fl-header {
  display: flex; align-items: center; gap: 0.9rem;
  padding-bottom: 1rem; margin-bottom: 1.1rem;
  border-bottom: 2px solid var(--fl-border);
}
.fl-header img { width: 52px; height: 52px; border-radius: 14px; }
.fl-header .fl-title {
  margin: 0; line-height: 1.05; color: var(--fl-maroon);
  font-size: 2.05rem !important; font-weight: 800 !important; letter-spacing: 0.02em;
}
.fl-header .fl-sub {
  margin: 0.28rem 0 0; color: var(--fl-muted); font-size: 0.95rem !important;
}
h1, h2, h3 { color: var(--fl-maroon) !important; }
.stat-row {
  display: grid; gap: 0.75rem; margin: 0.2rem 0 0.9rem;
  grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
}
.stat {
  background: #ffffff; padding: 0.7rem 0.9rem;
  border: 1px solid var(--fl-border); border-left: 4px solid var(--fl-rose);
  border-radius: 10px;
}
.stat--coral { border-left-color: var(--fl-coral); background: var(--fl-coral-tint); }
.stat--benchmark { border-left-style: dashed; border-left-color: var(--fl-maroon); }
.stat--neutral { border-left-color: var(--fl-border); }
.stat__label {
  font-size: 0.75rem !important; font-weight: 600; letter-spacing: 0.02em;
  color: var(--fl-muted);
}
.stat__value {
  font-size: 1.55rem !important; font-weight: 700; line-height: 1.3; color: var(--fl-ink);
}
.stat__sub { font-size: 0.78rem !important; color: var(--fl-muted); min-height: 1rem; }
</style>
"""


@st.cache_data
def logo_data_uri() -> str:
    return "data:image/png;base64," + base64.b64encode(LOGO.read_bytes()).decode()


def stat_card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    klass = f"stat stat--{tone}" if tone else "stat"
    return (f'<div class="{klass}">'
            f'<div class="stat__label">{html.escape(label)}</div>'
            f'<div class="stat__value">{html.escape(value)}</div>'
            f'<div class="stat__sub">{html.escape(sub)}</div></div>')


def stat_row(cards: list[str]) -> None:
    st.markdown(f'<div class="stat-row">{"".join(cards)}</div>', unsafe_allow_html=True)

SHAPE_HELP = {
    "linear": "Y increases linearly in X. θ2 is unused.",
    "quadratic": "Y bends in X: θ1 is the slope at X = 0, θ2 the curvature.",
    "threshold": "Y is flat until the cutoff θ2, then rises with slope θ1.",
    "cox": "Survival outcome; θ1 is a log hazard ratio. θ2 is unused.",
}


def sidebar_params() -> dict:
    """Every knob the simulator takes, grouped. Returns a runner parameter dict."""
    st.sidebar.header("Simulation parameters")
    p = {}
    p["shape"] = st.sidebar.selectbox(
        "Causal shape (X → Y)", runner.SHAPES,
        index=runner.SHAPES.index(runner.DEFAULTS["shape"]),
        help="The dose-response curve every site shares.",
    )
    st.sidebar.caption(SHAPE_HELP[p["shape"]])

    st.sidebar.subheader("Causal effect")
    p["theta1"] = st.sidebar.slider(
        "θ1 — slope", -1.0, 1.0, float(runner.DEFAULTS["theta1"]), 0.01,
        help="Linear/Cox: the effect. Quadratic: slope at X = 0. Threshold: slope above the cutoff.",
    )
    if p["shape"] in runner.CURVED_SHAPES:
        p["theta2"] = st.sidebar.slider(
            "θ2 — curvature / cutoff", -1.0, 1.0, float(runner.DEFAULTS["theta2"]), 0.01,
            help="Quadratic: curvature. Threshold: the cutoff in X.",
        )

    st.sidebar.subheader("Sites")
    p["n_sites"] = st.sidebar.slider("Number of sites", 2, 30, runner.DEFAULTS["n_sites"])
    p["n_snps"] = st.sidebar.slider("SNPs per site", 2, 100, runner.DEFAULTS["n_snps"])
    p["pop_min"], p["pop_max"] = st.sidebar.slider(
        "Population size range", 200, 50_000,
        (runner.DEFAULTS["pop_min"], runner.DEFAULTS["pop_max"]), step=100,
        help="Site sizes are spread across this range, one draw per equal-width bin.",
    )

    st.sidebar.subheader("Between-site heterogeneity")
    p["h2x_mean"] = st.sidebar.slider(
        "Mean SNP heritability of X", 0.01, 0.6, float(runner.DEFAULTS["h2x_mean"]), 0.01,
        help="Instrument strength. Low values make weak instruments and wide intervals.",
    )
    p["h2x_kappa"] = st.sidebar.slider(
        "Heritability concentration κ", 2.0, 200.0, float(runner.DEFAULTS["h2x_kappa"]), 1.0,
        help="Higher κ = sites more alike.",
    )
    p["gamma_mean"] = st.sidebar.slider(
        "Mean confounder effect γ", 0.0, 0.9, float(runner.DEFAULTS["gamma_mean"]), 0.01,
        help="How strongly the unmeasured confounder drives X and Y. Drives the naive estimate's bias.",
    )
    p["gamma_kappa"] = st.sidebar.slider(
        "Confounder concentration κ", 2.0, 200.0, float(runner.DEFAULTS["gamma_kappa"]), 1.0,
    )

    if p["shape"] == "cox":
        st.sidebar.subheader("Survival")
        p["censor_frac"] = st.sidebar.slider(
            "Censored fraction", 0.0, 0.9, float(runner.DEFAULTS["censor_frac"]), 0.05)
        p["followup"] = st.sidebar.slider(
            "Follow-up length", 0.5, 40.0, float(runner.DEFAULTS["followup"]), 0.5,
            help="Administrative end of follow-up. Short follow-up makes events rare.",
        )

    st.sidebar.subheader("Reproducibility")
    p["seed"] = st.sidebar.number_input("Base seed", 0, 10_000, runner.DEFAULTS["seed"], 1,
                                        help="Site i draws with seed + i + 1.")
    return runner.canonical(p)


def run_steps(steps, params, paths, force: bool) -> dict:
    """Execute steps in order, stopping at the first failure. Returns key -> StepResult."""
    results = {}
    # Later steps read what earlier ones wrote: the MR scripts overlay the
    # federated curves. So once a step actually runs, everything downstream of it
    # has to run too, or it would replay results computed from the old inputs.
    stale = force
    for step in steps:
        with st.status(f"{step.label}…", expanded=False) as status:
            result = runner.execute(step, params, paths, force=stale)
            stale = stale or not result.cached
            results[step.key] = result
            if result.cached:
                status.update(label=f"{step.label} — cached", state="complete")
            elif result.ok:
                status.update(label=f"{step.label} — {result.seconds:.1f}s", state="complete")
            else:
                status.update(label=f"{step.label} — failed", state="error")
            st.code(result.output or "(no output)", language="text")
        if not result.ok:
            st.error(f"`{step.key}` exited with code {result.returncode}. "
                     "The command and its output are above.")
            break
    return results


def show_outputs(step, params, paths) -> None:
    """Figures inline, tables as dataframes, everything downloadable."""
    for label, path in step.outputs(params, paths).items():
        if not path.exists():
            continue
        if path.suffix == ".png":
            st.image(str(path), caption=label, width="stretch")
        elif path.suffix == ".csv":
            st.caption(label)
            st.dataframe(pl.read_csv(path), width="stretch", hide_index=True)
        with open(path, "rb") as fh:
            st.download_button(f"Download {label} ({path.name})", fh.read(), file_name=path.name,
                               key=f"dl-{paths.root.name}-{step.key}-{label}")


def headline_metrics(summary: dict) -> None:
    """The estimates every run is judged on, from the summary-MR printout."""
    if not summary:
        return
    cards = []
    if "meta_ivw" in summary:
        est, se = summary["meta_ivw"]
        cards.append(stat_card("Meta IVW", f"{est:.3f}",
                               f"95% CI ± {1.96 * se:.3f}", "pink"))
    if "pooled" in summary:
        est, se = summary["pooled"]
        cards.append(stat_card("Pooled (individual-level)", f"{est:.3f}",
                               f"95% CI ± {1.96 * se:.3f}", "benchmark"))
    if "pooled_naive" in summary:
        cards.append(stat_card("Pooled naive (no IV)", f"{summary['pooled_naive'][0]:.3f}",
                               "confounded", "coral"))
    if "heterogeneity_q" in summary:
        cards.append(stat_card("Heterogeneity Q", f"{summary['heterogeneity_q'][0]:.1f}",
                               "across sites", "neutral"))
    stat_row(cards)

    if {"model_sumstats", "pooled_quadratic"} & summary.keys():
        rows = []
        for key, name in (("model_sumstats", "Summary statistics (meta of site quadratic fits)"),
                          ("pooled_quadratic", "Concatenated quadratic 2SLS")):
            if key in summary:
                t1, t1se, t2, t2se = summary[key]
                rows.append({"estimator": name, "θ1": t1, "θ1 se": t1se, "θ2": t2, "θ2 se": t2se})
        st.dataframe(pl.DataFrame(rows), width="stretch", hide_index=True)

    if "target_label" in summary:
        st.caption(f"Target: {summary['target_label']}")
    st.caption("Pink travels between sites · dashed needs pooled individual rows "
               "(benchmark only) · coral is the confounded estimate MR is there to beat.")


def comparison_table(summary: dict, federated: dict, params: dict) -> pl.DataFrame:
    """One row per estimator, with what each is allowed to see.

    The point of the comparison is that the rows needing individual-level data
    are benchmarks a real federation could not run.
    """
    curved = params["shape"] in runner.CURVED_SHAPES

    def row(estimator, sees, theta1=None, theta2=None, se1=None):
        return {"estimator": estimator, "sees": sees,
                "θ1": theta1, "θ1 se": se1, "θ2": theta2}

    rows = [row("Truth", "simulation parameters", params["theta1"],
                params.get("theta2") if curved else None)]
    if "meta_ivw" in summary:
        est, se = summary["meta_ivw"]
        rows.append(row("Summary-stat IVW meta", "per-SNP summary statistics", est,
                        None, se))
    if curved and "model_sumstats" in summary:
        t1, se1, t2, _ = summary["model_sumstats"]
        rows.append(row("Summary-stat model meta", "per-site quadratic fits", t1, t2, se1))
    if curved and "pooled_quadratic" in summary:
        t1, se1, t2, _ = summary["pooled_quadratic"]
        rows.append(row("Concatenated quadratic 2SLS", "pooled individual rows", t1, t2, se1))
    elif "pooled" in summary:
        est, se = summary["pooled"]
        rows.append(row("Concatenated 2SLS", "pooled individual rows", est, None, se))
    for method, coef in sorted(federated.items()):
        rows.append(row(f"Federated FedAvg · {method}", "model updates only",
                        coef[0], coef[1] if len(coef) > 1 else None))
    if "pooled_naive" in summary:
        rows.append(row("Pooled naive (no instruments)", "pooled individual rows",
                        summary["pooled_naive"][0], None))
    return pl.DataFrame(rows, infer_schema_length=None)


params = sidebar_params()
paths = runner.paths_for(params)
steps = runner.steps_for(params)

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    f'<div class="fl-header"><img src="{logo_data_uri()}" alt="FLAMINGO logo">'
    '<div><p class="fl-title">FLAMINGO</p>'
    '<p class="fl-sub">Federated Non-Linear Mendelian Randomization — simulate biobank '
    'sites, then compare conventional MR against pooled and federated estimates.</p>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.sidebar.divider()
st.sidebar.caption(f"Run `{paths.root.name}`")
force = st.sidebar.checkbox("Force re-run", value=False,
                            help="Recompute even when this parameter set already has results.")

tab_data, tab_experiments = st.tabs(["1 · Data generation", "2 · Experiments & results"])

with tab_data:
    st.subheader("Simulate federated biobank sites")
    st.markdown(
        "Every site shares one causal curve `X → Y`; sites differ in size, instrument "
        "strength (`h2_x`) and confounding (`γx`, `γy`). Set the parameters in the sidebar."
    )
    if st.button("Generate sites", type="primary"):
        runner.save_params(params, paths)
        run_steps([runner.STEPS["simulate"]], params, paths, force)

    sites_dir = paths.sites_for(params["shape"])
    manifest_path = sites_dir / "manifest.csv"
    if manifest_path.exists():
        st.success(f"{params['shape']} sites in `{sites_dir.relative_to(runner.REPO)}`")
        manifest = pl.read_csv(manifest_path)
        cards = [stat_card("Sites", f"{manifest.height}", "biobanks", "pink"),
                 stat_card("Total individuals", f"{manifest['n'].sum():,}", "across all sites", "pink")]
        if "events" in manifest.columns:
            cards.append(stat_card("Events", f"{manifest['events'].sum():,}", "uncensored", "coral"))
        cards += [stat_card("Mean h2_x", f"{manifest['h2_x'].mean():.3f}",
                            "instrument strength", "neutral"),
                  stat_card("Mean γx", f"{manifest['gamma_x'].mean():.3f}",
                            "confounding on X", "coral")]
        stat_row(cards)
        st.dataframe(manifest, width="stretch", hide_index=True)
        st.caption("Site size")
        st.bar_chart(manifest.to_pandas().set_index("site_id")[["n"]], height=200)

        with st.expander("Preview one site's individual-level data"):
            site = st.selectbox("Site", manifest["site_id"].to_list())
            df = pl.read_csv(sites_dir / f"{site}.csv")
            st.caption(f"{df.height:,} individuals × {df.width} columns")
            st.dataframe(df.head(50), width="stretch", hide_index=True)
            st.json(json.loads((sites_dir / f"{site}.truth.json").read_text()), expanded=False)
    else:
        st.info("No data for this parameter set yet — press **Generate sites**.")

with tab_experiments:
    assets = runner.data_assets()
    if not assets:
        st.info("No simulated data yet — generate a dataset on the "
                "**1 · Data generation** tab first.")
    else:
        st.subheader("Select a data asset")
        ids = [a["run_id"] for a in assets]
        default = ids.index(paths.root.name) if paths.root.name in ids else 0
        choice = st.selectbox(
            "Simulated dataset", range(len(assets)), index=default,
            format_func=lambda i: runner.asset_label(assets[i]),
            help="Every dataset generated on the first tab. Defaults to the one "
                 "matching the sidebar parameters.",
        )
        asset = assets[choice]
        run_params = dict(asset["params"])
        run_paths = asset["paths"]

        st.subheader("Federated learning")
        available = runner.fl_available()
        if not available:
            st.warning(
                f"No interpreter at `{runner.FL_PYTHON}`. Run `uv sync` in "
                "`federated_learning/`, or set `FLAMINGO_FL_PYTHON` to an environment "
                "with torch and NVFlare. The MR steps below run without it."
            )
        run_params["run_federated"] = st.checkbox(
            "Run the federated workflow", value=available, disabled=not available,
            help="Trains one client per site with FedAvg, then overlays the federated "
                 "curve on the MR plots below.",
        )
        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
        run_params["fl_methods"] = c1.multiselect(
            "Methods", runner.FL_METHODS, default=runner.EXPERIMENT_DEFAULTS["fl_methods"],
            disabled=not run_params["run_federated"],
            help="naive: outcome on X directly, the confounded association. "
                 "2sri: site-local first stage, then a federated control function. "
                 "2sps: site-local first stage, then federated on the predicted X.",
        )
        run_params["fl_engine"] = c2.selectbox(
            "Engine", ["local", "nvflare"], disabled=not run_params["run_federated"],
            help="local reproduces FedAvg's arithmetic in-process in seconds; nvflare "
                 "stands up the real simulated federation and costs about 40 s per job.",
        )
        run_params["fl_rounds"] = c3.number_input(
            "Rounds", 1, 50, runner.EXPERIMENT_DEFAULTS["fl_rounds"],
            disabled=not run_params["run_federated"])
        run_params["fl_epochs"] = c4.number_input(
            "Local epochs", 1, 20, runner.EXPERIMENT_DEFAULTS["fl_epochs"],
            disabled=not run_params["run_federated"])

        steps = runner.steps_for(run_params)
        st.subheader("Run the experiment chain")
        st.markdown("**" + " → ".join(s.label.split(" (")[0] for s in steps) + "**")
        st.caption("The federated step runs before the MR steps so its curves appear "
                   "on the forest and dose-response plots.")
        if st.button("Run experiments", type="primary"):
            runner.save_params(run_params, run_paths)
            results = run_steps(steps, run_params, run_paths, force)
            if results and all(r.ok for r in results.values()):
                st.rerun()

        summary_step = runner.STEPS["summary_mr"]
        if runner.is_complete(summary_step, run_params, run_paths):
            log = (run_paths.logs / "summary_mr.log")
            summary = runner.parse_summary(log.read_text()) if log.exists() else {}
            federated = runner.parse_federated(log.read_text()) if log.exists() else {}

            st.divider()
            st.subheader("Key results")
            headline_metrics(summary)

            st.markdown("#### All estimators")
            st.caption("Every row targets the same causal curve; they differ in what "
                       "each one is allowed to see.")
            st.dataframe(comparison_table(summary, federated, run_params),
                         width="stretch", hide_index=True)

            st.divider()
            for step in steps:
                if step.key == "simulate":
                    continue
                st.markdown(f"#### {step.label}")
                show_outputs(step, run_params, run_paths)

            with st.expander("Step logs"):
                for step in steps:
                    step_log = run_paths.logs / f"{step.key}.log"
                    if step_log.exists():
                        st.caption(step.label)
                        st.code(step_log.read_text(), language="text")
        else:
            st.info("No results for this data asset yet — press **Run experiments**.")

    with st.expander("Previous runs"):
        rows = [{"run_id": r["run_id"], **r["params"]} for r in runner.list_runs()]
        if rows:
            st.caption("Each parameter set gets its own directory under "
                       "`data/results/dashboard_runs/`; selecting the same parameters "
                       "in the sidebar brings its results back without recomputing.")
            st.dataframe(pl.DataFrame(rows, infer_schema_length=None),
                         width="stretch", hide_index=True)
        else:
            st.caption("No runs recorded yet.")
