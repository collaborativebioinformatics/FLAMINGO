"""One visual identity for every estimator, shared by the forest and dose-response plots
(data/scripts), the federated fitted-curve plots (federated_learning/src/plots.py) and the
dashboard charts. An estimator keeps its colour, line style and marker in every figure.

    truth          black, dashed
    concatenated   blue, solid, square          one 2SLS on the pooled rows (benchmark)
    sumstats       orange, solid, diamond       per-SNP IVW meta-analysis
    fed2sls        violet, dashed, down-triangle exact federated 2SLS, with its 95% band
    fed2sri        green, long-dash, up-triangle FedAvg-trained control-function network
    fed2sps        indigo, long-dash             FedAvg-trained predictor-substitution network
    naive          grey, solid, tick             no instruments (the confounded association)
    sites          light blue                    per-site estimates
    data           pale grey dots               pooled binned outcome
"""

INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"
BLUE, ORANGE, VIOLET, GREEN, INDIGO, SITE_BLUE, DATA_GREY = (
    "#2a78d6", "#eb6834", "#8a2be2", "#1baf7a", "#4a3aa7", "#7fa8dc", "#b8b8b5")

TRUTH = dict(color=INK, ls="--", lw=2)
CONCATENATED = dict(color=BLUE, ls="-", lw=2, marker="s")
SUMSTATS = dict(color=ORANGE, ls="-", lw=2, marker="D")
FED2SLS = dict(color=VIOLET, ls=(0, (4, 2)), lw=2, marker="v", band_alpha=0.15)
FED2SRI = dict(color=GREEN, ls=(0, (5, 2)), lw=2, marker="^")
FED2SPS = dict(color=INDIGO, ls=(0, (5, 2)), lw=2, marker="^")
NAIVE = dict(color=MUTED, ls="-", lw=2, marker="|")
SITES = dict(color=SITE_BLUE, marker="o")
DATA = dict(color=DATA_GREY, marker="o")

# keyed by the federated_learning/job.py method names
BY_METHOD = {"naive": NAIVE, "2sri": FED2SRI, "2sps": FED2SPS, "fed2sls": FED2SLS}

# The short name of each estimator, used verbatim in legends, row labels, cards and tables.
# Explanations belong in help text and captions, not in the name.
NAME = {
    "truth": "Truth",
    "pooled": "Concatenated 2SLS",
    "sumstats": "Sumstats IVW",
    "fed2sls": "Fed-2SLS",
    "2sri": "Fed-2SRI",
    "2sps": "Fed-2SPS",
    "naive": "Naive",
}


def line(style: dict) -> dict:
    """The matplotlib kwargs for drawing a line in this style."""
    return {"color": style["color"], "ls": style["ls"], "lw": style["lw"]}
