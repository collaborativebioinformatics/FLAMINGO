"""Models. The scalar output means predicted Y, a logit, or a log relative
hazard depending on the outcome family (see tasks.py).

MLP      naive fit of outcome on X: out = f(X)
MRModel  second stage of a two-stage Mendelian randomization. The first
         stage (X ~ SNPs) is fitted locally at each site by OLS, because the
         simulated sites have their own SNP effects and allele frequencies,
         so only this second stage is federated:
            2sri   out = f(X) + c (X - X_hat)  control function: the first-stage
                                               residual carries the confounder; the
                                               term is linear so that f is identified
                                               (a flexible h would be collinear with f
                                               when the instruments are weak)
            2sps   out = f(X_hat)               predictor substitution
         f is the causal-curve head.
"""

import torch.nn as nn


def _mlp(hidden):
    return nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))


class MLP(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.hidden = hidden          # NVFlare rebuilds the model from attributes named like the init args
        self.net = _mlp(hidden)

    def forward(self, x, xhat=None):
        return self.net(x).squeeze(-1)

    def curve(self, x):
        return self.net(x).squeeze(-1)


class MRModel(nn.Module):
    def __init__(self, method: str = "2sri", hidden: int = 32):
        super().__init__()
        assert method in ("2sri", "2sps")
        self.method, self.hidden = method, hidden
        self.f = _mlp(hidden)
        self.h = nn.Linear(1, 1, bias=False) if method == "2sri" else None

    def forward(self, x, xhat):
        if self.method == "2sps":
            return self.f(xhat).squeeze(-1)
        return self.f(x).squeeze(-1) + self.h(x - xhat).squeeze(-1)

    def curve(self, x):
        """The causal-curve head alone, on a grid of X values."""
        return self.f(x).squeeze(-1)
