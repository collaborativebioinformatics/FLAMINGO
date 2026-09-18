"""Malicious-site behaviour.

Threat model: a malicious site runs the protocol's software as it likes. It
knows the global model it receives and its own data, but not the other sites'
updates (no omniscient or colluding attacks such as "a little is enough").
It cannot break the transport or the cryptography.

Model poisoning replaces the update u = local - global that the site sends:
    sign_flip   -scale * u
    scale        scale * u        (boosting; scale ~ K replaces the global model under FedAvg)
    gaussian     N(0, noise_std^2) per coordinate
Data poisoning changes what the site trains on and then sends its update honestly:
    outcome_flip     the loss sees the negated model output. For continuous Y this is training on
                     -Y, for binary Y on 1 - Y, for survival on the reversed hazard ordering. It
                     pushes f toward the mirror image of the causal curve.
    outcome_shuffle  outcomes permuted within the site's training split: X and the SNPs keep their
                     joint distribution but lose any link to the outcome, pulling f toward flat
                     (a "hide the effect" attack).
weight_multiplier inflates the sample size the site reports; it only matters with weights: reported.
The site's own test split stays clean, so its reported test metrics are honest.
"""

import numpy as np


def poison_update(u, attack, rng):
    kind = attack.kind
    if kind == "sign_flip":
        return -attack.scale * u
    if kind == "scale":
        return attack.scale * u
    if kind == "gaussian":
        return rng.normal(0.0, attack.noise_std, size=u.shape)
    return u


class _FlippedLoss:
    """The site's Task with the model output negated inside the loss; everything else delegated."""

    def __init__(self, base):
        self._base = base

    def loss(self, out, tgt):
        return self._base.loss(-out, tgt)

    def __getattr__(self, name):
        return getattr(self._base, name)


class DataPoisoning:
    """Swaps a Site's training targets or loss before each round it attacks in."""

    def __init__(self, site, attack, seed, site_index):
        self.attack = attack
        self.clean_task, self.clean_targets = site.task, site.t_tr
        self.poisoned_task, self.poisoned_targets = site.task, site.t_tr
        if attack.kind == "outcome_flip":
            self.poisoned_task = _FlippedLoss(site.task)
        elif attack.kind == "outcome_shuffle":
            perm = np.random.default_rng([seed, 6, site_index]).permutation(site.n_train)
            # all target tensors are permuted together (time and event stay paired)
            self.poisoned_targets = {k: v[perm] for k, v in site.t_tr.items()}

    def prepare(self, site, rnd):
        on = rnd >= self.attack.start_round
        site.task = self.poisoned_task if on else self.clean_task
        site.t_tr = self.poisoned_targets if on else self.clean_targets
