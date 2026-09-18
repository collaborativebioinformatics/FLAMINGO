"""Secure aggregation by pairwise additive masking (the core of Bonawitz et al. 2017).

Each site encodes its pre-weighted update as fixed-point integers mod 2^64 and
adds a mask. For every pair of sites (i, j), i < j, both derive the same
pseudorandom vector r_ij from a key only they share; site i adds r_ij, site j
subtracts it. Every mask cancels in the sum, so the server recovers
sum_k (weighted update_k) exactly (up to fixed-point rounding) while each
individual message is uniformly distributed mod 2^64 and carries no
information about that site's update.

Simplifications of this simulation, all documented in ROBUST_PRIVATE.md:
- Pairwise keys come from a session secret shared by the sites (standing in
  for a Diffie-Hellman key agreement). The server code never receives it.
- No dropout recovery: every site must report every round (NVFlare's FedAvg
  here waits for all sites anyway). A missing site would leave its masks
  uncancelled and the sum unusable.
- Honest-but-curious server that does not collude with sites. A server that
  colludes with K-2 sites learns the remaining two sites' sum only.
- The server learns the sum, hence the aggregate: secure aggregation alone
  does not stop inference from the global model; combine with dp for that.
"""

import hashlib

import numpy as np

_LIMIT = 2.0 ** 62


def session_key(seed: int) -> int:
    """The sites' shared session secret, derived from the run seed (simulation of key agreement)."""
    return int.from_bytes(hashlib.sha256(f"fedsec-secagg-session-{seed}".encode()).digest()[:8], "little")


def encode(x, frac_bits):
    x = np.asarray(x, dtype=np.float64)
    scaled = np.round(x * 2.0 ** frac_bits)
    if np.any(~np.isfinite(scaled)) or np.any(np.abs(scaled) >= _LIMIT / 64):
        raise OverflowError("secagg: value too large for the fixed-point range; lower secagg.frac_bits")
    return scaled.astype(np.int64).view(np.uint64)


def decode(s, frac_bits):
    return np.asarray(s, dtype=np.uint64).view(np.int64).astype(np.float64) / 2.0 ** frac_bits


def _pair_mask(key, i, j, rnd, size):
    return np.random.PCG64(np.random.SeedSequence([key, 4, i, j, rnd])).random_raw(size).astype(np.uint64)


def mask(site_index, n_sites, rnd, key, size):
    """sum_{j > i} r_ij - sum_{j < i} r_ji  (mod 2^64)."""
    m = np.zeros(size, dtype=np.uint64)
    for j in range(n_sites):
        if j == site_index:
            continue
        r = _pair_mask(key, min(site_index, j), max(site_index, j), rnd, size)
        if site_index < j:
            m += r
        else:
            m -= r
    return m


def mask_message(vec, site_index, n_sites, rnd, key, frac_bits):
    enc = encode(vec, frac_bits)
    return enc + mask(site_index, n_sites, rnd, key, enc.size)


def unmask_sum(messages, frac_bits):
    """Sum of masked messages mod 2^64, decoded. Correct only if every site's message is present."""
    total = np.zeros_like(np.asarray(messages[0], dtype=np.uint64))
    for m in messages:
        total += np.asarray(m, dtype=np.uint64)
    return decode(total, frac_bits)
