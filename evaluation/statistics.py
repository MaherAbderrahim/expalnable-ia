"""Statistical analysis: bootstrap CIs + paired DeLong + multi-seed variance.

Everything here protects one idea: *is an observed improvement actually real
enough to claim?* No experiment writes into this module except numbers.
"""

from __future__ import annotations

import math

import numpy as np


def _rankdata_py(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1
        ranks[order[i: j + 1]] = avg
        i = j + 1
    return ranks


def _auc_pairewise(x: np.ndarray, y: np.ndarray) -> float:
    """AUC of score x with labels y (1=pos, 0=neg), Mann-Whitney definition."""
    pos = x[y == 1]
    neg = x[y == 0]
    gt = 0
    eq = 0
    for p in pos:
        gt += int((p > neg).sum())
        eq += int((p == neg).sum())
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan")
    return (gt + 0.5 * eq) / (m * n)


def _midrank(z: np.ndarray) -> np.ndarray:
    order = np.argsort(z, kind="mergesort")
    ranks = np.empty(len(z), dtype=np.float64)
    sz = z[order]
    i = 0
    while i < len(z):
        j = i
        while j + 1 < len(z) and sz[j + 1] == sz[i]:
            j += 1
        avg = 0.5 * (i + j) + 1
        ranks[order[i: j + 1]] = avg
        i = j + 1
    return ranks


def delong_test(y_true, y_score_a, y_score_b) -> dict:
    """Fast DeLong (Sun & Xu 2014 vectorized midrank version), paired.

    Returns z, p (two-sided), auc_a, auc_b. Same samples, two score vectors.
    """
    y = np.asarray(list(map(int, y_true)))
    xa = np.asarray(list(map(float, y_score_a)))
    xb = np.asarray(list(map(float, y_score_b)))
    pos = y == 1
    neg = y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return {"z": None, "p_value": None, "auc_a": None, "auc_b": None,
                "note": "degenerate labels"}

    z = np.concatenate([xa, xb])
    z_ranks = np.concatenate([_midrank(xa), _midrank(xb)])
    z_all = _midrank(z)
    rank_a = z_all[: len(xa)]
    rank_b = z_all[len(xa):]
    v01 = (rank_a[pos] - rank_b[pos]) / m           # positive contributions
    v10 = (rank_a[neg] - rank_b[neg]) / n
    s01 = float(np.var(v01, ddof=1)) if m > 1 else 0.0
    s10 = float(np.var(v10, ddof=1)) if n > 1 else 0.0
    if s01 <= 0 and s10 <= 0:
        return {"z": 0.0, "p_value": 1.0,
                "auc_a": _auc_pairewise(xa, y), "auc_b": _auc_pairewise(xb, y),
                "note": "scores identical"}
    denom = math.sqrt((s01 / m) + (s10 / n))
    auc_a = _auc_pairewise(xa, y)
    auc_b = _auc_pairewise(xb, y)
    z_stat = (auc_a - auc_b) / denom if denom > 0 else 0.0
    from scipy.stats import norm
    p = 2.0 * norm.sf(abs(z_stat))
    return {"z": float(z_stat), "p_value": float(p), "auc_a": float(auc_a), "auc_b": float(auc_b),
            "s01": s01, "s10": s10, "m": m, "n": n}


# ---------------------------------------------------------------------------
# bootstrap comparison of two AUCs (correlated per-sample bootstrap)
# ---------------------------------------------------------------------------

def bootstrap_auc_difference(y_true, y_score_a, y_score_b, n_boot: int = 2000,
                             alpha: float = 0.05, seed: int = 2026) -> dict:
    """CI + p-value of AUROC(a) - AUROC(b) via case-level bootstrap.

    Stratified resampling by class; both score vectors resampled with the SAME
    indices (paired design).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(list(map(int, y_true)))
    a = np.asarray(list(map(float, y_score_a)))
    b = np.asarray(list(map(float, y_score_b)))
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    diffs = []
    for _ in range(n_boot):
        try:
            ip = rng.choice(pos, size=len(pos), replace=True)
            inn = rng.choice(neg, size=len(neg), replace=True)
            idx = np.concatenate([ip, inn])
            da = _auc_pairewise(a[idx], y[idx])
            db = _auc_pairewise(b[idx], y[idx])
            if not math.isnan(da) and not math.isnan(db):
                diffs.append(da - db)
        except ValueError:
            continue
    if not diffs:
        return {"mean_diff": _auc_pairewise(a, y) - _auc_pairewise(b, y),
                "ci_lower": None, "ci_upper": None, "p_value": None, "n_boot": n_boot}
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided percentile p-value: probability of crossing 0
    p_val = min(1.0, 2.0 * min(float((diffs > 0).mean()), float((diffs < 0).mean())))
    return {
        "mean_diff": float(_auc_pairewise(a, y) - _auc_pairewise(b, y)),
        "ci_lower": float(lo), "ci_upper": float(hi),
        "p_value": float(p_val), "n_boot": len(diffs),
    }


# ---------------------------------------------------------------------------
# variance across inference runs (fold-level diversity as seed proxy)
# ---------------------------------------------------------------------------

def across_runs_variance(per_run_auc: list[float]) -> dict:
    """mean ± sd (and CoV) of a metric across repeated runs."""
    v = np.asarray([x for x in per_run_auc if x is not None and not math.isnan(x)],
                   dtype=np.float64)
    if v.size == 0:
        return {"mean": None, "sd": None, "cov": None, "n_runs": 0}
    return {
        "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "cov": float(v.std(ddof=1) / abs(v.mean())) if v.size > 1 else 0.0,
        "values": [float(x) for x in v],
        "n_runs": int(v.size),
    }
