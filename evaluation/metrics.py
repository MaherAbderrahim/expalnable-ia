"""Complete medical evaluation metrics for csPCa detection (case level + lesion level).

Design principles
-----------------
- Every function is pure (numpy in / pure python or numpy out).
- Case-level metrics use configurable operating thresholds.
- Confidence intervals via bootstrap (percentile method) — see bootstrap_ci().
- All metrics gracefully handle degenerate inputs (no positives -> None).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# confusion matrix / operating-point metrics
# ---------------------------------------------------------------------------

def confusion_counts(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(int(t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    fp = sum(int(t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum(int(t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fn = sum(int(t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _ratio(num: int, den: int) -> float | None:
    return num / den if den > 0 else None


def operating_point_metrics(y_true, y_score, threshold: float) -> dict:
    """sens/spec/precision/recall/F1/bal-acc + confusion matrix at one threshold."""
    y_t = [int(bool(v)) for v in y_true]
    y_p = [int(s >= threshold) for s in y_score]
    c = confusion_counts(y_t, y_p)
    sens = _ratio(c["tp"], c["tp"] + c["fn"])          # == recall == TPR
    spec = _ratio(c["tn"], c["tn"] + c["fp"])          # == TNR
    precision = _ratio(c["tp"], c["tp"] + c["fp"])     # == PPV
    f1 = (2 * precision * sens / (precision + sens)
          if (precision is not None and sens is not None and (precision + sens) > 0)
          else None)
    bal_acc = (0.5 * (sens + spec) if (sens is not None and spec is not None) else None)
    acc = _ratio(c["tp"] + c["tn"], c["tp"] + c["tn"] + c["fp"] + c["fn"])
    npv = _ratio(c["tn"], c["tn"] + c["fn"])
    out = {
        "threshold": float(threshold),
        "n_cases": len(y_t),
        "n_positive": sum(y_t),
        "n_negative": len(y_t) - sum(y_t),
        **c,
        "sensitivity": sens,
        "specificity": spec,
        "precision": precision,
        "recall": sens,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
    }
    return out


# ---------------------------------------------------------------------------
# ranking metrics
# ---------------------------------------------------------------------------

def roc_auc(y_true, y_score) -> float | None:
    if len(set(int(v) for v in y_true)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(list(map(int, y_true)), list(map(float, y_score))))


def pr_auc(y_true, y_score) -> float | None:
    """Average precision == area under the precision-recall curve."""
    if len(set(int(v) for v in y_true)) < 2:
        return None
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(list(map(int, y_true)), list(map(float, y_score))))


# ---------------------------------------------------------------------------
# threshold sweeps
# ---------------------------------------------------------------------------

def threshold_sweep(y_true, y_score, thresholds=None) -> list[dict]:
    """Metrics for a configurable set of decision thresholds."""
    if thresholds is None:
        thresholds = [round(t / 20 * 100) / 100 for t in range(0, 21)]  # 0.00..1.00 step 0.05
    return [operating_point_metrics(y_true, y_score, t) for t in thresholds]


def best_threshold_youden(y_true, y_score) -> tuple[float, float]:
    """sweep up to 401 candidate score thresholds; return (threshold, balanced acc)."""
    scores = sorted(set(float(s) for s in y_score))
    cand = [0.0, 1.0]
    if scores:
        mid = [(scores[i] + scores[i + 1]) / 2 for i in range(len(scores) - 1)]
        cand += mid
    cand = sorted(set(cand))
    best_t, best_ba = 0.5, -1.0
    for t in cand:
        m = operating_point_metrics(y_true, y_score, t)
        if m["balanced_accuracy"] is not None and m["balanced_accuracy"] > best_ba:
            best_ba, best_t = m["balanced_accuracy"], t
    return best_t, best_ba


def best_threshold_f1(y_true, y_score) -> tuple[float, float]:
    scores = sorted(set(float(s) for s in y_score))
    mid = [(scores[i] + scores[i + 1]) / 2 for i in range(len(scores) - 1)] if scores else []
    cand = sorted(set([0.0, 1.0] + mid))
    best_t, best_f1 = 0.5, -1.0
    for t in cand:
        m = operating_point_metrics(y_true, y_score, t)
        if m["f1"] is not None and m["f1"] > best_f1:
            best_f1, best_t = m["f1"], t
    return best_t, best_f1


def threshold_at_target_sensitivity(y_true, y_score, target_sens: float) -> tuple[float | None, float]:
    """lowest threshold reaching >= target sensitivity; (None, -1) if impossible."""
    scores = sorted(set(float(s) for s in y_score))
    for t in scores:
        m = operating_point_metrics(y_true, y_score, t)
        if m["sensitivity"] is not None and m["sensitivity"] >= target_sens:
            return t, m["sensitivity"]
    return None, -1.0


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> tuple[float, list[dict]]:
    """ECE with equal-width bins + bin table for the reliability diagram."""
    y_t = np.asarray(list(map(int, y_true)))
    y_s = np.asarray(list(map(float, y_score)))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_s > lo) if i == 0 else (y_s >= lo)
        mask &= (y_s <= hi) if i == n_bins - 1 else (y_s < hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({"bin": i, "n": 0, "mean_score": None, "frac_positive": None,
                         "gap": None, "ece_contribution": 0.0})
            continue
        mean_s = float(y_s[mask].mean())
        frac_p = float(y_t[mask].mean())
        gap = abs(frac_p - mean_s)
        ece += (n / len(y_s)) * gap
        bins.append({"bin": i, "n": n, "mean_score": mean_s,
                     "frac_positive": frac_p, "gap": gap, "ece_contribution": (n / len(y_s)) * gap})
    return float(ece), bins


def fit_platt_scale(y_true, y_score) -> tuple[float, float]:
    """logistic calibration a,b: p = sigmoid(a * score + b), fit on (score, label).

    Simple unregularized logistic regression via scipy if available; a closed
    tiny Newton loop otherwise. Returns (a, b).
    """
    x = np.asarray(list(map(float, y_score)), dtype=np.float64)
    y = np.asarray(list(map(int, y_true)), dtype=np.float64)
    # guard: shift tiny scores away from exact 0/1 saturation
    x = np.clip(x, 1e-6, 1 - 1e-6)
    z = np.log(x / (1 - x))  # logit of raw score

    a, b = 1.0, 0.0
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-(a * z + b)))
        w = p * (1 - p)
        grad = np.array([(z * (p - y)).sum(), (p - y).sum()])
        h11 = (z * z * w).sum()
        h12 = (z * w).sum()
        h22 = w.sum()
        hmat = np.array([[h11, h12], [h12, h22]])
        try:
            step = np.linalg.solve(hmat, grad)
        except np.linalg.LinAlgError:
            break
        a -= step[0]
        b -= step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(a), float(b)


def apply_platt(y_score, a: float, b: float):
    x = np.clip(np.asarray(list(map(float, y_score))), 1e-6, 1 - 1e-6)
    z = np.log(x / (1 - x))
    return list(map(float, 1.0 / (1.0 + np.exp(-(a * z + b)))))


# ---------------------------------------------------------------------------
# bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(metric_fn, y_true, y_score, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 2026) -> dict:
    """percentile bootstrap CI for metric_fn(y_true, y_score).

    Works for scalar ranking metrics (AUROC/AP). Stratified by class so every
    resample contains both classes (avoids undefined metric draws).
    """
    rng = np.random.default_rng(seed)
    y_t = np.asarray(list(map(int, y_true)))
    y_s = np.asarray(list(map(float, y_score)))
    pos_idx = np.flatnonzero(y_t == 1)
    neg_idx = np.flatnonzero(y_t == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {"value": metric_fn(y_true, y_score), "ci_lower": None, "ci_upper": None,
                "n_boot": n_boot}
    vals = []
    for _ in range(n_boot):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, n])
        val = metric_fn(y_t[idx].tolist(), y_s[idx].tolist())
        if val is not None:
            vals.append(val)
    if not vals:
        return {"value": metric_fn(y_true, y_score), "ci_lower": None, "ci_upper": None,
                "n_boot": n_boot}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": metric_fn(y_true, y_score), "ci_lower": float(lo),
            "ci_upper": float(hi), "n_boot": n_boot}


# ---------------------------------------------------------------------------
# lesion-localization metrics (3D volumes)
# ---------------------------------------------------------------------------

def detection_overlap(per_case_overlap: dict) -> dict:
    """Aggregates per-case IoU / Dice / lesion-TP statistics.

    per_case_overlap: {case_id: {"iou": float, "dice": float, "overlap_10pct": bool,
                                 "gt_positive": bool}}
    """
    import math

    ious = [(k, v["iou"]) for k, v in per_case_overlap.items()
            if v.get("iou") is not None and not math.isnan(v["iou"])]
    dices = [(k, v["dice"]) for k, v in per_case_overlap.items()
             if v.get("dice") is not None and not math.isnan(v["dice"])]
    pos_overlap = {k: v for k, v in per_case_overlap.items() if v.get("gt_positive")}
    n_lesion_tp = sum(1 for v in pos_overlap.values() if v.get("overlap_10pct"))
    return {
        "n_cases_with_gt": len(per_case_overlap),
        "n_positive_cases": len(pos_overlap),
        "n_lesion_overlaps_10pct": n_lesion_tp,
        "lesion_detection_rate_10pct": (_ratio(n_lesion_tp, len(pos_overlap))
                                        if pos_overlap else None),
        "mean_iou": (float(np.mean([v for _, v in ious])) if ious else None),
        "mean_iou_positives": (float(np.mean([v for k, v in ious
                                              if per_case_overlap[k].get("gt_positive")]))
                               if any(per_case_overlap[k].get("gt_positive") for k, _ in ious)
                               else None),
        "mean_dice": (float(np.mean([v for _, v in dices])) if dices else None),
    }


def mask_iou_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    """3D IoU + Dice of the binarized detection map vs the binary GT mask."""
    p = pred_mask > 0
    g = gt_mask > 0
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    denom = p.sum() + g.sum()
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / denom if denom > 0 else 0.0
    # PI-CAI official lesion-localization criterion: >=10% IoU overlap counts
    # as a lesion-level true positive (when GT has a lesion)
    return float(iou), float(dice)


# ---------------------------------------------------------------------------
# full case-level report
# ---------------------------------------------------------------------------

def full_case_report(y_true, y_score, threshold: float | None = None,
                     label_variant: str = "raw") -> dict:
    """Complete case-level evaluation on one set of scores.

    threshold: operating threshold. If None, uses the F1-optimal threshold on
    THESE scores — only appropriate for exploratory views; definitive reports
    must pass the threshold frozen on the validation split.
    """
    y_t = list(map(int, y_true))
    y_s = [float(s) for s in y_score]
    if threshold is None:
        threshold, _ = best_threshold_f1(y_t, y_s)

    report = {
        "label_variant": label_variant,
        "n_cases": len(y_t),
        "n_positive": sum(y_t),
        "auc_roc": roc_auc(y_t, y_s),
        "pr_auc": pr_auc(y_t, y_s),
        "operating_point": operating_point_metrics(y_t, y_s, threshold),
    }
    ece, bins = expected_calibration_error(y_t, y_s)
    report["ece_10bins"] = ece
    report["calibration_bins"] = bins
    report["sweep"] = threshold_sweep(y_t, y_s)
    r, sens90 = roc_auc(y_t, y_s), threshold_at_target_sensitivity(y_t, y_s, 0.90)
    report["threshold_at_sens90"] = {"threshold": sens90[0], "achieved_sens": sens90[1]}
    ys, _ = best_threshold_youden(y_t, y_s)
    report["threshold_youden_j"] = ys
    return report
