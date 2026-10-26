"""Error analysis: per-case decisions + patterns, produced as error_analysis.csv.

For the frozen final config on a given split we classify every case as
TP / TN / FP / FN and join: confidence, IoU/Dice (annotation overlap when
available), clinical metadata (PI-RADS, ISUP, PSA, PSAD, prostate volume,
histopath type, center, age) and timing. Frequency summaries of the four
error types finalize the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation.paths import EXPERIMENTS_RESULTS
from evaluation.eval_model import load_split_cases


def _bin_class(t: int, p: int) -> str:
    if t == 1 and p == 1:
        return "TP"
    if t == 0 and p == 0:
        return "TN"
    if t == 0 and p == 1:
        return "FP"
    return "FN"


def build_error_analysis(split: str, config_name: str, threshold: float,
                         out_name: str | None = None,
                         scores_csv: Path | None = None) -> pd.DataFrame:
    """Per-case error table for one (split, config, threshold)."""
    base = EXPERIMENTS_RESULTS / "error_analysis"
    base.mkdir(parents=True, exist_ok=True)
    if scores_csv is not None:
        scores = pd.read_csv(scores_csv)
    else:
        scores = pd.read_csv(
            EXPERIMENTS_RESULTS / "final_evaluation" / f"{split}_scores_long.csv"
        )
    sub = scores[(scores["config"] == config_name)].copy()
    sub["pred_at_threshold"] = (sub["likelihood"] >= threshold).astype(int)
    sub["error_type"] = [
        _bin_class(int(t), int(p)) for t, p in zip(sub["y_true"], sub["pred_at_threshold"])
    ]
    sub["confidence_tier"] = pd.cut(
        sub["likelihood"], bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["low(<0.2)", "mid(0.2-0.5)", "high(0.5-0.8)", "very_high(>0.8)"],
        include_lowest=True,
    )

    # messy meta columns may be missing if marksheet rows lack them
    cols = [
        "case_id", "patient_id", "study_id", "center", "split", "y_true", "likelihood",
        "pred_at_threshold", "error_type", "confidence_tier", "iou", "dice",
        "overlap_10pct", "gt_mask_available", "config",
    ]
    present = [c for c in cols if c in sub.columns]
    out = sub[present].copy()
    if "split" not in out.columns:
        out.insert(1, "split", split)
    out["dataset"] = "PI-CAI public v2.0 (subset stratified)"
    out.insert(len(out.columns), "threshold", threshold)

    if "lesion_PIRADS" in sub.columns:
        out["max_lesion_PIRADS"] = sub["lesion_PIRADS"].apply(_max_pirads)
        out["difficulty"] = [
            ("hard_PIRADS3_or_less" if (t == 1 and mp <= 3) else
             "hard_unannotated" if (t == 1 and gt_avail is not True) else
             "standard")
            for t, mp, gt_avail in zip(out["y_true"], out["max_lesion_PIRADS"],
                                       out.get("gt_mask_available", [True] * len(out)))
        ]
    out["metadata"] = sub.apply(
        lambda r: json.dumps({
            k: (None if pd.isna(r.get(k)) else (r.get(k)))
            for k in ["case_ISUP", "lesion_PIRADS", "prostate_volume", "psa", "psad",
                      "histopath_type", "patient_age"] if k in sub.columns
        }, default=str),
        axis=1,
    )
    out.to_csv(base / f"{out_name or split}_{config_name}_error_analysis.csv", index=False)
    return out


def _max_pirads(pirads) -> int:
    import math
    import re
    if isinstance(pirads, float) and math.isnan(pirads):
        return 0
    vals = re.findall(r"\d+", str(pirads))
    return max((int(v) for v in vals), default=0)


def summarize_patterns(err: pd.DataFrame, split: str, config_name: str) -> dict:
    """Where does the model fall short? (baseline descriptives, no overfitting)"""
    by_type = err["error_type"].value_counts().to_dict()
    n_pos = int((err["y_true"] == 1).sum())
    n_neg = len(err) - n_pos

    def m_fil(df, col, default=None):
        return float(df[col].mean()) if (col in df.columns and len(df)) else default

    patterns = {
        "split": split,
        "config": config_name,
        "counts_by_type": by_type,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "FN_mean_likelihood": m_fil(err[err.error_type == "FN"], "likelihood"),
        "FP_mean_likelihood": m_fil(err[err.error_type == "FP"], "likelihood"),
        "TP_mean_likelihood": m_fil(err[err.error_type == "TP"], "likelihood"),
        "TN_mean_likelihood": m_fil(err[err.error_type == "TN"], "likelihood"),
        "FN_mean_iou_when_annotated": m_fil(
            err[err.error_type == "FN"], "iou"),
        "FN_by_PIRADS": err[err.error_type == "FN"].groupby("max_lesion_PIRADS").size().to_dict()
        if (err.error_type == "FN").any() and "max_lesion_PIRADS" in err.columns else {},
        "FP_by_center": err[err.error_type == "FP"].groupby("center").size().to_dict()
        if (err.error_type == "FP").any() else {},
        "FN_by_center": err[err.error_type == "FN"].groupby("center").size().to_dict()
        if (err.error_type == "FN").any() else {},
    }
    return patterns
