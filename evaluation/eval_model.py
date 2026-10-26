"""Evaluation engine: score saved inference components with post-processing
configs and produce complete metric reports for a split.

Pipeline position: run AFTER scripts/run_inference.py (which saves per-case
npz with folds_orig/folds_flip/native_fused). Everything here is offline/CPU.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from evaluation import metrics as M
from evaluation.paths import (
    IMAGES_DIR,
    MARKSHEET_PATH,
    MASKS_DIR,
    PREDICTIONS_DIR,
    SPLITS_DIR,
    EXPERIMENTS_RESULTS,
)
from evaluation.postproc import OFFICIAL_CONFIG, VARIANT_CONFIGS, resolve_folds, score_case


# ---------------------------------------------------------------------------

def clinical_table() -> pd.DataFrame:
    marksheet = pd.read_csv(MARKSHEET_PATH)
    marksheet["case_id"] = (
        marksheet["patient_id"].astype(str) + "_" + marksheet["study_id"].astype(str)
    )
    return marksheet


def load_split_cases(split: str, case_ids: list[str] | None = None) -> pd.DataFrame:
    """Cases of a split with labels + clinical metadata joined."""
    sel = json.loads(
        (SPLITS_DIR / "case_selection_seed2026.json").read_text(encoding="utf-8")
    ) if (SPLITS_DIR / "case_selection_seed2026.json").exists() else None
    if sel is not None and case_ids is None:
        case_ids = sel["case_ids"]
    split_file = sorted(SPLITS_DIR.glob("patient_split_*.json"))
    if split_file:
        sdata = json.loads(split_file[0].read_text(encoding="utf-8"))
        if case_ids is None:
            case_ids = sdata["splits"][split]
        else:
            case_ids = [c for c in case_ids if c in sdata["splits"][split]]
    mk = clinical_table()
    df = mk[mk["case_id"].isin(case_ids)].copy()
    df["case_id"] = pd.Categorical(df["case_id"], categories=case_ids, ordered=True)
    return df.sort_values("case_id")


def available_predictions(split: str) -> list[str]:
    root = PREDICTIONS_DIR / split
    if not root.exists():
        return []
    return sorted(p.name[: -len(".npz")] for p in root.glob("*.npz"))


def load_case_gt_mask(case_id: str):
    """expert lesion annotation, or None."""
    p = MASKS_DIR / f"{case_id}.nii.gz"
    if p.exists():
        return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
    return None


def mask_shape_matches(case_id: str, native_shape: tuple[int, int, int]) -> bool:
    mask = load_case_gt_mask(case_id)
    return mask is not None and mask.shape == tuple(native_shape)


def _score_case_all_configs(case_row, configs: list[dict], split: str) -> list[dict]:
    """Worker: one case (row) scored under every config. Module-level for
    ProcessPoolExecutor (Windows spawn)."""
    case_id = str(case_row["case_id"])
    npz_path = PREDICTIONS_DIR / split / f"{case_id}.npz"
    if not npz_path.exists():
        return []
    npz = np.load(npz_path, allow_pickle=True)
    native_shape = tuple(int(x) for x in npz["native_fused"].shape)
    gt = load_case_gt_mask(case_id)
    gt_ok = gt is not None and gt.shape == native_shape
    y_true = int(case_row["case_csPCa"] == "YES")
    rows = []
    for cfg in configs:
        t0 = time.perf_counter()
        out = score_case(npz, cfg, want_map=True)
        dt = time.perf_counter() - t0
        row = {
            "case_id": case_id,
            "patient_id": int(case_row["patient_id"]),
            "study_id": int(case_row["study_id"]),
            "center": case_row["center"],
            "split": split,
            "y_true": y_true,
            "config": cfg["name"],
            "likelihood": out["likelihood"],
            "score_seconds": round(dt, 3),
        }
        if gt_ok:
            gt_bool = gt > 0
            det_bool = out["det_map"] > 0
            inter = int(np.logical_and(det_bool, gt_bool).sum())
            union = int(np.logical_or(det_bool, gt_bool).sum())
            denom = int(det_bool.sum()) + int(gt_bool.sum())
            row["iou"] = inter / union if union else 0.0
            row["dice"] = 2 * inter / denom if denom else 0.0
            row["overlap_10pct"] = bool(inter / union > 0.10) if union else False
            row["gt_mask_available"] = True
        else:
            row["iou"] = None
            row["dice"] = None
            row["overlap_10pct"] = None
            row["gt_mask_available"] = bool(gt is not None)
        rows.append(row)
    return rows


def score_dataframe(df: pd.DataFrame, configs: list[dict], split: str,
                    compute_stats: bool = True, verbose: bool = True,
                    n_workers: int = 1) -> pd.DataFrame:
    """Score every case (with saved npz) under every config.

    Returns a long-format DataFrame:
        case_id | y_true | config | likelihood | (+ optionally per-case stats)
    IoU/Dice vs expert annotation is computed for each case when the mask
    exists AND matches the native grid (positives and negatives alike —
    for negatives the IoU is 0.0 by definition, no lesion to overlap).
    n_workers > 1 spreads cases over processes (offline post-processing is
    CPU-bound; per-case results are order-independent, output re-ordered).
    """
    rows_df = df.reset_index(drop=True)
    if n_workers and n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        chunks = [(case_row, configs, split) for (_, case_row)
                  in rows_df.iterrows()]
        rows = []
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for sub in ex.map(_score_case_all_configs,
                              [c[0] for c in chunks],
                              [c[1] for c in chunks],
                              [c[2] for c in chunks]):
                rows.extend(sub)
                done += 1
                if verbose and done % 20 == 0:
                    print(f"  [score] {done}/{len(rows_df)} cases ({split})",
                          flush=True)
    else:
        rows = []
        for i, case_row in rows_df.iterrows():
            sub = _score_case_all_configs(case_row, configs, split)
            rows.extend(sub)
            if verbose and (i + 1) % 20 == 0:
                print(f"  [score] {i + 1}/{len(df)} cases ({split})", flush=True)
    if not rows:
        return pd.DataFrame(columns=["case_id", "y_true", "config", "likelihood"])
    scores = pd.DataFrame(rows)
    # merge selected clinical covariates for error analysis convenience
    extra_cols = [c for c in ["case_ISUP", "lesion_PIRADS", "prostate_volume",
                              "psa", "psad", "histopath_type", "patient_age"]
                  if c in df.columns]
    cov = rows_df[["case_id"] + extra_cols].copy()
    cov["case_id"] = cov["case_id"].astype(str)
    scores["case_id"] = scores["case_id"].astype(str)
    scores = scores.merge(cov, on="case_id", how="left")
    return scores


def evaluate_config(scores: pd.DataFrame, config_name: str, threshold: float,
                    label_variant: str = "raw") -> dict:
    """complete metric report for one config on one (filtered) score set."""
    sub = scores[scores["config"] == config_name]
    y_true = sub["y_true"].astype(int).tolist()
    y_score = sub["likelihood"].astype(float).tolist()
    if not y_true:
        return {"config": config_name, "error": "no scores"}
    report = M.full_case_report(y_true, y_score, threshold=threshold, label_variant=label_variant)
    n_pos = int(sub["y_true"].sum())
    if n_pos > 0:
        pos = sub[sub["y_true"] == 1]
        lesion_stats = {
            "n_positive_cases": len(pos),
            "n_positive_with_gt_mask": int(pos["gt_mask_available"].fillna(False).sum()),
            "n_overlap10": int(pos["overlap_10pct"].fillna(False).sum()),
            "lesion_detection_rate_10pct": (int(pos["overlap_10pct"].fillna(False).sum())
                                            / int(pos["gt_mask_available"].fillna(False).sum())
                                            if pos["gt_mask_available"].fillna(False).sum() else None),
            "mean_iou_positives": (float(pos["iou"].dropna().mean())
                                   if not pos["iou"].dropna().empty else None),
            "mean_dice_positives": (float(pos["dice"].dropna().mean())
                                    if not pos["dice"].dropna().empty else None),
        }
        report["lesion_localization"] = lesion_stats
    return report


def save_experiment(name: str, payload: dict) -> Path:
    out_dir = EXPERIMENTS_RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "metrics.json"
    if out.exists():  # merge multiple configs into one result file
        existing = json.loads(out.read_text(encoding="utf-8"))
        existing.update(payload)
        payload = existing
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return out
