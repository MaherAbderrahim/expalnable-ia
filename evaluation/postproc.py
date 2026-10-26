"""Offline post-processing of saved inference components — official + variants.

The official PI-CAI baseline pipeline for the case-level likelihood is:

    native_fused (5-fold, TTA, sigma 1.5)
      -> extract_lesion_candidates(threshold='dynamic')      [Report-Guided-Annotation]
      -> det_map[det_map < max/5] = 0                        [ring suppression]
      -> case likelihood = max(det_map)

This module recomputes that official result EXACTLY from the saved npz, and
exposes a small pre-registered set of post-processing variants ("configs"):
fold subset, TTA on/off, smoothing sigma, extraction mode, aggregation.
Variants are selected ONLY on the validation split; the test split is scored
once with the frozen choice.

All geometry is defined by (native_shape_zyx, native_spacing_xyz) only — no
need to reload the raw scans.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from picai_prep.preprocessing import crop_or_pad, resample_img
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter

from evaluation.infer import SIGMA_OFFICIAL

MODEL_SPACING_ZYX = (3.0, 0.5, 0.5)   # IMG_SPEC spacing in array order (z, y, x)


# ---------------------------------------------------------------------------
# variant configs
# ---------------------------------------------------------------------------

OFFICIAL_CONFIG = {
    "name": "official_default",
    "folds": "all",       # 5-fold ensemble (official)
    "tta": True,          # official horizontal-flip TTA
    "sigma": SIGMA_OFFICIAL,  # official gaussian sigma=1.5 (model grid)
    "extract": "dynamic", # official lesion candidate extraction
    "suppress_ring": True,  # official det < max/5 -> 0
    "aggregate": "max",   # official max after extraction
}

VARIANT_CONFIGS = [
    OFFICIAL_CONFIG,
    # --- fold-count ablation (kept sigma/TTA/extraction as official) ---
    {**OFFICIAL_CONFIG, "name": "fold1", "folds": [0]},
    {**OFFICIAL_CONFIG, "name": "fold3", "folds": [0, 1, 2]},
    # --- TTA ablation ---
    {**OFFICIAL_CONFIG, "name": "no_tta", "tta": False},
    # --- smoothing sigma ablation ---
    {**OFFICIAL_CONFIG, "name": "sigma05", "sigma": 0.5},
    {**OFFICIAL_CONFIG, "name": "sigma10", "sigma": 1.0},
    # --- extraction/aggregation variants on the OFFICIAL fused heatmap ---
    {**OFFICIAL_CONFIG, "name": "extract_fixed01", "extract": 0.1},
    {**OFFICIAL_CONFIG, "name": "no_extract", "extract": "none"},
    {**OFFICIAL_CONFIG, "name": "agg_p90", "aggregate": "p90"},
    {**OFFICIAL_CONFIG, "name": "no_supp_max", "suppress_ring": False, "aggregate": "max"},
]


def resolve_folds(cfg: dict, n_folds_in_file: int = 5) -> list[int]:
    f = cfg["folds"]
    if f in ("all", None):
        return list(range(n_folds_in_file))
    return [min(int(i), n_folds_in_file - 1) for i in f]


# ---------------------------------------------------------------------------
# core computations
# ---------------------------------------------------------------------------

def fused_native_from_npz(npz: dict, cfg: dict) -> np.ndarray:
    """native-grid fused heatmap for a config. Uses the SAVED native_fused
    (official 5-fold/TTA/sigma1.5) when the config equals the official ensemble
    composition; otherwise rebuilds it from the per-fold components."""
    official_composition = (
        resolve_folds(cfg, npz["folds_orig"].shape[0]) == list(range(npz["folds_orig"].shape[0]))
        and cfg["tta"] and abs(float(cfg["sigma"]) - SIGMA_OFFICIAL) < 1e-6
    )
    if official_composition:
        fusion = np.asarray(npz["native_fused"], dtype=np.float32)
    else:
        fusion = fused_from_components(
            npz["folds_orig"], npz["folds_flip"], resolve_folds(cfg, npz["folds_orig"].shape[0]),
            float(cfg["sigma"]), bool(cfg["tta"]),
            _get_spacing(npz), _get_shape(npz),
        )
    return fusion


def fused_from_components(folds_orig: np.ndarray, folds_flip: np.ndarray,
                          fold_idx: list[int], sigma: float, tta: bool,
                          native_spacing_xyz, native_shape_zyx) -> np.ndarray:
    """Compose model-grid fused heat from saved per-fold components, then map
    to the native grid (reverse crop/pad + reverse resample) — process.py order."""
    per_fold = []
    for i in fold_idx:
        src = (0.5 * (folds_orig[i].astype(np.float32) + folds_flip[i].astype(np.float32))
               if tta else folds_orig[i].astype(np.float32))
        per_fold.append(gaussian_filter(src, sigma=sigma))
    fused = np.mean(per_fold, axis=0).astype(np.float32)

    native_spacing_zyx = (
        float(native_spacing_xyz[2]), float(native_spacing_xyz[1]), float(native_spacing_xyz[0])
    )
    resampled_shape = tuple(
        int(np.round(native_shape_zyx[i] * native_spacing_zyx[i] / MODEL_SPACING_ZYX[i]))
        for i in range(3)
    )
    m = sitk.GetImageFromArray(crop_or_pad(fused, size=resampled_shape))
    m.SetSpacing(list(reversed(MODEL_SPACING_ZYX)))
    m = resample_img(m, out_spacing=list(reversed([float(s) for s in native_spacing_xyz])))
    out = sitk.GetArrayFromImage(m)
    return np.asarray(crop_or_pad(out, size=tuple(native_shape_zyx)), dtype=np.float32)


def official_detection_map(native_fused: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Official lesion-candidate extraction on the native-grid fused heatmap."""
    cfg = cfg or OFFICIAL_CONFIG
    heat = np.asarray(native_fused, dtype=np.float32)
    if cfg["extract"] == "none":
        det = heat.copy()
    elif isinstance(cfg["extract"], (int, float)):
        det = extract_lesion_candidates(heat, threshold=float(cfg["extract"]))[0]
    else:
        det = extract_lesion_candidates(heat, threshold="dynamic")[0]
    if cfg.get("suppress_ring", True):
        det[det < (np.max(det) / 5)] = 0
    return det


def case_likelihood(det: np.ndarray, cfg: dict) -> float:
    """case-level likelihood for an extraction/aggregation variant."""
    agg = cfg.get("aggregate", "max")
    if agg == "max":
        return float(np.max(det))
    if agg == "p90":
        vals = det[det > 0]
        return float(np.percentile(vals, 90)) if vals.size else 0.0
    raise ValueError(f"unknown aggregation {agg}")


def _get_spacing(npz: dict) -> tuple[float, float, float]:
    sp = npz["meta_spacing_xyz"]
    return tuple(float(s) for s in sp)


def _get_shape(npz: dict) -> tuple[int, int, int]:
    n = npz["native_fused"].shape
    return tuple(int(x) for x in n)


# ---------------------------------------------------------------------------
# per-case scoring entry point
# ---------------------------------------------------------------------------

def score_case(npz: dict, cfg: dict, want_map: bool = False) -> dict:
    """Evaluate one config on one case (from the saved inference components).

    Returns {"likelihood": float, "det_map": np.ndarray | None}.With
    want_map=True the binarizable detection map (native grid) is returned for
    IoU/FROC/error analysis.
    """
    fusion = fused_native_from_npz(npz, cfg)
    det = official_detection_map(fusion, cfg)
    out = {"likelihood": case_likelihood(det, cfg), "det_map": None}
    if want_map:
        out["det_map"] = det
    return out
