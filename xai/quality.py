"""Quality of XAI explanations — do NOT trust a pretty heatmap.

Implemented checks (each documented in reports):
1. localization agreement with expert annotations:
     - top-5% attribution mask vs lesion mask: IoU
     - attribution mass inside the lesion vs lesion volume share (enrichment)
2. insertion / deletion curves (faithfulness, on the fold-averaged raw fused
   score; blurred baseline for insertion, original for deletion, AU(N)):
3. randomization sanity check (per Adebayo et al. 2018): randomize the final
   up-path weights of one fold; a real CAM must change materially;
4. stability: Spearman correlation of the attribution by +/-1-voxel jittered
   inputs (0 injection of noise; pure geometry).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from xai.methods import (
    ensemble_gradcam,
    fused_score_from_batch,
)


# ---------------------------------------------------------------------------
# 1. localization agreement
# ---------------------------------------------------------------------------

def localization_agreement(heatmap: np.ndarray, gt_mask: np.ndarray,
                           top_frac: float = 0.05) -> dict:
    """Compare top-q% attribution voxels with the expert lesion mask.

    Works on a common 3D grid: both arrays are aligned to the model grid by
    the caller (the annotation famously lives on the native grid, which for
    PI-CAI cases is registered to the model grid approximately; we resample
    both to a common grid at call site).
    gt_mask: bool-ish array (0/labels>0 = lesion) on the same grid.
    """
    h = np.asarray(heatmap, dtype=np.float32)
    g = np.asarray(gt_mask) > 0
    n = h.size
    thr = np.quantile(h, 1.0 - top_frac)
    top = h >= thr
    inter = int(np.logical_and(top, g).sum())
    union = int(np.logical_or(top, g).sum())
    iou = inter / union if union else 0.0

    # precision: share of top voxels inside the lesion
    denom_top = int(top.sum())
    precision_in_lesion = inter / denom_top if denom_top else 0.0

    # enrichment: P	top|lesion) / P(lesion)
    lesion_share = g.mean()
    top_in_lesion_share = (top & g).sum() / denom_top if denom_top else 0.0
    enrichment = (top_in_lesion_share / lesion_share) if lesion_share > 0 else None

    return {"iou_top_vs_lesion": float(iou),
            "precision_top_in_lesion": float(precision_in_lesion),
            "enrichment": (None if enrichment is None else float(enrichment)),
            "lesion_volume_share": float(lesion_share)}


# ---------------------------------------------------------------------------
# 2. insertion / deletion curves (faithfulness)
# ---------------------------------------------------------------------------

def _blur_baseline(x: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Blurred reference image on the (1, 3, d, h, w) preprocessed tensor."""
    from scipy.ndimage import gaussian_filter
    b = np.asarray(x, dtype=np.float32).copy()
    for c in range(b.shape[1]):
        b[0, c] = gaussian_filter(b[0, c], sigma=sigma)
    return b


def insertion_deletion(models, input_tensor: np.ndarray, attribution: np.ndarray,
                       device: str, steps: int = 16) -> dict:
    """Faithfulness curves for ONE attribution map (spatial, (d,h,w)).

    Score = fold-averaged max of the fused csPCa probability map (fast proxy).
    insertion: start from blurred baseline and insert pixels in DESCENDING
    attribution order; deletion: start from the original and delete in the
    SAME order. Returns AUC Domain curves + the per-step score series.
    """
    order = np.argsort(np.asarray(attribution, dtype=np.float32).ravel())[::-1]
    base = _blur_baseline(input_tensor)
    n = np.prod(attribution.shape)
    step = max(n // steps, 1)

    ins_scores, del_scores = [], []
    for k in range(steps + 1):
        keep_set = order[: k * step]
        x_ins = base.copy()
        x_del = input_tensor.copy()
        src = input_tensor.reshape(3, -1)
        dst_ins = x_ins.reshape(3, -1)
        dst_del = x_del.reshape(3, -1)
        ins_base = base.reshape(3, -1)
        dst_ins[:, keep_set] = src[:, keep_set]
        dst_del[:, keep_set] = ins_base[:, keep_set]
        x_ins = dst_ins.reshape(base.shape)
        x_del = dst_del.reshape(base.shape)
        with torch.no_grad():
            ins_scores.append(
                float(fused_score_from_batch(models, torch.from_numpy(x_ins).float(), device)[0])
            )
            del_scores.append(
                float(fused_score_from_batch(models, torch.from_numpy(x_del).float(), device)[0])
            )

    def curve_auc(series):
        s = np.asarray(series, dtype=np.float64)
        rng = s.max() - s.min()
        if rng < 1e-9 or len(s) < 2:
            return float("nan")
        ys = (s - s.min()) / rng
        trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(trapz(ys) / (len(s) - 1))

    return {"insertion_scores": ins_scores, "deletion_scores": del_scores,
            "insertion_auc": curve_auc(ins_scores), "deletion_auc": curve_auc(del_scores)}


# ---------------------------------------------------------------------------
# 3. randomization sanity check (Adebayo et al. 2018)
# ---------------------------------------------------------------------------

def randomization_test(models, input_tensor, device: str, method: str = "gradcam",
                       seed: int = 2026) -> dict:
    """Randomize the final conv weights of fold 0 and rerun the CAM.

    A faithful CAM must DETIORATE (higher dissimilarity) after randomization.
    Returns {original_cam, randomized_cam, similarity (Pearson r)}.
    """
    m0 = models[0]
    ref = ensemble_gradcam(models, torch.from_numpy(input_tensor).float(),
                           device, method=method)
    ang = copy.deepcopy(m0)
    target_dev = next(m0.parameters()).device
    g = torch.Generator(device=target_dev.type).manual_seed(seed)
    with torch.no_grad():
        for p in ang.model[-1].parameters():
            p.copy_(torch.empty_like(p).uniform_(-1.0, 1.0, generator=g) * 0.1)
    rand = gradcam_single(ang, torch.from_numpy(input_tensor).float(), method)
    a = ref.ravel()
    b = rand.ravel()
    a -= a.mean()
    b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    sim = float(np.dot(a, b) / den)
    return {"cam_original_maxcam_ref": float(ref.max()), "cam_randomized_max": float(rand.max()),
            "pearson_similarity_orig_vs_randomized": sim}


def gradcam_single(model, input_tensor: torch.Tensor, method: str = "gradcam") -> np.ndarray:
    from xai.methods import gradcam_3d
    dev = next(model.parameters()).device
    return gradcam_3d(model, input_tensor.to(dev), method=method)


# ---------------------------------------------------------------------------
# 4. stability (+/- 1-voxel shift)
# ---------------------------------------------------------------------------

def stability_jitter(models, input_tensor: np.ndarray, device: str,
                     method: str = "gradcam") -> dict:
    """Pearson between the attribution of the original input and the by-1-voxel
    shifted input (no noise added — pure geometry)."""
    t = torch.from_numpy(input_tensor).float()
    ref = ensemble_gradcam(models, t, device, method=method)
    tt = torch.from_numpy(np.roll(input_tensor, 1, axis=-2)).float()
    alt = ensemble_gradcam(models, tt, device, method=method)
    a = ref.ravel()
    b = alt.ravel()
    a -= a.mean()
    b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    r = float(np.dot(a, b) / den)
    return {"pearson_shift_y": r, "ref_max": float(ref.max()), "alt_max": float(alt.max())}
