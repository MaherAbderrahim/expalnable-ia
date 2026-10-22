"""Simple, readable visualization of model predictions (INPUT + MODEL OUTPUT).

Phase 1 of the project intentionally does NOT yet contain any XAI method
(Grad-CAM, Integrated Gradients, ...). Everything shown here is:

    the input MRI            (what the model sees)
    the model's own outputs  (raw ensemble heatmap + final detection map)
    the expert annotation    (ground truth, when available)

A later phase will add explainability on top of the same outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def _best_slice(det_map: np.ndarray | None, n_slices: int) -> int:
    """Axial slice to display.

    If the model shows a meaningful response (max > 0.1), show the slice of
    that response; otherwise show the middle slice (visual aid only — the
    model's case-level decision uses the full 3D volume).
    """
    if det_map is not None and det_map.shape[0] == n_slices and np.max(det_map) > 0.1:
        scores = np.max(det_map, axis=(1, 2))
        return int(np.argmax(scores))
    return n_slices // 2


def _norm_slice(vol: np.ndarray, z: int, p=(1, 99)) -> np.ndarray:
    """Percentile-normalized slice; zero-padding voxels do not skew the stretch."""
    sl = vol[z].astype(np.float32)
    vals = sl[sl > 0]
    if vals.size < 100:
        vals = sl
    lo, hi = np.percentile(vals, p)
    if hi > lo:
        sl = np.clip(sl, lo, hi)
        sl = (sl - lo) / (hi - lo)
    return sl


def _overlay(ax, t2w_slice, value_slice, title: str, alpha: float = 0.55):
    ax.imshow(t2w_slice, cmap="gray", origin="lower")
    d = np.ma.masked_less(value_slice, 1e-6)
    im = ax.imshow(d, cmap="jet", vmin=0, vmax=1, alpha=alpha, origin="lower")
    ax.set_title(title)
    return im


def case_figure(
    case,
    result=None,
    save_path: Path | None = None,
    title: str | None = None,
) -> Path | None:
    """Figure for one case:
      row 1: the three input modalities (T2W, ADC, DWI/HBV), native resolution
      row 2: model detection map overlay, raw ensemble heatmap overlay,
             expert annotation contour overlay (when available)
    """
    t2w = sitk.GetArrayFromImage(case.images["t2w"])   # (z, y, x)
    adc = sitk.GetArrayFromImage(case.images["adc"])
    hbv = sitk.GetArrayFromImage(case.images["hbv"])

    det = None
    raw = None
    if result is not None:
        det = result.detection_map if result.detection_map.shape == t2w.shape else None
        raw = result.raw_softmax_heatmap

    z = _best_slice(det, t2w.shape[0])

    # expert ground truth on the T2W grid, if loaded
    gt = None
    if getattr(case, "lesion_annotation", None) is not None:
        gt_arr = sitk.GetArrayFromImage(case.lesion_annotation)
        if gt_arr.shape == t2w.shape:
            gt = gt_arr

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    t2w_img = _norm_slice(t2w, z)

    # ---- row 1: inputs ---------------------------------------------------
    axes[0, 0].imshow(t2w_img, cmap="gray", origin="lower")
    axes[0, 0].set_title(f"T2W input (native, slice {z})")
    rel_z = z / max(t2w.shape[0], 1)
    for col, (name, vol) in enumerate([("ADC input", adc), ("DWI/HBV input", hbv)], start=1):
        zz = int(np.clip(np.round(rel_z * vol.shape[0]), 0, vol.shape[0] - 1))
        axes[0, col].imshow(_norm_slice(vol, zz), cmap="gray", origin="lower")
        axes[0, col].set_title(f"{name} (slice {zz})")

    # ---- row 2: outputs --------------------------------------------------
    # panel (1,0): detection map (+ GT contour for direct comparison)
    if det is not None:
        meaningful = np.max(det) > 0.1
        det_title = (f"model detection map @ max response (slice {z})" if meaningful
                     else f"model detection map @ middle slice (slice {z})")
        im = _overlay(axes[1, 0], t2w_img, det[z], det_title)
        if gt is not None and gt.shape == t2w.shape:
            axes[1, 0].contour(gt[z] > 0, levels=[0.5], colors="lime", linewidths=1.5)
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    else:
        axes[1, 0].imshow(t2w_img, cmap="gray", origin="lower")
        axes[1, 0].set_title("model detection map: no inference result")

    # panel (1,1): raw ensemble heatmap (model grid) on its own slice
    if raw is not None:
        zz = int(np.clip(np.round(z / max(t2w.shape[0], 1) * raw.shape[0]), 0, raw.shape[0] - 1))
        im = axes[1, 1].imshow(_norm_slice(raw, zz, p=(0, 99.9)), cmap="inferno",
                               origin="lower", vmin=0, vmax=1)
        axes[1, 1].set_title(f"raw ensemble heatmap (resampled grid, slice {zz})")
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    else:
        axes[1, 1].set_title("raw ensemble heatmap: no inference result")

    # panel (1,2): expert annotation at the slice of the lesion's maximum extent,
    # with the detection map contour drawn at the SAME slice for comparison
    if gt is not None and np.max(gt) > 0:
        z_gt = int(np.argmax(gt.sum(axis=(1, 2))))
        gt_img = _norm_slice(t2w, z_gt)
        axes[1, 2].imshow(gt_img, cmap="gray", origin="lower")
        axes[1, 2].contour(gt[z_gt] > 0, levels=[0.5], colors="lime", linewidths=2.0)
        if det is not None and det.shape == t2w.shape:
            axes[1, 2].contour(det[z_gt] > 0, levels=[0.5], colors="cyan", linewidths=1.2)
        axes[1, 2].set_title(f"expert lesion (lime) vs detection (cyan), slice {z_gt}")
    else:
        axes[1, 2].imshow(t2w_img, cmap="gray", origin="lower")
        if gt is not None:
            axes[1, 2].set_title("expert annotation: negative case (no lesion)")
        else:
            axes[1, 2].set_title("expert annotation: not available")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title or f"PI-CAI case {case.case_id}")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return None
