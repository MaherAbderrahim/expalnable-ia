"""XAI visualizations: original image + model detection + XAI heatmaps.

Single-row figure layout (left to right):
    T2W input | detection overlay | Grad-CAM | Grad-CAM++ | IG | [occlusion]

All overlays are aligned to the native T2W grid: heatmaps computed on the
model grid (20, 256, 256) are trilinearly zoomed to the native (y, x) size,
which is the same forward mapping the model applies (documented as an
approximation: the reverse crop/pad origin is shared by construction).
Display scale is capped at the 99.5th percentile of the slice so that a
single speckle cannot saturate the color range (Integrated Gradients in
particular has isolated extreme voxels).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


METHOD_ORDER = ["gradcam", "gradcampp", "integrated_gradients", "occlusion"]


def _norm_slice(vol: np.ndarray, z: int, p=(1, 99)) -> np.ndarray:
    sl = vol[z].astype(np.float32)
    vals = sl[sl > 0]
    if vals.size < 100:
        vals = sl
    lo, hi = np.percentile(vals, p)
    if hi > lo:
        sl = np.clip(sl, lo, hi)
        sl = (sl - lo) / (hi - lo)
    return sl


def _to_native(heat: np.ndarray, native_shape_zyx: tuple) -> np.ndarray:
    """Zoom a model-grid heatmap (z, y, x) to the native grid (order=1)."""
    scale = [n / m for n, m in zip(native_shape_zyx, heat.shape)]
    return zoom(heat.astype(np.float32), scale, order=1)


def _display_clamp(overlay: np.ndarray, pct: float = 99.5) -> np.ndarray:
    mx = np.percentile(overlay, pct)
    if mx <= 0:
        return overlay
    return np.clip(overlay, 0.0, mx) / mx


def _peak_slice(mask_or_heat: np.ndarray, fallback: int) -> int:
    flat = mask_or_heat.reshape(mask_or_heat.shape[0], -1)
    if np.max(mask_or_heat) <= 0.05 * np.max(np.abs(mask_or_heat) + 1e-12):
        return fallback
    return int(np.argmax(flat.max(axis=1)))


def explain_figure(t2w_native: np.ndarray, det_map: np.ndarray | None,
                   heatmaps: dict[str, np.ndarray], case_info: dict,
                   save_path: Path | None = None, dpi: int = 140):
    """Build (and optionally save) the one-row explanation figure.

    t2w_native: (z, y, x) native T2W volume.
    det_map:    (z, y, x) native detection map (post extract_lesion_candidates)
                or None.
    heatmaps:   name -> (20, 256, 256) model-grid attribution map.
    case_info:  {case_id, likelihood, threshold, y_true}.
    """
    nz, ny, nx = t2w_native.shape
    methods = [k for k in METHOD_ORDER if k in heatmaps]
    ncols = len(methods) + 2
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 3.8))

    z0 = nz // 2
    # reference slice: the detection/attribution-rich slice if any
    ref_heat = det_map if det_map is not None else next(iter(heatmaps.values()))
    z_ref = _peak_slice(_to_native(ref_heat, t2w_native.shape), z0) \
        if det_map is None else z0
    if det_map is not None and det_map.shape == t2w_native.shape:
        z_ref = _peak_slice(det_map, z0)

    base = _norm_slice(t2w_native, z_ref)

    # 1. input
    axes[0].imshow(base, cmap="gray", origin="lower")
    axes[0].set_title(f"T2W input (slice {z_ref}/{nz})")

    # 2. detection overlay (native grid == t2w grid)
    if det_map is not None and det_map.shape == t2w_native.shape:
        d = _display_clamp(det_map[z_ref])
        axes[1].imshow(base, cmap="gray", origin="lower")
        axes[1].imshow(np.ma.masked_less(d, 1e-3), cmap="jet", vmin=0, vmax=1,
                       alpha=0.55, origin="lower")
        axes[1].set_title(f"detection  max={det_map.max():.2f}")
    else:
        axes[1].imshow(base, cmap="gray", origin="lower")
        axes[1].set_title("detection overlay n/a")

    # 3+. XAI methods (model grid -> native overlay)
    for col, name in enumerate(methods, start=2):
        h = heatmaps[name]
        native_h = _to_native(h, t2w_native.shape)
        zz = _peak_slice(native_h, z_ref)
        ov = _display_clamp(native_h[zz])
        axes[col].imshow(_norm_slice(t2w_native, zz), cmap="gray", origin="lower")
        axes[col].imshow(np.ma.masked_less(ov, 0.05), cmap="inferno", vmin=0, vmax=1,
                         alpha=0.6, origin="lower")
        axes[col].set_title(f"{name} (slice {zz})")

    label = "csPCa+" if case_info.get("y_true") == 1 else "csPCa-"
    lik = case_info.get("likelihood")
    thr = case_info.get("threshold")
    pred = "det" if (lik is not None and thr is not None and lik >= thr) else "no-det"
    fig.suptitle(
        f"{case_info.get('case_id')} | likelihood {lik if lik is not None else '-'} "
        f"@ thr {thr} -> {pred} | GT {label}",
        fontsize=11,
    )
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return None
