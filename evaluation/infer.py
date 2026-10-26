"""Single-pass inference engine for the official PI-CAI 5-fold ensemble.

For every case we run the OFFICIAL preprocessing and the OFFICIAL per-fold
inference pipeline (TTA flip, sigmoid, gaussian smoothing sigma=1.5, per-fold
TTA averaging) — but we SAVE the intermediate components instead of only the
final fused output, so that every post-processing variant (fold subset, sigma,
TTA on/off, lesion-candidate extraction, likelihood aggregation) can be
evaluated OFFLINE afterwards without touching the GPU again.

Per case stored (data/predictions/<split>/<case_id>.npz):
    folds_orig   (5, 20, 256, 256) float16 — per-fold sigmoid maps from the
                 unflipped input, BEFORE gaussian smoothing
    folds_flip   (5, 20, 256, 256) float16 — per-fold sigmoid maps from the
                 horizontally flipped input, already flipped back, pre-smoothing
    native_fused (z, y, x) float32 — official fused heatmap (per-fold TTA mean,
                 sigma 1.5, mean over folds) after reverse crop/pad + reverse
                 resampling = exactly what process.py feeds to
                 extract_lesion_candidates (kept at float32 precision so the
                 official likelihood reproduces the Phase-1 golden values)
    meta fields (in the .json): spacing, native shape, timings, folds used

The OFFICIAL case-level likelihood (max of the 'dynamic'-extracted detection
map after the max/5 ring suppression) is recomputed from native_fused in
evaluation/postproc.py and was verified against the Phase-1 golden test on
case 10032_1000032.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from picai_prep.preprocessing import crop_or_pad, resample_img
from scipy.ndimage import gaussian_filter

from evaluation.paths import (
    DEMO_DIR,
    IMAGES_DIR,
    IMG_SPEC,
    PREDICTIONS_DIR,
    UNET_CHANNELS,
    UNET_STRIDES,
    WEIGHTS_DIR,
)

sys.path.insert(0, str(DEMO_DIR))  # reuse the Phase-1 inference stack verbatim

from src.inference import get_device, load_models  # noqa: E402
from src.preprocessing import make_tensor, preprocess_images  # noqa: E402

SIGMA_OFFICIAL = 1.5


def case_image_paths(case_id: str) -> dict[str, Path]:
    patient_dir = IMAGES_DIR / case_id.split("_")[0]
    return {m: patient_dir / f"{case_id}_{m}.mha" for m in ("t2w", "adc", "hbv")}


def load_case_images(case_id: str) -> tuple[sitk.Image, sitk.Image, sitk.Image]:
    paths = case_image_paths(case_id)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing images for {case_id}: {missing}")
    return tuple(sitk.ReadImage(str(paths[m])) for m in ("t2w", "adc", "hbv"))


def fold_maps(preproc_tensor: torch.Tensor, models: list, device: str):
    """Official per-fold computation, keeping the two TTA components separate.

    Returns (orig_maps, flip_maps): per-fold arrays of sigmoid(csPCa) maps,
    in model load order. flip maps come from the horizontally-flipped input
    and are already flipped back (official process.py behavior).
    """
    orig_maps, flip_maps = [], []
    img_for_pred = [preproc_tensor.to(device), torch.flip(preproc_tensor, [4]).to(device)]
    with torch.no_grad():
        for model in models:
            preds = [
                torch.sigmoid(model(x))[:, 1, ...].detach().cpu().numpy() for x in img_for_pred
            ]
            preds[1] = np.flip(preds[1], [3])   # (1,20,256,256): axis 3 = x-axis
            orig_maps.append(preds[0][0].astype(np.float32))   # -> (20, 256, 256)
            flip_maps.append(preds[1][0].astype(np.float32))
    return orig_maps, flip_maps


def reverse_to_native(raw_heatmap: np.ndarray, t2w_sitk: sitk.Image) -> np.ndarray:
    """Official reverse crop/pad + reverse resampling applied to a (20,256,256)
    model-grid heatmap; output is on the native T2W grid (z, y, x).

    This is the step of process.py performed just BEFORE extract_lesion_candidates
    (which itself is delegated to evaluation/postproc.py to allow variants).
    """
    spacing = IMG_SPEC["spacing"]
    arr = sitk.GetArrayFromImage(t2w_sitk)  # (z, y, x)
    native_spacing_zyx = [
        t2w_sitk.GetSpacing()[2], t2w_sitk.GetSpacing()[1], t2w_sitk.GetSpacing()[0]
    ]
    resampled_shape = tuple(
        int(np.round(arr.shape[i] * native_spacing_zyx[i] / spacing[i])) for i in range(3)
    )
    cspca_det_map_sitk = sitk.GetImageFromArray(crop_or_pad(raw_heatmap, size=resampled_shape))
    cspca_det_map_sitk.SetSpacing(list(reversed(spacing)))
    cspca_det_map_sitk = resample_img(
        cspca_det_map_sitk, out_spacing=list(reversed(t2w_sitk.GetSpacing()))
    )
    out = sitk.GetArrayFromImage(cspca_det_map_sitk)
    return np.asarray(crop_or_pad(out, size=arr.shape), dtype=np.float32)


def official_fused_model_grid(orig: np.ndarray, flip: np.ndarray,
                              fold_idx: list[int] | None = None,
                              sigma: float = SIGMA_OFFICIAL,
                              tta: bool = True) -> np.ndarray:
    """Compose the fused (20,256,256) model-grid heatmap from saved components.

    fold_idx selects the fold subset, sigma the gaussian strength, TTA toggles
    the flip component. Defaults reproduce process.py exactly.
    """
    idx = list(range(len(orig))) if fold_idx is None else list(fold_idx)
    per_fold = []
    for i in idx:
        comp = 0.5 * (orig[i] + flip[i]) if tta else orig[i]
        per_fold.append(gaussian_filter(comp, sigma=sigma))
    fused = np.mean(per_fold, axis=0).astype(np.float32)
    return fused


def run_inference_pass(split: str = "val", case_ids: list[str] | None = None,
                       folds: list[int] | None = None, device: str | None = None,
                       out_root: Path | None = None, log_every: int = 10,
                       save_raw: bool = True) -> dict:
    """Run the single inference pass for one split (resumable case by case).

    Saves data/predictions/<split>/<case_id>.npz (+ timing .json per case).
    """
    device = device or get_device()[0]
    out_root = out_root or (PREDICTIONS_DIR / split)
    out_root.mkdir(parents=True, exist_ok=True)

    models, used_folds = load_models(device=device, weights_dir=WEIGHTS_DIR, folds=folds)
    n_done = 0
    t_total = time.time()
    for n, case_id in enumerate(case_ids, start=1):
        out_npz = out_root / f"{case_id}.npz"
        if out_npz.exists():
            n_done += 1
            continue
        try:
            t0 = time.perf_counter()
            t2w, adc, hbv = load_case_images(case_id)
            preproc = preprocess_images([t2w, adc, hbv])
            tensor = make_tensor(preproc, device)
            t_preproc = time.perf_counter() - t0

            t0 = time.perf_counter()
            orig, flip = fold_maps(tensor, models, device)
            t_gpu = time.perf_counter() - t0

            t0 = time.perf_counter()
            per_fold_gauss = [
                gaussian_filter(0.5 * (o + f), sigma=SIGMA_OFFICIAL)
                for o, f in zip(orig, flip)
            ]
            fused_model_grid = np.mean(per_fold_gauss, axis=0).astype(np.float32)
            native_fused = reverse_to_native(fused_model_grid, t2w)
            t_post = time.perf_counter() - t0

            payload = {
                "meta_spacing_xyz": np.asarray([float(s) for s in t2w.GetSpacing()], dtype=np.float32),
                "native_fused": native_fused.astype(np.float32),  # fp32: golden fidelity
            }
            if save_raw:
                payload["folds_orig"] = np.asarray(orig, dtype=np.float16)
                payload["folds_flip"] = np.asarray(flip, dtype=np.float16)
            np.savez(out_npz, **payload)

            meta = {
                "case_id": case_id,
                "split": split,
                "device": device,
                "folds_used": used_folds,
                "s_preprocess": round(t_preproc, 3),
                "s_inference": round(t_gpu, 3),
                "s_reverse_postprocess": round(t_post, 3),
                "native_shape_zyx": [int(x) for x in sitk.GetArrayFromImage(t2w).shape],
            }
            with open(out_root / f"{case_id}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            n_done += 1
            elapsed = time.time() - t_total
            if n_done % log_every == 0 or n_done == 1:
                rate = n_done / max(elapsed, 1e-9)
                eta = (len(case_ids) - n_done) / max(rate, 1e-9)
                print(f"[infer] {case_id} case {n_done}/{len(case_ids)} "
                      f"(preproc {t_preproc:.1f}s gpu {t_gpu:.1f}s post {t_post:.1f}s) "
                      f"rate {rate:.2f} cps eta {eta / 60:.1f} min", flush=True)
        except Exception as e:  # noqa: BLE001
            (out_root / f"{case_id}.ERROR.txt").write_text(
                f"{type(e).__name__}: {e}\n", encoding="utf-8"
            )
            print(f"[infer][ERROR] {case_id}: {type(e).__name__}: {e}", flush=True)
    minutes = (time.time() - t_total) / 60
    return {
        "split": split,
        "n_cases_requested": len(case_ids),
        "n_cases_saved": n_done,
        "minutes": round(minutes, 2),
        "folds_used": used_folds,
    }
