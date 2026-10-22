"""Inference with the PI-CAI reference U-Net — replicates the official predict() exactly.

Official inference script (source of every step, verbatim):
    https://github.com/DIAGNijmegen/picai_unet_gc_algorithm/blob/main/process.py

Summary of what happens (see docs/MODEL_EXPLANATION.md for details):

    input tensor (1, 3, 20, 256, 256)
        + test-time augmentation (horizontal flip)
        -> 5 U-Net members (5-fold cross-validation ensemble)
        -> sigmoid, keep csPCa channel (index 1)
        -> gaussian smoothing (sigma 1.5, anti checkerboard artifact)
        -> mean over TTA variants, then mean over the 5 folds
        -> raw ensemble heatmap (20, 256, 256)
        -> reverse crop/pad + reverse resampling to the native T2W grid
        -> lesion candidate extraction with a 'dynamic' threshold
           (package report_guided_annotation) -> final detection map
        -> case-level likelihood = max of the detection map

The TTA flip is applied on axis 4 of the (1, 3, 20, 256, 256) tensor,
equivalent to flipping the axial transverse image left-right.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import crop_or_pad, resample_img
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter

from .config import IMG_SPEC, N_FOLDS, WEIGHTS_DIR
from .preprocessing import preprocess_images, make_tensor
from .unet_arch import UNet

# Architecture hyperparameters, verbatim from picai_baseline
# (picai_baseline/unet/training_setup/default_hyperparam.py).
UNET_STRIDES = [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
UNET_CHANNELS = [32, 64, 128, 256, 512, 1024]


@dataclass
class InferenceResult:
    case_id: str
    device: str
    n_models_used: int
    preprocessed_shape: tuple
    t2w_native_shape: tuple
    t2w_spacing: tuple                       # SimpleITK order (x, y, z)
    case_level_likelihood: float             # max of the detection map, in [0, 1]
    detection_map: np.ndarray | None = None  # on the native T2W grid (z, y, x)
    raw_softmax_heatmap: np.ndarray | None = None  # (20, 256, 256) ensemble output
    preprocessing_time_s: float = 0.0
    inference_time_s: float = 0.0
    model_load_time_s: float = 0.0


def get_device(prefer_gpu: bool = True) -> tuple[str, str]:
    """Pick the compute device and return (device, human-readable note)."""
    if prefer_gpu and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return "cuda", f"CUDA ({name}, {vram:.1f} GB)"
    return "cpu", "CPU (inference will be slower)"


def load_models(
    device: str = "cuda",
    weights_dir: Path | None = None,
    folds: list[int] | None = None,
) -> tuple[list, list[int]]:
    """Instantiante the official UNet(s) and load the released checkpoints.

    folds=None loads all 5 folds (the official configuration). A subset,
    e.g. folds=[0], may be used for quick smoke tests.
    Returns (models, folds_actually_loaded).
    """
    weights_dir = Path(weights_dir) if weights_dir is not None else WEIGHTS_DIR
    folds = list(range(N_FOLDS)) if folds is None else folds

    models = []
    loaded = []
    t0 = time.perf_counter()
    for fold in folds:
        checkpoint_path = weights_dir / f"unet_F{fold}.pt"
        if not checkpoint_path.exists():
            continue
        model = UNet(
            spatial_dims=len(IMG_SPEC["image_shape"]),
            in_channels=IMG_SPEC["num_channels"],
            out_channels=IMG_SPEC["num_classes"],
            strides=UNET_STRIDES,
            channels=UNET_CHANNELS,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
        loaded.append(fold)
    model_load_s = time.perf_counter() - t0

    if len(models) == 0:
        raise FileNotFoundError(
            f"No model weights found in {weights_dir}. "
            "Git LFS pointer files are not weights: run 'git lfs pull' to get the "
            "real .pt files (each should be ~127,229,381 bytes)."
        )
    return models, loaded


def predict(preproc_tensor: torch.Tensor, models: list, device: str) -> np.ndarray:
    """Run one case through the ensemble (official logic, verbatim).

    Returns the raw ensemble heatmap of shape (20, 256, 256).
    """
    # test-time augmentation (horizontal flipping)
    img_for_pred = [preproc_tensor.to(device)]
    img_for_pred += [torch.flip(preproc_tensor, [4]).to(device)]

    outputs = []
    with torch.no_grad():
        for model in models:
            preds = [
                torch.sigmoid(model(x))[:, 1, ...].detach().cpu().numpy() for x in img_for_pred
            ]
            preds[1] = np.flip(preds[1], [3])
            outputs.append(np.mean([gaussian_filter(x, sigma=1.5) for x in preds], axis=0)[0])

    return np.mean(outputs, axis=0).astype("float32")


def run_inference(
    case,
    device: str = "cuda",
    folds: list[int] | None = None,
    weights_dir: Path | None = None,
    models: list | None = None,
    model_load_time_s: float = 0.0,
) -> InferenceResult:
    """Preprocessed-image -> likelihood + detection map for one case.

    ``case`` is a ``PicaIcase`` from src.data_loader. Models can be reused
    across cases (pass ``models=...``) to skip repeated checkpoint loading.
    """
    scan_t2w, scan_adc, scan_hbv = case.images["t2w"], case.images["adc"], case.images["hbv"]

    if models is None:
        models, model_load_time_s = load_models(device=device, weights_dir=weights_dir, folds=folds)

    t0 = time.perf_counter()
    preproc_img = preprocess_images([scan_t2w, scan_adc, scan_hbv])
    tensor = make_tensor(preproc_img, device)
    preprocessing_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw_heatmap = predict(tensor, models, device)
    inference_s = time.perf_counter() - t0

    raw_heatmap_582 = raw_heatmap  # raw ensemble output, (20, 256, 256)
    likelihood, det_map_zyx = _official_postprocess(raw_heatmap_582, scan_t2w)

    native_t2w = sitk.GetArrayFromImage(scan_t2w)
    return InferenceResult(
        case_id=case.case_id,
        device=device,
        n_models_used=len(models),
        preprocessed_shape=tuple(preproc_img.shape),
        t2w_native_shape=native_t2w.shape,
        t2w_spacing=scan_t2w.GetSpacing(),
        case_level_likelihood=likelihood,
        detection_map=det_map_zyx,
        raw_softmax_heatmap=raw_heatmap_582,
        preprocessing_time_s=preprocessing_s,
        inference_time_s=inference_s,
        model_load_time_s=model_load_time_s,
    )


def _official_postprocess(raw_heatmap: np.ndarray, t2w_sitk: sitk.Image) -> tuple[float, np.ndarray]:
    """Reverse the preprocessing and derive the final detection map (official logic).

    Steps copied verbatim from process.py:
      1. crop_or_pad back to the resampled T2W shape, assign model spacing
      2. resample back to the native T2W spacing
      3. extract lesion candidates with the 'dynamic' threshold
         (Report-Guided-Annotation): keeps intensity values in [0, 1]
      4. remove secondary concentric/ring detections below max/5
      5. crop_or_pad to the exact native T2W shape and copy T2W geometry

    Returns (case_level_likelihood, detection_map_zyx).
    """
    spacing = IMG_SPEC["spacing"]

    # shape of the resampled T2W at the model spacing (target of reverse crop)
    arr = sitk.GetArrayFromImage(t2w_sitk)          # (z, y, x)
    native_spacing_zyx = [
        t2w_sitk.GetSpacing()[2], t2w_sitk.GetSpacing()[1], t2w_sitk.GetSpacing()[0]
    ]
    resampled_shape = tuple(
        int(np.round(arr.shape[i] * native_spacing_zyx[i] / spacing[i])) for i in range(3)
    )

    # reverse center crop
    cspca_det_map_sitk = sitk.GetImageFromArray(crop_or_pad(raw_heatmap, size=resampled_shape))
    cspca_det_map_sitk.SetSpacing(list(reversed(spacing)))

    # reverse resampling
    cspca_det_map_sitk = resample_img(
        cspca_det_map_sitk, out_spacing=list(reversed(t2w_sitk.GetSpacing()))
    )

    # lesion candidate extraction
    det_map_npy = extract_lesion_candidates(
        sitk.GetArrayFromImage(cspca_det_map_sitk), threshold="dynamic"
    )[0]

    # remove (some) secondary concentric/ring detections
    det_map_npy[det_map_npy < (np.max(det_map_npy) / 5)] = 0

    # match the native shape exactly
    det_map_sitk = sitk.GetImageFromArray(
        crop_or_pad(det_map_npy, size=arr.shape)
    )
    det_map_sitk.CopyInformation(t2w_sitk)

    likelihood = float(np.max(sitk.GetArrayFromImage(det_map_sitk)))
    return likelihood, sitk.GetArrayFromImage(det_map_sitk)


def save_result(result: InferenceResult, out_dir: Path) -> None:
    """Persist one inference result (detection map .mha + prediction .json)."""
    out_dir = Path(out_dir)
    case_dir = out_dir / result.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    if result.detection_map is not None:
        img = sitk.GetImageFromArray(result.detection_map)
        img.SetSpacing([result.t2w_spacing[0], result.t2w_spacing[1], result.t2w_spacing[2]])
        atomic_image_write(img, str(case_dir / "cspca_detection_map.mha"))

    payload = {
        "case_id": result.case_id,
        "device": result.device,
        "n_models_used": result.n_models_used,
        "preprocessed_shape": [int(x) for x in result.preprocessed_shape],
        "t2w_native_shape": [int(x) for x in result.t2w_native_shape],
        "t2w_spacing_mm": [float(x) for x in result.t2w_spacing],
        "case_level_likelihood": result.case_level_likelihood,
        "preprocessing_time_s": result.preprocessing_time_s,
        "inference_time_s": result.inference_time_s,
        "model_load_time_s": result.model_load_time_s,
    }
    with open(case_dir / "prediction.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
