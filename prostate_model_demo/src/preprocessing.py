"""Preprocessing of PI-CAI examinations — replicates the official pipeline exactly.

Official inference script (source of every step, verbatim):
    https://github.com/DIAGNijmegen/picai_unet_gc_algorithm/blob/main/process.py

Pipeline (axial MRI, modified from PicaiBaseline/Preprocessing.py of
the PI-CAI challenge):

    raw MRI (T2W, ADC, HBV on their native grids)
        -> simple center-of-mass alignment (picai_prep Sample.preprocess())
        -> center crop / pad to image_shape = (20, 256, 256)
        -> resample to spacing = (3.0, 0.5, 0.5) mm
        -> per-modality z-score normalization (clipped at percentile 99.5)
        -> stack into 3-channel volume (1, 3, 20, 256, 256)

We reuse the exact same library functions as the official script
(picai_prep.preprocessing Sample/PreprocessingSettings and
picai_baseline.unet.training_setup.preprocess_utils.z_score_norm)
rather than reimplementing them.
"""

from __future__ import annotations

import SimpleITK as sitk
import numpy as np
import torch

from picai_baseline.unet.training_setup.preprocess_utils import z_score_norm
from picai_prep.preprocessing import PreprocessingSettings, Sample

from .config import IMG_SPEC


def preprocess_images(scans: list[sitk.Image]) -> np.ndarray:
    """Run the official preprocessing on raw SimpleITK scans.

    Parameter
    ---------
    scans: list of the three axial scans in order [t2w, adc, hbv].

    Returns
    -------
    (1, 3, 20, 256, 256) float32 numpy array, the model input tensor.
    """
    sample = Sample(
        scans=scans,
        settings=PreprocessingSettings(
            matrix_size=IMG_SPEC["image_shape"],
            spacing=IMG_SPEC["spacing"],
        ),
    )

    # align, center-crop, resample
    sample.preprocess()
    cropped_img = [sitk.GetArrayFromImage(x) for x in sample.scans]

    # intensity normalization + channel dim
    preproc_img = [z_score_norm(np.expand_dims(x, axis=0), percentile=99.5) for x in cropped_img]

    # concatenate channels + add batch dim: (1, 3, 20, 256, 256)
    return np.expand_dims(np.vstack(preproc_img), axis=0)


def make_tensor(preproc_img: np.ndarray, device: str) -> torch.Tensor:
    """Convert the preprocessed array into the torch input tensor."""
    return torch.from_numpy(preproc_img).to(device)
