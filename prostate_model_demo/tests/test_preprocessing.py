"""Test the official preprocessing pipeline (shapes + value ranges)."""

from __future__ import annotations

import numpy as np
import torch

from src.preprocessing import preprocess_images


def test_preprocess_output_shape(official_case):
    scans = [official_case.images[m] for m in ("t2w", "adc", "hbv")]
    out = preprocess_images(scans)
    # official spec: batch of 1, 3 modalities, 20 slices, 256 x 256
    assert out.shape == (1, 3, 20, 256, 256)
    assert out.dtype == np.float32


def test_preprocess_normalization(official_case):
    scans = [official_case.images[m] for m in ("t2w", "adc", "hbv")]
    out = preprocess_images(scans)
    per_mod = out[0]  # (3, 20, 256, 256)
    # z-score normalization: near-zero mean and near-unit std (order of magnitude)
    for m in range(per_mod.shape[0]):
        slab = per_mod[m]
        assert abs(float(slab.mean())) < 0.5
        assert 0.05 < float(slab.std()) < 2.0
    assert not np.isnan(out).any()


def test_tensor_conversion(device, official_case):
    scans = [official_case.images[m] for m in ("t2w", "adc", "hbv")]
    out = preprocess_images(scans)
    t = torch.from_numpy(out).to(device)
    assert t.shape == (1, 3, 20, 256, 256)
    assert t.device.type == device
