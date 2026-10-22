"""Test model loading (architecture + checkpoint compatibility)."""

from __future__ import annotations

import torch

from src.inference import UNET_CHANNELS, UNET_STRIDES, load_models, predict
from src.preprocessing import preprocess_images, make_tensor
from src.unet_arch import UNet


def test_model_instantiation(device):
    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        strides=UNET_STRIDES,
        channels=UNET_CHANNELS,
    )
    assert model is not None
    n_params = sum(p.numel() for p in model.parameters())
    # official baseline U-Net has ~32.5M parameters per fold
    assert 10e6 < n_params < 100e6


def test_checkpoint_loads(device, loaded_models):
    models, folds = loaded_models
    assert len(models) == 5
    assert folds == [0, 1, 2, 3, 4]
    for m in models:
        # strict state dict load already happened inside load_models;
        # additionally verify the model moved to the right device
        assert next(m.parameters()).device.type == device


def test_forward_output_shape(device, official_case):
    scans = [official_case.images[m] for m in ("t2w", "adc", "hbv")]
    tensor = make_tensor(preprocess_images(scans), device)
    model = UNet(
        spatial_dims=3, in_channels=3, out_channels=2,
        strides=UNET_STRIDES, channels=UNET_CHANNELS,
    ).to(device)
    models, _ = load_models(device=device, folds=[0])
    with torch.no_grad():
        out = models[0](tensor)
    assert out.shape == (1, 2, 20, 256, 256)


def test_ensemble_prediction_shape_and_range(device, official_case, loaded_models):
    scans = [official_case.images[m] for m in ("t2w", "adc", "hbv")]
    tensor = make_tensor(preprocess_images(scans), device)
    models, _ = loaded_models
    heat = predict(tensor, models, device)
    assert heat.shape == (20, 256, 256)
    assert float(heat.min()) >= 0.0 and float(heat.max()) <= 1.0
