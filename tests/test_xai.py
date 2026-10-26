"""XAI unit tests on a tiny synthetic MONAI UNet (same structural contract as
the PI-CAI ensemble: model.model[-1][1] at the last skip connection).

No reference weights are loaded — every test here validates SHAPES,
NUMERICAL BEHAVIOR and the reliability machinery, deterministically.
"""

import numpy as np
import torch
import pytest

torch.manual_seed(2026)


def _tiny_models(n: int = 2, device: str = "cpu"):
    import sys
    from evaluation.infer import DEMO_DIR
    if str(DEMO_DIR) not in sys.path:
        sys.path.insert(0, str(DEMO_DIR))
    from src.unet_arch import UNet
    models = []
    for _ in range(n):
        m = UNet(spatial_dims=3, in_channels=3, out_channels=2,
                 strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
                 channels=[2, 4, 8, 16, 32, 64])
        m = m.to(device).eval()
        models.append(m)
    return models


def _blob_input(device: str = "cpu") -> torch.Tensor:
    """(1, 3, 20, 256, 256): one Gaussian blob in the T2W channel."""
    z, y, x = np.mgrid[0:20, 0:256, 0:256]
    r2 = (z - 10) ** 2 + (y - 128) ** 2 + (x - 128) ** 2
    vol = 3.0 * np.exp(-r2 / (2 * 25.0 ** 2)) + np.random.default_rng(0).normal(0, 0.05, r2.shape)
    arr = np.zeros((1, 3, 20, 256, 256), dtype=np.float32)
    arr[0, 0] = vol
    return torch.from_numpy(arr).to(device)


def test_gradcam_shape_and_range():
    from xai.methods import gradcam_3d
    m = _tiny_models(1)[0]
    x = _blob_input()
    cam = gradcam_3d(m, x, method="gradcam")
    assert cam.shape == (20, 256, 256)
    assert np.isfinite(cam).all()
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6
    assert cam.max() > 0.0


def test_gradcampp_differs_from_gradcam():
    from xai.methods import gradcam_3d
    m = _tiny_models(1)[0]
    x = _blob_input()
    cam1 = gradcam_3d(m, x, method="gradcam")
    cam2 = gradcam_3d(m, x, method="gradcampp")
    assert cam2.shape == cam1.shape
    assert np.isfinite(cam2).all()
    assert not np.allclose(cam1, cam2)


def test_ensemble_gradcam_averages_folds():
    from xai.methods import ensemble_gradcam
    models = _tiny_models(2)
    x = _blob_input()
    cam = ensemble_gradcam(models, x, "cpu", method="gradcam")
    assert cam.shape == (20, 256, 256)
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6


def test_integrated_gradients_shape_and_range():
    from xai.methods import ensemble_integrated_gradients
    models = _tiny_models(1)
    x = _blob_input()
    ig = ensemble_integrated_gradients(models, x, "cpu", steps=4)
    assert ig.shape == (20, 256, 256)
    assert np.isfinite(ig).all()
    assert 0.0 <= ig.min() and ig.max() <= 1.0 + 1e-6


def test_occlusion_scan_shape():
    from xai.methods import occlusion_scan
    models = _tiny_models(1)
    x = _blob_input()
    heat = occlusion_scan(models, x, "cpu", slab=(2, 16, 16), stride=(4, 64, 64))
    assert heat.shape == (20, 256, 256)
    assert np.isfinite(heat).all()
    assert heat.max() <= 1.0


def test_fused_score_from_batch_per_item():
    from xai.methods import fused_score_from_batch
    models = _tiny_models(1)
    x = _blob_input()
    s = fused_score_from_batch(models, x, "cpu")
    assert s.shape == (1,)
    assert 0.0 <= s[0] <= 1.0


def test_localization_agreement_aligned_vs_offset():
    from xai.quality import localization_agreement
    gt = np.zeros((10, 20, 20), dtype=bool)
    gt[4:7, 8:12, 8:12] = True
    hot = np.zeros((10, 20, 20), dtype=np.float32)
    hot[4:7, 8:12, 8:12] = 1.0
    aligned = localization_agreement(hot, gt, top_frac=0.01)
    shifted = localization_agreement(np.roll(hot, 4, axis=2), gt, top_frac=0.01)
    assert aligned["iou_top_vs_lesion"] > shifted["iou_top_vs_lesion"]
    assert aligned["iou_top_vs_lesion"] > 0.5
    assert aligned["enrichment"] > 1.0
    assert 0.0 <= shifted["iou_top_vs_lesion"] < 0.1


def test_randomization_changes_cam():
    from xai.quality import randomization_test
    models = _tiny_models(1)
    x = _blob_input().numpy()
    res = randomization_test(models, x, "cpu", "gradcam", seed=2026)
    assert np.isfinite(res["pearson_similarity_orig_vs_randomized"])
    assert res["pearson_similarity_orig_vs_randomized"] < 0.95


def test_stability_jitter_correlates():
    from xai.quality import stability_jitter
    models = _tiny_models(1)
    x = _blob_input().numpy()
    res = stability_jitter(models, x, "cpu", "gradcam")
    assert np.isfinite(res["pearson_shift_y"])
    assert res["pearson_shift_y"] < 1.0 + 1e-6


def test_insertion_deletion_structure():
    from xai.quality import insertion_deletion
    models = _tiny_models(1)
    x = _blob_input().numpy()
    cam = np.zeros((20, 256, 256), dtype=np.float32)
    z, y, xx = np.mgrid[0:20, 0:256, 0:256]
    cam = np.exp(-((z - 10) ** 2 + (y - 128) ** 2 + (xx - 128) ** 2) / (2 * 30.0 ** 2))
    res = insertion_deletion(models, x, cam, "cpu", steps=4)
    assert len(res["insertion_scores"]) == 5
    assert len(res["deletion_scores"]) == 5
    assert all(np.isfinite(s) and 0.0 <= s <= 1.0 for s in res["insertion_scores"])
    assert all(np.isfinite(s) and 0.0 <= s <= 1.0 for s in res["deletion_scores"])


def test_explain_figure_smoke(tmp_path):
    from xai.visualize import explain_figure
    import argparse
    t2w = np.random.default_rng(0).normal(0.4, 0.1, (20, 256, 256)).astype(np.float32)
    det = np.clip(t2w, 0, 1)
    hms = {"gradcam": np.clip(t2w, 0, 1).astype(np.float32)}
    p = explain_figure(t2w, det, hms,
                       {"case_id": "synthetic", "likelihood": 0.42,
                        "threshold": 0.5, "y_true": 1},
                       save_path=tmp_path / "fig.png")
    assert p is not None and p.exists() and p.stat().st_size > 0
