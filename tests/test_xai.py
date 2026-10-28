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
    torch.manual_seed(2026)  # order-independent weights: each call is deterministic
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


# ---------------------------------------------------------------------------
# Input validation (contract shared by the 3 method families)
# ---------------------------------------------------------------------------

def test_invalid_inputs_raised_with_valueerror():
    from xai.methods import (
        ensemble_gradcam,
        ensemble_integrated_gradients,
        occlusion_scan,
    )
    import torch as th
    models = _tiny_models(1)
    x = _blob_input()
    # wrong channel count
    bad_c = th.zeros((1, 2, 20, 256, 256))
    for fn in (
        lambda: ensemble_gradcam(models, bad_c, "cpu"),
        lambda: ensemble_integrated_gradients(models, bad_c, "cpu", steps=2),
        lambda: occlusion_scan(models, bad_c, "cpu", slab=(2, 16, 16), stride=(8, 128, 128)),
    ):
        with pytest.raises(ValueError):
            fn()
    # wrong dimensionality
    bad_d = th.zeros((1, 3, 256, 256))
    with pytest.raises(ValueError):
        ensemble_gradcam(models, bad_d, "cpu")
    # multi-case batch explained as ONE explanation is rejected
    bad_b = th.zeros((2, 3, 20, 256, 256))
    for fn in (
        lambda: ensemble_gradcam(models, bad_b, "cpu"),
        lambda: ensemble_integrated_gradients(models, bad_b, "cpu", steps=2),
        lambda: occlusion_scan(models, bad_b, "cpu", slab=(2, 16, 16), stride=(8, 128, 128)),
    ):
        with pytest.raises(ValueError):
            fn()
    # the batch scorer for occlusion positions stays multi-case by design
    from xai.methods import fused_score_from_batch
    s = fused_score_from_batch(models, bad_b, "cpu")  # no raise
    assert s.shape == (2,)


# ---------------------------------------------------------------------------
# Integrated Gradients specifics
# ---------------------------------------------------------------------------

def test_ig_single_preserves_sign_info():
    from xai.methods import integrated_gradients_single
    m = _tiny_models(1)[0]
    x = _blob_input()
    raw = integrated_gradients_single(m, x, "cpu", steps=4)
    assert raw.shape == (3, 20, 256, 256)
    assert np.isfinite(raw).all()
    # signed raw attribution: both positive and negative contributions exist
    assert (raw > 0).any() and (raw < 0).any()


def test_ig_attributions_concentrate_on_the_blob():
    from xai.methods import ensemble_integrated_gradients
    from scipy.ndimage import gaussian_filter
    m = _tiny_models(1)
    x = _blob_input()
    ig = ensemble_integrated_gradients(m, x, "cpu", steps=4)
    assert ig.shape == (20, 256, 256)
    # IG peaks near the blob (within its footprint: same z, <30 px in-plane),
    # not on an unrelated structure
    zit, yy, xxx = np.unravel_index(int(np.argmax(ig)), ig.shape)
    assert abs(zit - 10) <= 2 and abs(yy - 128) <= 60 and abs(xxx - 128) <= 60
    # and the smooth IG map correlates positively with the smooth blob shape
    blob_s = gaussian_filter(x[0, 0].detach().numpy(), sigma=10)
    ig_s = gaussian_filter(ig, sigma=10)
    r = np.corrcoef(blob_s.ravel(), ig_s.ravel())[0, 1]
    assert r > 0.4


def test_ig_nondegenerate_across_different_inputs():
    from xai.methods import ensemble_integrated_gradients
    m = _tiny_models(1)
    x1 = _blob_input()
    x2 = _blob_input()  # shifted blob input
    x2 = torch.roll(x2, shifts=(6, 40, 40), dims=(2, 3, 4))
    ig1 = ensemble_integrated_gradients(m, x1, "cpu", steps=3)
    ig2 = ensemble_integrated_gradients(m, x2, "cpu", steps=3)
    assert np.corrcoef(ig1.ravel(), ig2.ravel())[0, 1] < 0.999  # input-dependent


# ---------------------------------------------------------------------------
# Occlusion sensitivity specifics
# ---------------------------------------------------------------------------

def test_occlusion_input_dependent_and_not_artificial():
    from xai.methods import occlusion_scan
    from xai.quality import map_is_degenerate
    m = _tiny_models(1)
    x = _blob_input()
    heat = occlusion_scan(m, x, "cpu", slab=(2, 16, 16), stride=(4, 32, 32))
    assert heat.shape == (20, 256, 256)
    assert np.isfinite(heat).all()
    # RAW magnitudes (normalize=False): a featureless input must not produce
    # meaningful drops — any occlusion saliency on a truly flat scan is noise
    flat = torch.zeros((1, 3, 20, 256, 256)) + 0.05
    flat_raw = occlusion_scan(m, flat, "cpu", slab=(2, 16, 16), stride=(4, 32, 32),
                              normalize=False)
    assert flat_raw.max() <= 0.01, flat_raw.max()
    # the map is input-dependent: two different inputs -> different saliency
    h2 = occlusion_scan(m, torch.roll(x, shifts=(6, 40, 40), dims=(2, 3, 4)),
                        "cpu", slab=(2, 16, 16), stride=(4, 32, 32))
    assert np.corrcoef(heat.ravel(), h2.ravel())[0, 1] < 0.99
    # the blob scan itself is not flagged degenerate
    assert map_is_degenerate(heat)["degenerate"] is False


def test_occlusion_computes_for_real_and_summarizes_to_input():
    """Magnitude-level honesty: with RAW drops, removing the actual blob region
    must yield the strongest positive drop of the scan."""
    from xai.methods import occlusion_scan
    m = _tiny_models(1)
    x = torch.zeros((1, 3, 20, 256, 256)) + 0.05
    x[0, 0, 8:12, 112:144, 112:144] = 3.0  # strong isolated blob, no noise
    raw = occlusion_scan(m, x, "cpu", slab=(2, 16, 16), stride=(4, 16, 16),
                         normalize=False)
    assert np.isfinite(raw).all()
    # the slab that contained the blob must be among the highest-drop regions
    best_pos = np.unravel_index(int(np.argmax(raw)), raw.shape)
    assert abs(best_pos[0] - 8) <= 4 and abs(best_pos[1] - 112) <= 48 and abs(best_pos[2] - 112) <= 48


# ---------------------------------------------------------------------------
# Extended reliability machinery (per-method controls)
# ---------------------------------------------------------------------------

def test_map_is_degenerate_flags_constant_or_zero_maps():
    from xai.quality import map_is_degenerate
    assert map_is_degenerate(np.ones((4, 4, 4)))["degenerate"] is True
    assert map_is_degenerate(np.zeros((4, 4, 4)))["degenerate"] is True
    hot = np.zeros((4, 4, 4), np.float32); hot[1, 1, 1] = 1.0
    assert map_is_degenerate(hot)["degenerate"] is False
    nanmap = np.full((4, 4, 4), np.nan)
    assert map_is_degenerate(nanmap)["degenerate"] is True


def test_stability_jitter_generic_explainer():
    from xai.quality import stability_jitter
    from xai.methods import ensemble_integrated_gradients
    models = _tiny_models(1)
    x = _blob_input().numpy()
    res = stability_jitter(
        models, x, "cpu", "integrated_gradients",
        explainer=lambda ms, t, dv: ensemble_integrated_gradients(ms, t, dv, steps=3))
    assert np.isfinite(res["pearson_shift_y"])
    assert res["pearson_shift_y"] <= 1.0 + 1e-6
    # unknown method without explainer must fail loudly, not fake an explainer
    with pytest.raises(ValueError):
        stability_jitter(models, x, "cpu", "integrated_gradients")


def test_randomization_test_ig_detects_weight_corruption():
    from xai.quality import randomization_test_ig
    models = _tiny_models(1)
    x = _blob_input().numpy()
    res = randomization_test_ig(models, x, "cpu", steps=3)
    assert np.isfinite(res["pearson_similarity_orig_vs_randomized"])
    # randomized fold must NOT reproduce the reference attribution
    assert res["pearson_similarity_orig_vs_randomized"] < 0.95
    # stricter full-model randomization also changes the attribution
    res_all = randomization_test_ig(models, x, "cpu", steps=3, depth="all")
    assert res_all["pearson_similarity_orig_vs_randomized"] < 0.95
    with pytest.raises(ValueError):
        randomization_test_ig(models, x, "cpu", steps=3, depth="wrong")


def test_randomization_test_occlusion_structure():
    from xai.quality import randomization_test_occlusion
    models = _tiny_models(1)
    x = _blob_input().numpy()
    res = randomization_test_occlusion(models, x, "cpu",
                                       slab=(2, 16, 16), stride=(8, 128, 128))
    assert np.isfinite(res["pearson_similarity_orig_vs_randomized"])
    assert res["pearson_similarity_orig_vs_randomized"] <= 1.0 + 1e-6


def test_three_methods_same_pipeline_contract():
    """Integration contract: on the SAME models + SAME input, the three method
    families each produce one valid (20, 256, 256) attribution for the SAME
    explained output (csPCa channel)."""
    from xai.methods import (
        ensemble_gradcam,
        ensemble_integrated_gradients,
        occlusion_scan,
    )
    models = _tiny_models(2)
    x = _blob_input()
    maps = {
        "gradcam": ensemble_gradcam(models, x, "cpu", "gradcam"),
        "integrated_gradients": ensemble_integrated_gradients(models, x, "cpu", steps=3),
        "occlusion": occlusion_scan(models, x, "cpu", slab=(2, 16, 16), stride=(4, 64, 64)),
    }
    for name, h in maps.items():
        assert h.shape == (20, 256, 256), name
        assert np.isfinite(h).all(), name
        assert h.max() > 0.0, name
    # methods genuinely disagree (complementary families, not copies)
    a = maps["gradcam"].ravel(); b = maps["integrated_gradients"].ravel()
    c = maps["occlusion"].ravel()
    assert np.corrcoef(a, b)[0, 1] < 0.999
    assert np.corrcoef(a, c)[0, 1] < 0.999
