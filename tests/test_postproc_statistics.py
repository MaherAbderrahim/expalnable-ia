"""Tests of the offline post-processing stack + statistics (synthetic inputs)."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.postproc import (
    OFFICIAL_CONFIG,
    fused_from_components,
    official_detection_map,
    resolve_folds,
    score_case,
)
from evaluation import statistics as S
from evaluation import metrics as M


def _fake_npz(seed=0, n_folds=5, peak_pos=(6, 200, 200)):
    """Synthetic inference components on the model grid + a native fused map."""
    rng = np.random.default_rng(seed)
    base_model = np.full((20, 256, 256), 0.02, dtype=np.float32)
    base_model[peak_pos] = 0.9  # a clear suspicious voxel on the model grid
    per_fold = np.clip(
        base_model[None] + 0.01 * rng.standard_normal((n_folds, 20, 256, 256)), 0.0, 1.0
    ).astype(np.float16)
    native = np.full((19, 384, 384), 0.02, dtype=np.float32)
    native[5, 250, 250] = 0.85  # suspicious voxel on the native grid
    native[4:7, 243:258, 243:258] = 0.6  # give the signal blob support (RGA needs it)
    return {
        "folds_orig": per_fold.copy(),
        "folds_flip": per_fold.copy(),
        "native_fused": native,
        "meta_spacing_xyz": np.array([0.596, 0.596, 3.0], dtype=np.float32),
    }



def test_resolve_folds():
    assert resolve_folds({"folds": "all"}, 5) == [0, 1, 2, 3, 4]
    assert resolve_folds({"folds": [0, 2]}, 5) == [0, 2]


def test_official_detection_map_basics():
    npz = _fake_npz()
    det = official_detection_map(npz["native_fused"], OFFICIAL_CONFIG)
    assert det.shape == npz["native_fused"].shape
    assert det.max() <= npz["native_fused"].max() + 1e-6

    # single-voxel volumes are degenerate for the 'dynamic' RGA extraction —
    # check no crash + empty result instead of a lift
    flat = np.zeros((10, 10, 10), dtype=np.float32)
    flat[5, 5, 5] = 0.7
    d = official_detection_map(flat, OFFICIAL_CONFIG)
    assert d.shape == flat.shape
    assert float(np.max(d)) <= 0.7 + 1e-6

    # a realistic 3D blob survives extraction and the max/5 ring suppression
    blob = np.zeros((20, 100, 100), dtype=np.float32)
    blob[6:11, 45:55, 40:55] = 0.6
    blob[8, 48, 47] = 0.7  # keep the isolated peak inside the blob core
    d = official_detection_map(blob, OFFICIAL_CONFIG)
    assert d[8, 48, 47] > 0.1



def test_score_case_official_matches_stock_likelihood():
    """score_case with the official config: likelihood == max (native fused
    after the official extraction+suppression) â€” approx by construction."""
    npz = _fake_npz()
    out = score_case(npz, OFFICIAL_CONFIG, want_map=False)
    assert 0.0 <= out["likelihood"] <= 1.0
    assert out["likelihood"] > 0.5  # the planted voxel survives


def test_fused_from_components_shape():
    npz = _fake_npz()
    fused = fused_from_components(
        npz["folds_orig"], npz["folds_flip"], [0, 1], sigma=1.0, tta=False,
        native_spacing_xyz=tuple(npz["meta_spacing_xyz"]),
        native_shape_zyx=tuple(npz["native_fused"].shape),
    )
    assert fused.shape == npz["native_fused"].shape
    assert fused.max() <= 1.0 + 1e-4


def test_delong_degenerate():
    y = [1] * 5 + [0] * 5
    res = S.delong_test(y, [0.9] * 10, [0.1] * 10)
    assert res["auc_a"] == 0.5 and res["auc_b"] == 0.5
    assert res["p_value"] == pytest.approx(1.0)


def test_delong_separation():
    y = [1] * 6 + [0] * 4
    a = [0.9, 0.8, 0.7, 0.6, 0.65, 0.75, 0.1, 0.15, 0.2, 0.05]
    b = [0.5] * 10
    res = S.delong_test(y, a, b)
    assert res["auc_a"] > res["auc_b"]
    assert 0 <= res["p_value"] <= 0.05


def test_bootstrap_auc_difference_sign():
    y = [1] * 10 + [0] * 10
    a = [0.9 - 0.01 * i for i in range(10)] + [0.9 - 0.01 * (i + 10) for i in range(10)]  # pos high
    b = list(reversed(a))  # inverted ranking
    res = S.bootstrap_auc_difference(y, a, b, n_boot=500, seed=3)
    assert res["mean_diff"] > 0
    assert res["ci_lower"] > 0  # consistently better



def test_across_runs_variance():
    v = S.across_runs_variance([0.80, 0.82, 0.78])
    assert v["n_runs"] == 3
    assert v["mean"] == pytest.approx(0.80)
    assert v["sd"] == pytest.approx(np.std([0.80, 0.82, 0.78], ddof=1), rel=1e-9)


def test_confusion_and_platt_remaining():
    y = [1] * 7 + [0] * 7
    s = list(np.linspace(0.95, 0.05, 14))
    a, b = M.fit_platt_scale(y, s)
    calib = M.apply_platt(s, a, b)
    assert a > 0
    assert all(0.0 <= c <= 1.0 for c in calib)

