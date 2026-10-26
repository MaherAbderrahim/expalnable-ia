"""Unit tests of the metric suite (operating points, AUCs, calibration, CI)."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation import metrics as M


def _labelled_scores():
    y = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]  # 6 pos / 4 neg
    s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    return y, s


def test_operating_point_counts():
    ok = M.operating_point_metrics([1, 0, 1, 0], [0.9, 0.6, 0.4, 0.2], threshold=0.5)
    assert ok["tp"] == 1 and ok["fn"] == 1 and ok["fp"] == 1 and ok["tn"] == 1
    assert ok["sensitivity"] == 0.5 and ok["specificity"] == 0.5
    assert ok["precision"] == 0.5 and ok["f1"] == 0.5
    assert ok["balanced_accuracy"] == 0.5


def test_roc_pr_auc_known_values():
    y, s = _labelled_scores()
    # hand-computed: 11 of the 24 (pos,neg) pairs rank correctly
    assert M.roc_auc(y, s) == pytest.approx(11 / 24)
    # AP = mean precision at the 6 positive ranks (score-desc order):
    # (1 + 2/3 + 3/4 + 4/7 + 5/9 + 3/5) / 6
    ap_expected = (1 + 2 / 3 + 3 / 4 + 4 / 7 + 5 / 9 + 3 / 5) / 6
    assert M.pr_auc(y, s) == pytest.approx(ap_expected, rel=1e-9)
    # degenerate: single class
    assert M.roc_auc([1, 1], [0.1, 0.2]) is None
    assert M.pr_auc([1, 1], [0.1, 0.2]) is None


def test_threshold_sweep_and_optima():
    y, s = _labelled_scores()
    sweep = M.threshold_sweep(y, s, thresholds=[0.0, 0.35, 1.0])
    assert [r["threshold"] for r in sweep] == [0.0, 0.35, 1.0]
    # threshold 0 -> predict all positive
    assert sweep[0]["sensitivity"] == 1.0 and sweep[0]["specificity"] == 0.0
    # downward-ordered scores: to match every positive with >=, need threshold <= 0.05
    t, sens = M.threshold_at_target_sensitivity(y, s, 1.0)
    assert t == pytest.approx(0.05) and sens == 1.0
    t_f1, f1 = M.best_threshold_f1(y, s)
    assert f1 is not None and 0 <= t_f1 <= 1


def test_calibration_ece_and_platt():
    y, s = _labelled_scores()
    ece, bins = M.expected_calibration_error(y, s, n_bins=10)
    assert 0.0 <= ece <= 1.0
    n_covered = sum(b["n"] for b in bins)
    assert n_covered == len(y)
    a, b = M.fit_platt_scale(y, s)
    calibrated = M.apply_platt(s, a, b)
    # calibration must be monotonic w.r.t. raw score (strictly increasing a>0)
    assert a > 0
    rank = np.argsort(s)
    assert np.all(np.diff(np.asarray(calibrated)[rank]) >= 0)


def test_bootstrap_ci_sanity():
    y_good = [1] * 20 + [0] * 20
    y, s_good = [], []
    for i in range(20):
        y += [1, 0]
        s_good += [0.9 - 0.01 * i, 1.0 - 0.01 * i - 0.6]
    ci = M.bootstrap_ci(M.roc_auc, y, s_good, n_boot=300, seed=1)
    assert ci["value"] == 1.0
    assert ci["ci_lower"] == pytest.approx(1.0, abs=1e-9)
    _, s_bad = [], []
    for i in range(20):
        s_bad += [1.0 - 0.01 * i - 0.6, 0.9 - 0.01 * i]
    lo_bad = M.bootstrap_ci(M.roc_auc, y, s_bad, n_boot=300, seed=1)
    assert lo_bad["ci_upper"] <= 1.0 + 1e-9


def test_full_report_structure():
    y, s = _labelled_scores()
    rep = M.full_case_report(y, s, threshold=0.5)
    for key in ["auc_roc", "pr_auc", "operating_point", "ece_10bins", "sweep"]:
        assert key in rep
    assert rep["operating_point"]["threshold"] == 0.5


def test_mask_overlap_metrics():
    pred = np.zeros((4, 4, 4), dtype=np.float32)
    gt = np.zeros((4, 4, 4), dtype=np.float32)
    pred[0:2, 0:2, 0:2] = 1
    gt[0:2, 0:2, 0:2] = 1
    iou, dice = M.mask_iou_dice(pred, gt)
    assert iou == pytest.approx(1.0) and dice == pytest.approx(1.0)
    # disjoint masks
    gt2 = np.zeros((4, 4, 4), dtype=np.float32)
    gt2[2:4, 2:4, 2:4] = 1
    iou, dice = M.mask_iou_dice(pred, gt2)
    assert iou == 0.0 and dice == 0.0
