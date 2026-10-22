"""Simple evaluation against ground truth — PIPELINE VERIFICATION ONLY.

IMPORTANT
=========
The demo runs on a handful of cases. Any number produced here verifies that
the pipeline behaves as expected; it is NOT a clinical evaluation of the
model. The published performance of this model is (PI-CAI hidden test set,
Saha et al., Lancet Oncology 2024): AUROC 0.878 (95% CI 0.855-0.900),
AP 0.558, sensitivity 70.9% at PPV 35.5% (operating point threshold 0.50).
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from .config import CASE_LEVEL_THRESHOLD


def case_level_metrics(cases_and_preds: list[tuple[dict, float]]) -> dict:
    """Accuracy / sensitivity / specificity using the PI-CAI operating point.

    cases_and_preds: list of (ground_truth_dict, predicted_likelihood)
    ground_truth_dict["case_level_csPCa"] is the boolean reference label.
    """
    y_true = []
    y_score = []
    for gt, score in cases_and_preds:
        y_true.append(int(bool(gt["case_level_csPCa"])))
        y_score.append(float(score))
    y_pred_thresholded = [s >= CASE_LEVEL_THRESHOLD for s in y_score]

    tp = sum(t and p for t, p in zip(y_true, y_pred_thresholded))
    fn = sum(t and not p for t, p in zip(y_true, y_pred_thresholded))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred_thresholded))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred_thresholded))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    accuracy = (tp + tn) / len(y_true) if y_true else None

    roc_auc = None
    if len(set(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            roc_auc = None

    return {
        "n_cases": len(y_true),
        "n_positive": int(sum(y_true)),
        "threshold": CASE_LEVEL_THRESHOLD,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "y_true": y_true,
        "y_score": y_score,
    }


def overlap_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """3D IoU between the binarized detection map and the expert annotation."""
    p = pred_mask > 0
    g = gt_mask > 0
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    return inter / union if union > 0 else 0.0


def lesion_localization_case(result_det_map: np.ndarray, gt_annotation) -> dict:
    """Did the model's detection overlap the expert-annotated lesion(s)?

    PI-CAI counts a predicted lesion as a true positive when it overlaps the
    expert annotation with at least 10% IoU in 3D. We report the IoU of the
    whole detection map versus the whole expert annotation for this case.
    """
    gt_arr = sitk.GetArrayFromImage(gt_annotation)
    return {
        "iou_detection_vs_gt": overlap_iou(result_det_map, gt_arr),
        # expert 'resampled' annotations carry labels 0 (ISUP<=1) or 2..5 (ISUP
        # grade); the Pooch25 annotations are binary 0/1 — both are_gt>0
        "gt_has_lesion": gt_arr.max() > 0,
        "max_gt_label": int(gt_arr.max()),  # 0=neg; 1=binary ISUP>=2; 2..5 = ISUP grade
    }
