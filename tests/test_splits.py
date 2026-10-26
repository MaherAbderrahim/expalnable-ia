"""Anti-leakage verification of the patient-level split (hard fail on violation)."""

from __future__ import annotations

import pytest

from evaluation.splits import check_no_leakage, make_patient_split, save_split, split_case_label


def test_split_is_patient_level_and_disjoint(selection_manifest_path):
    split = make_patient_split(seed=2026, test_frac=0.20, val_frac=0.20)
    verdict = check_no_leakage(split)  # raises on any overlap
    assert verdict["ok"] is True

    # patient disjointness explicitly re-asserted three ways
    p_train = set(split["splits"]["train_patients"])
    p_val = set(split["splits"]["val_patients"])
    p_test = set(split["splits"]["test_patients"])
    assert not (p_train & p_val) and not (p_train & p_test) and not (p_val & p_test)

    # case disjointness
    c_train, c_val, c_test = (set(split["splits"][k]) for k in ["train", "val", "test"])
    assert not (c_train & c_val) and not (c_train & c_test) and not (c_val & c_test)

    # every selected case belongs to exactly one split, one patient per split
    selection = pytest.importorskip("json")
    import json
    sel = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    all_cases = [c for c in sel["case_ids"]]
    covered = c_train | c_val | c_test
    assert covered == set(all_cases), "split must cover the whole selection"


def test_split_determinism(selection_manifest_path):
    a = make_patient_split(seed=2026)
    b = make_patient_split(seed=2026)
    assert a["splits"]["test_patients"] == b["splits"]["test_patients"]
    assert a["splits"]["val"] == b["splits"]["val"]


def test_split_labels_stratification_positive_share(selection_manifest_path):
    """pos share of val/test should roughly reflect the selection (~28%)."""
    split = make_patient_split(seed=2026)
    for name in ["val", "test"]:
        labels = split_case_label(split["splits"][name])
        share = sum(labels) / len(labels)
        assert 0.15 < share < 0.45, f"positive share {share:.2f} out of range"
