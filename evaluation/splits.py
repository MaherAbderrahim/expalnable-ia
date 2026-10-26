"""Patient-level splits of the selected PI-CAI subset — leakage-safe by design.

RULES (non-negotiable)
----------------------
1. Split by *patient_id*: all case_ids of one patient stay in the same split.
2. Splits are stratified by case-level csPCa status (and center when possible).
3. Deterministic given the seed (documented in the JSON).
4. `check_no_leakage()` recomputes disjointness automatically and raises on
   any overlap — called by the automatic verification test.

We create train/val/test (default 60/20/20) so that, when heavy training is
enabled later, the split files remain identical (frozen test = test until the
end). Phase 2 (no training) only consumes val (for threshold/calibration/
post-processing selection) and test (final evaluation, used once).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.paths import SPLITS_DIR, MARKSHEET_PATH


def load_selection_manifest(seed: int = 2026, selection_path: Path | None = None) -> dict:
    path = selection_path or (SPLITS_DIR / f"case_selection_seed{seed}.json")
    if not path.exists():
        raise FileNotFoundError(
            f"selection manifest not found: {path}. Run scripts/prepare_data.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def make_patient_split(seed: int = 2026, test_frac: float = 0.20, val_frac: float = 0.20,
                       selection_path: Path | None = None,
                       split_name: str | None = None) -> dict:
    """Stratified patient-level split of the SELECTION (not the whole enrollment)."""
    manifest = load_selection_manifest(seed, selection_path)
    case_ids = manifest["case_ids"]
    marksheet = pd.read_csv(MARKSHEET_PATH)
    marksheet["case_id"] = (
        marksheet["patient_id"].astype(str) + "_" + marksheet["study_id"].astype(str)
    )
    sub = marksheet[marksheet["case_id"].isin(case_ids)][
        ["case_id", "patient_id", "study_id", "center", "case_csPCa"]
    ].copy()
    # patient-level records: label = max csPCa over the patient's cases, center
    pat = (
        sub.groupby("patient_id")
        .agg(
            center=("center", "first"),
            positive=("case_csPCa", lambda s: int((s.astype(str) == "YES").any())),
            n_cases=("case_id", "count"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    n_test = int(round(len(pat) * test_frac))
    n_val = int(round(len(pat) * val_frac))

    # stratified allocation: shuffled patients within stratum (center, positive)
    pat["stratum"] = pat["center"].astype(str) + "_" + pat["positive"].astype(str)
    test_pat: list[str] = []
    val_pat: list[str] = []
    for stratum, group in pat.groupby("stratum"):
        g = list(group["patient_id"])
        rng.shuffle(g)
        # proportional allocation of val/test inside each stratum
        k_test = int(round(len(g) / len(pat) * n_test))
        k_val = int(round(len(g) / len(pat) * n_val))
        take_test = g[:k_test]
        take_val = g[k_test: k_test + k_val]
        test_pat += take_test
        val_pat += take_val
    # fix rounding by topping up from remaining train pool
    remaining = [p for p in pat["patient_id"] if p not in test_pat and p not in val_pat]
    while len(test_pat) < n_test and remaining:
        test_pat.append(remaining.pop(rng.integers(len(remaining))))
    while len(val_pat) < n_val and remaining:
        val_pat.append(remaining.pop(rng.integers(len(remaining))))

    split = {
        "seed": int(seed),
        "created": pd.Timestamp.now().isoformat(),
        "fractions": {"test": test_frac, "val": val_frac, "train": 1 - test_frac - val_frac},
        "rule": (
            "split by patient_id; no patient appears in two splits; stratified "
            "by (center, csPCa status); test and val used for evaluation "
            "(val = selection/threshold/calibration, test = final evaluation "
            "used once); train kept for future training runs"
        ),
        "splits": {
            "train_patients": sorted(str(p) for p in remaining),
            "val_patients": sorted(str(p) for p in val_pat),
            "test_patients": sorted(str(p) for p in test_pat),
        },
        "stratum Representation": {k: int(v) for k, v in pat["stratum"].value_counts().items()},
    }
    # derive case lists (patient -> its cases; marksheet patient ids are ints)
    case_ids_by_patient = sub.groupby("patient_id")["case_id"].apply(list).to_dict()
    case_ids_by_patient = {str(k): v for k, v in case_ids_by_patient.items()}
    for name in ["train", "val", "test"]:
        cids = sorted(
            cid for p in split["splits"][f"{name}_patients"]
            for cid in case_ids_by_patient.get(str(p), [])
        )
        split["splits"][name] = cids
        split["splits"][f"n_{name}_cases"] = len(cids)
        split["splits"][f"n_{name}_patients"] = len(split["splits"][f"{name}_patients"])
    del split["stratum Representation"]
    return split


def save_split(split: dict, split_name: str = "patient_split") -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(
        json.dumps(split["splits"]["test"], sort_keys=True).encode()
    ).hexdigest()[:10]
    out = SPLITS_DIR / f"{split_name}_{split['seed']}_{h}.json"
    out.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return out


def check_no_leakage(split: dict) -> dict:
    """Assert patient- and case-level disjointness; raise on violation."""
    s = split["splits"]
    for name in ["train", "val", "test"]:
        assert s.get(name), f"empty split: {name}"
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        pa = set(s[f"{a}_patients"])
        pb = set(s[f"{b}_patients"])
        overlap = pa & pb
        if overlap:
            raise AssertionError(
                f"DATA LEAKAGE: patients shared between '{a}' and '{b}': {sorted(overlap)[:5]} ..."
            )
        ca = set(s[a])
        cb = set(s[b])
        oc = ca & cb
        if oc:
            raise AssertionError(
                f"DATA LEAKAGE: cases shared between '{a}' and '{b}': {sorted(oc)[:5]} ..."
            )
    return {
        "ok": True,
        "n_patients": {
            "train": len(s["train_patients"]),
            "val": len(s["val_patients"]),
            "test": len(s["test_patients"]),
        },
        "n_cases": {"train": len(s["train"]), "val": len(s["val"]), "test": len(s["test"])},
    }


def split_case_label(case_ids: list[str]) -> list[int]:
    """case-level ground-truth labels for a list of case_ids (from the marksheet)."""
    marksheet = pd.read_csv(MARKSHEET_PATH)
    marksheet["case_id"] = (
        marksheet["patient_id"].astype(str) + "_" + marksheet["study_id"].astype(str)
    )
    lut = dict(zip(marksheet["case_id"], (marksheet["case_csPCa"].astype(str) == "YES").astype(int)))
    return [lut[c] for c in case_ids]
