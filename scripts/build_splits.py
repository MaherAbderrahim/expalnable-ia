"""Build the leakage-safe patient-level split for the selected PI-CAI subset.

Usage:
    python scripts/build_splits.py [--seed 2026] [--test-frac 0.20] [--val-frac 0.20]

Writes data/splits/patient_split_<seed>_<hash>.json and prints the
disjointness verification result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.paths import SPLITS_DIR  # noqa: E402
from evaluation.splits import check_no_leakage, make_patient_split, save_split, split_case_label  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--selection", type=str, default=None,
                    help="path to an alternative selection manifest")
    args = ap.parse_args()

    split = make_patient_split(seed=args.seed, test_frac=args.test_frac,
                               val_frac=args.val_frac,
                               selection_path=Path(args.selection) if args.selection else None)
    verification = check_no_leakage(split)
    out = save_split(split)

    s = split["splits"]
    print(f"split file : {out.name}")
    for name in ["train", "val", "test"]:
        labels = split_case_label(s[name])
        print(f"{name:6s}: {len(s[f'{name}_patients'])} patients / {len(s[name])} cases "
              f"({sum(labels)} csPCa+)")
    print(f"leakage check: {'OK — zero shared patients/cases between splits' if verification['ok'] else 'FAILED'}")
    print(json.dumps(verification["n_cases"], indent=2))


if __name__ == "__main__":
    main()
