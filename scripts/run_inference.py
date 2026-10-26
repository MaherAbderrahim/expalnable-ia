"""Run inference (one pass) for the val/test splits — saves per-case components.

Usage:
    python scripts/run_inference.py [--split val] [--split test] [--folds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.infer import run_inference_pass  # noqa: E402
from evaluation.paths import PREDICTIONS_DIR  # noqa: E402
from evaluation.eval_model import load_split_cases  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="append", default=None,
                    choices=["val", "test", "train"])
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="subset of folds to load (default: all 5)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of cases (smoke tests)")
    args = ap.parse_args()

    splits = args.split or ["val", "test"]
    summary = {}
    for split in splits:
        df = load_split_cases(split)
        all_ids = list(df["case_id"].astype(str))
        if args.limit is not None:
            case_ids = [c for c in all_ids if case_images_ready(c)][: args.limit]
        else:
            case_ids = all_ids
        done = {p.name[: -len(".npz")] for p in (PREDICTIONS_DIR / split).glob("*.npz")} \
            if (PREDICTIONS_DIR / split).exists() else set()
        todo = [c for c in case_ids if c not in done and case_images_ready(c)]
        print(f"[run_inference] split={split}: {len(todo)} case(s) to run "
              f"({len(done)} already done of {len(case_ids)})", flush=True)
        if not todo:
            continue
        summary[split] = run_inference_pass(split=split, case_ids=todo, folds=args.folds)
        print(json.dumps(summary[split], indent=2), flush=True)

    (Path("experiments") / "logs" / "run_inference_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def case_images_ready(case_id: str) -> bool:
    from evaluation.infer import case_image_paths
    return all(p.exists() for p in case_image_paths(case_id).values())


if __name__ == "__main__":
    main()
