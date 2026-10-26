"""Select the best post-processing configuration on the VALIDATION split only.

Pre-registered selection rule (written BEFORE looking at any test number):
  1. Rank configs by validation PR-AUC (average precision), then AUROC tie-break.
  2. For the winning config, optimize the decision threshold on validation by
     maximizing F1 (medical screening favors a calibration-aware, but simple,
     criterion; Youden J is reported for comparison).
  3. Freeze: (config, threshold) -> scored ONCE on the test split.

Also evaluates the OFFICIAL default configuration as the baseline reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_model import evaluate_config, load_split_cases, score_dataframe  # noqa
from evaluation.metrics import best_threshold_f1, best_threshold_youden  # noqa
from evaluation.paths import (
    EXPERIMENTS_CONFIGS,
    EXPERIMENTS_RESULTS,
)  # noqa
from evaluation.postproc import OFFICIAL_CONFIG, VARIANT_CONFIGS  # noqa


def run_selection(val_split: str = "val", verbose: bool = True,
                  n_workers: int = 4) -> dict:
    df = load_split_cases(val_split)
    print(f"validation cases: {len(df)} "
          f"({int((df['case_csPCa'] == 'YES').sum())} csPCa+)", flush=True)

    configs = VARIANT_CONFIGS  # includes OFFICIAL_CONFIG (name official_default)
    scores = score_dataframe(df, configs, split=val_split, verbose=verbose,
                             n_workers=n_workers)
    scores_path = EXPERIMENTS_RESULTS / "postproc_selection" / "val_scores_long.csv"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(scores_path, index=False)

    table = []
    for cfg in configs:
        rep = evaluate_config(scores, cfg["name"], threshold=0.5)
        op = rep["operating_point"]
        table.append({
            "config": cfg["name"],
            "auc_roc": rep["auc_roc"],
            "pr_auc": rep["pr_auc"],
            "f1_at_05": op["f1"],
            "sens_at_05": op["sensitivity"],
            "spec_at_05": op["specificity"],
            "bal_acc_at_05": op["balanced_accuracy"],
            "ece_10bins": rep["ece_10bins"],
        })
    tab = pd.DataFrame(table)
    tab = tab.sort_values(["pr_auc", "auc_roc"], ascending=False).reset_index(drop=True)
    tab_path = (EXPERIMENTS_RESULTS / "postproc_selection" / "val_comparison.csv")
    tab.to_csv(tab_path, index=False)
    if verbose:
        print(tab.to_string(index=False))

    winner = tab.iloc[0]["config"]
    w_scores = scores[scores["config"] == winner]
    thr_f1, f1 = best_threshold_f1(w_scores["y_true"].tolist(), w_scores["likelihood"].tolist())
    thr_yj, ba = best_threshold_youden(w_scores["y_true"].tolist(), w_scores["likelihood"].tolist())

    frozen = {
        "selection_rule": (
            "max val PR-AUC (tie-break AUROC); decision threshold = max F1 on "
            "validation; Youden J reported; test scored once, after freeze"
        ),
        "val_split": val_split,
        "val_scores_csv": str(scores_path.relative_to(REPO_ROOT)),
        "val_comparison_csv": str(tab_path.relative_to(REPO_ROOT)),
        "selected_config": winner,
        "threshold_f1_val": float(thr_f1),
        "f1_val": float(f1),
        "threshold_youden_val": float(thr_yj),
        "balanced_accuracy_youden_val": float(ba),
        "official_threshold_reference": 0.5,
        "table": tab.to_dict(orient="records"),
        "created": pd.Timestamp.now().isoformat(),
    }
    out = EXPERIMENTS_CONFIGS / "selected_postproc.json"
    out.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"\n[selected] config={winner} thr_f1={thr_f1:.4f} "
          f"(Youden: {thr_yj:.4f}) -> {out.name}")
    return frozen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-split", type=str, default="val")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run_selection(args.val_split, n_workers=args.workers)


if __name__ == "__main__":
    main()
