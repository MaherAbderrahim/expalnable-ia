"""Final evaluation: freeze the selected post-processing config, score VAL+TEST
once, produce the complete metric reports + comparison table + statistics.

Usage:
    python scripts/eval_final.py

Conventions (pre-registered):
    - 'official_default'  = the Phase-1/PI-CAI official post-processing (threshold 0.5)
    - '<selected>'        = config chosen on VALIDATION (PR-AUC rule) + val-F1 threshold
    - the TEST split is scored exactly once, after the freeze.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation import statistics as S  # noqa: E402
from evaluation.eval_model import (  # noqa: E402
    evaluate_config,
    load_split_cases,
    score_dataframe,
)
from evaluation.metrics import bootstrap_ci, roc_auc, pr_auc  # noqa: E402
from evaluation.paths import EXPERIMENTS_CONFIGS, EXPERIMENTS_RESULTS  # noqa: E402
from evaluation.postproc import OFFICIAL_CONFIG, VARIANT_CONFIGS  # noqa: E402


def load_frozen_config() -> dict:
    path = EXPERIMENTS_CONFIGS / "selected_postproc.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/select_config.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def run(val_split: str = "val", test_split: str = "test", verbose: bool = True,
        skip_fold_variance: bool = False, n_workers: int = 4) -> dict:
    frozen = load_frozen_config()
    configs_by_name = {c["name"]: c for c in VARIANT_CONFIGS}
    selected_name = frozen["selected_config"]
    thr_selected = float(frozen["threshold_f1_val"])

    eval_configs = [
        OFFICIAL_CONFIG,           # baseline (threshold 0.50 official)
        configs_by_name[selected_name],  # selected post-processing
    ]
    results = {"frozen": frozen, "splits": {}}
    table_rows = []
    for split in (val_split, test_split):
        df = load_split_cases(split)
        t0 = time.time()
        scores = score_dataframe(df, eval_configs, split=split, verbose=verbose,
                                 n_workers=n_workers)
        scores_dir = EXPERIMENTS_RESULTS / "final_evaluation"
        scores_dir.mkdir(parents=True, exist_ok=True)
        scores.to_csv(scores_dir / f"{split}_scores_long.csv", index=False)
        print(f"[eval] {split}: {len(scores)} rows scored in {time.time()-t0:.0f}s")

        metrics_this = {}
        for cfg, thr in (
            (OFFICIAL_CONFIG, 0.5),
            (configs_by_name[selected_name], thr_selected),
        ):
            rep = evaluate_config(scores, cfg["name"], threshold=thr)
            metrics_this[cfg["name"]] = rep
            op = rep["operating_point"]
            table_rows.append({
                "model": "PI-CAI official 5-fold UNet (pretrained)",
                "config": cfg["name"],
                "dataset": split,
                "threshold": thr,
                "auc_roc": rep["auc_roc"],
                "pr_auc": rep["pr_auc"],
                "accuracy_at_threshold": op["accuracy"],
                "sensitivity": op["sensitivity"],
                "specificity": op["specificity"],
                "precision": op["precision"],
                "f1": op["f1"],
                "balanced_accuracy": op["balanced_accuracy"],
                "ece_10bins": rep["ece_10bins"],
                "n_cases": rep["n_cases"],
                "n_positive": rep["n_positive"],
                "evaluated_on": "validation" if split == val_split else "test(frozen)",
            })
        results["splits"][split] = metrics_this

    # ---- statistics: bootstrap CIs + DeLong + official-vs-selected --------
    stats = {}
    for split in (val_split, test_split):
        df = pd.read_csv(EXPERIMENTS_RESULTS / "final_evaluation" / f"{split}_scores_long.csv")
        off_df = df[df.config == OFFICIAL_CONFIG["name"]]
        sel_df = df[df.config == selected_name]
        y = off_df["y_true"].astype(int).tolist()
        off = off_df["likelihood"].astype(float).tolist()
        sel = sel_df["likelihood"].astype(float).tolist()
        stats[split] = {
            "auc_roc_official_ci": bootstrap_ci(roc_auc, y, off, n_boot=2000, seed=2026),
            "pr_auc_official_ci": bootstrap_ci(pr_auc, y, off, n_boot=2000, seed=2026),
            "auc_roc_selected_ci": bootstrap_ci(roc_auc, y, sel, n_boot=2000, seed=2026),
            "pr_auc_selected_ci": bootstrap_ci(pr_auc, y, sel, n_boot=2000, seed=2026),
            "delong_official_vs_selected": S.delong_test(y, off, sel),
            "bootstrap_auc_diff_official_vs_selected": S.bootstrap_auc_difference(y, off, sel),
        }
    if not skip_fold_variance:
        stats["fold_run_variance_val"] = fold_run_variance(val_split)
    results["statistics"] = stats

    # ---- comparison table (markdown) --------------------------------------
    tab = pd.DataFrame(table_rows)
    out_dir = EXPERIMENTS_RESULTS / "final_evaluation"
    tab.to_csv(out_dir / "comparison_table.csv", index=False)
    md_lines = ["| " + " | ".join(tab.columns) + " |",
                "|" + "|".join(["---"] * len(tab.columns)) + "|"]
    for _, r in tab.iterrows():
        fmt = []
        for c in tab.columns:
            v = r[c]
            if isinstance(v, float):
                fmt.append(f"{v:.4f}" if v is not None and not pd.isna(v) else "—")
            else:
                fmt.append(str(v))
        md_lines.append("| " + " | ".join(fmt) + " |")
    (out_dir / "comparison_table.md").write_text("\n".join(md_lines), encoding="utf-8")

    with open(out_dir / "eval_final_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # ---- error analysis (selected config at its frozen threshold) ---------
    from evaluation.error_analysis import build_error_analysis, summarize_patterns
    for split in (val_split, test_split):
        ea = build_error_analysis(split, selected_name, thr_selected)
        patterns = summarize_patterns(ea, split, selected_name)
        (EXPERIMENTS_RESULTS / "error_analysis").mkdir(parents=True, exist_ok=True)
        (EXPERIMENTS_RESULTS / "error_analysis" / f"{split}_patterns.json").write_text(
            json.dumps(patterns, indent=2, default=str), encoding="utf-8"
        )
        results.setdefault("error_analysis", {})[split] = {
            "counts": ea["error_type"].value_counts().to_dict(),
            "patterns": patterns,
        }
    return results


def fold_run_variance(split: str = "val") -> dict:
    """mean ± sd of AUROC/AP across the 5 single-fold ensembles (stability proxy).

    Uses the single-fold configs (fold0..fold4) — computed directly from saved
    npz via score_dataframe; light because only 5 configs × N cases.
    """
    single_folds = [
        {**OFFICIAL_CONFIG, "name": f"fold{i}", "folds": [i]}
        for i in range(5)
    ]
    df = load_split_cases(split)
    scores = score_dataframe(df, single_folds, split=split, verbose=False,
                             n_workers=4)
    aucs, aps = [], []
    for cfg in single_folds:
        rep = evaluate_config(scores, cfg["name"], threshold=0.5)
        aucs.append(rep["auc_roc"])
        aps.append(rep["pr_auc"])
    return {"auc_roc": S.across_runs_variance(aucs), "pr_auc": S.across_runs_variance(aps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fold-variance", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    res = run(skip_fold_variance=args.no_fold_variance, n_workers=args.workers)
    print(json.dumps({"selected": res["frozen"]["selected_config"],
                      "splits": list(res["splits"].keys())}, indent=2))


if __name__ == "__main__":
    main()
