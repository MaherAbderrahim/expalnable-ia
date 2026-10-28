"""Reproducible XAI comparison report — regenerates reports/XAI_COMPARISON.md.

Reads the per-method quality artifacts produced by scripts/explain.py and adds
two documented supplementary measurements that cost extra GPU passes and are
therefore run here (on the same cases, not inside explain.py):

  1. IG model-randomization at depth="all" (stricter variant; explain.py logs
     the depth="top" scheme which is protocol-comparable with the Grad-CAM
     control).
  2. Cross-case variability (are the maps input-dependent in practice, or
     near-constant across the audited cases?).

Usage:
    python scripts/xai_report.py [--n 6]

Writes reports/XAI_COMPARISON.md + experiments/results/xai/xai_supplementary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.infer import load_case_images  # noqa: E402
from evaluation.paths import (  # noqa: E402
    EXPERIMENTS_RESULTS, REPORTS_DIR, WEIGHTS_DIR,
)
from src.inference import get_device, load_models  # noqa: E402
from src.preprocessing import make_tensor, preprocess_images  # noqa: E402
from xai.methods import (  # noqa: E402
    ensemble_gradcam,
    ensemble_integrated_gradients,
    occlusion_scan,
)
from xai.quality import (  # noqa: E402
    map_is_degenerate,
    randomization_test_ig,
)

FAMILIES = {
    "gradcam": "CAM (class activation mapping)",
    "gradcampp": "CAM (variant Grad-CAM++)",
    "integrated_gradients": "gradient-path (integrated gradients)",
    "occlusion": "perturbation (occlusion sensitivity)",
}


def _safe_pair_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson that returns None when either map is constant (zero variance):
    a constant map has no direction to correlate — the pair is reported as
    'undefined' instead of poisoning the mean with NaN."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def measure_supplementary(case_ids: list[str], steps_ig: int = 24) -> dict:
    device, _ = get_device()
    sys.path.insert(0, str(REPO_ROOT / "prostate_model_demo"))
    models, _ = load_models(device=device, weights_dir=WEIGHTS_DIR)
    out: dict = {"ig_randomization_all": {}, "cross_case_variability": {}}
    per_method_maps: dict[str, list[np.ndarray]] = {}

    for case_id in case_ids:
        t2w, adc, hbv = load_case_images(case_id)
        preproc = preprocess_images([t2w, adc, hbv])
        tensor = make_tensor(preproc, device).to(device)
        arr = preproc.copy()

        # 1. stricter IG randomization control (all fold-0 parameters)
        res_all = randomization_test_ig(models, arr, device, steps=8, depth="all")
        out["ig_randomization_all"][case_id] = res_all[
            "pearson_similarity_orig_vs_randomized"]

        # 2. cross-case variability per method family
        maps = {
            "gradcam": ensemble_gradcam(models, tensor, device, "gradcam"),
            "integrated_gradients": ensemble_integrated_gradients(
                models, tensor, device, steps=steps_ig),
            "occlusion": occlusion_scan(models, tensor, device,
                                        slab=(2, 32, 32), stride=(4, 48, 48)),
        }
        for name, m in maps.items():
            per_method_maps.setdefault(name, []).append(m)
        print(f"[report] {case_id} done", flush=True)

    for name, list_of_maps in per_method_maps.items():
        stacked = np.stack(list_of_maps)
        pair_corr = [
            c for i in range(len(stacked)) for j in range(i + 1, len(stacked))
            for c in [_safe_pair_corr(stacked[i], stacked[j])]
            if c is not None
        ]
        n_undefined = sum(
            1 for i in range(len(stacked)) for j in range(i + 1, len(stacked))
            if _safe_pair_corr(stacked[i], stacked[j]) is None
        )
        out["cross_case_variability"][name] = {
            "mean_pairwise_pearson": (None if not pair_corr
                                      else float(np.mean(pair_corr))),
            "n_valid_pairs": len(pair_corr),
            "n_undefined_pairs_constant_map": n_undefined,
            "map_volume_std_across_cases": float(stacked.std(axis=0).mean()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6,
                    help="number of audited cases to reuse (same selection as explain)")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse experiments/results/xai/xai_supplementary.json "
                         "instead of re-measuring on GPU")
    args = ap.parse_args()

    quality_csv = EXPERIMENTS_RESULTS / "xai" / "xai_quality.csv"
    if not quality_csv.exists():
        raise FileNotFoundError("run scripts/explain.py first -> no quality artifacts")
    import pandas as pd
    quality = pd.read_csv(quality_csv)

    supp_json = EXPERIMENTS_RESULTS / "xai" / "xai_supplementary.json"
    if args.reuse and supp_json.exists():
        supp = json.loads(supp_json.read_text(encoding="utf-8"))
    else:
        case_ids = sorted({str(c) for c in quality["case_id"]})[: args.n]
        supp = measure_supplementary(case_ids)
        supp_json.write_text(
            json.dumps(supp, indent=2, allow_nan=False, default=float),
            encoding="utf-8")

    # ---- comparison table from REAL measured values ----------------------
    # one row per method FAMILY: mean over the audited cases
    loc = quality[~quality["method"].str.contains("summary")].copy()
    fid = quality[quality["method"].str.contains("summary")].copy()
    fid["family"] = fid["method"].str.removesuffix("_summary")
    fid = fid.groupby("family").mean(numeric_only=True).reset_index()
    loc = loc.groupby("method").mean(numeric_only=True).reset_index()

    lines = [
        "# XAI — comparaison des trois familles (valeurs réellement mesurées)",
        "",
        "> Généré par `scripts/xai_report.py` à partir de "
        "`experiments/results/xai/xai_quality.csv` (run real pipeline, "
        f"{int(quality['case_id'].nunique())} cas de validation, "
        "poids officiels PI-CAI 5 plis, sortie expliquée = score csPCa fusionné). "
        "Aucune valeur n'est simulée ; les moyennes par famille agrègent les cas audités.",
        "",
        "| Méthode | Famille | Loc. IoU top-5% (vs contours experts) | Enrichissement | "
        "Insertion AUC (fidelité↑) | Deletion AUC (fidelité↓) | "
        "Randomisation r (sensibilité modèle ↓) | Stabilité (jitter ±1 voxel) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in fid.sort_values("family").iterrows():
        lrow = loc[loc["method"] == r["family"]]
        iou = float(lrow["iou_top_vs_lesion"].iloc[0]) if len(lrow) else float("nan")
        enr = float(lrow["enrichment"].iloc[0]) if len(lrow) else float("nan")
        lines.append(
            f"| {r['family']} | {FAMILIES.get(r['family'], '?')} | "
            f"{iou:.4f} | {enr:.2f} | {r['insertion_auc']:.3f} | "
            f"{r['deletion_auc']:.3f} | {r['randomization_pearson']:.3f} | "
            f"{r['stability_shift_y_pearson']:.3f} |"
        )
    lines += [
        "",
        "Protocols & reading guide (NOT comparable as a single ranking):",
        "- insertion AUC: higher = the model regains score by re-inserting the",
        "  top-attributed voxels on a blurred baseline (1 = best possible in this",
        "  normalized-curve protocol).",
        "- deletion AUC: LOWER = removing the top-attributed voxels destroys the",
        "  score faster (better faithfulness).",
        "- randomization r: LOWER = the map reacts to weight corruption",
        "  (Adebayo-style control, top-block scheme for CAM/IG-depth-top,",
        "  single-fold coarse scan for occlusion).",
        "- stability: 1 = invariant to a 1-voxel y-shift (0 < r <= 1).",
        "",
        f"## Supplementary control — IG randomization depth='all'",
        "",
        "| case | IG r (fold-0 fully randomized) |",
        "|---|---|",
    ]
    for k, v in supp["ig_randomization_all"].items():
        lines.append(f"| {k} | {v:.3f} |")
    lines += ["", "## Cross-case variability (input-dependence in practice)", "",
              "| method | mean pairwise Pearson across cases | undefined pairs |", "|---|---|---|"]
    for name, v in supp["cross_case_variability"].items():
        cc = v["mean_pairwise_pearson"]
        lines.append(
            f"| {name} | {'undefined (constant maps)' if cc is None else f'{cc:.3f}'} "
            f"| {v.get('n_undefined_pairs_constant_map', 0)} |"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "XAI_COMPARISON.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    print(f"wrote reports/XAI_COMPARISON.md")


if __name__ == "__main__":
    main()
