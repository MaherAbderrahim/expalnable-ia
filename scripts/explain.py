"""Run the XAI stack (Grad-CAM, Grad-CAM++, Integrated Gradients, occlusion)
on a sample of cases; produce overlay figures + per-method quality metrics.

Usage:
    python scripts/explain.py --n-cases 12 [--split val] [--no-occlusion]

Occlusion is computed by default (three complementary XAI families); use
--no-occlusion for quick smoke runs (~8x faster per case).

Selection of cases (documented, pre-registered):
    2 per quartile of the FROZEN score distribution + all FPs/FNs from the
    final error analysis when available — so that the demo shows where the
    model is right, wrong, and uncertain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.infer import load_case_images, preprocess_images, make_tensor  # noqa
from evaluation.postproc import VARIANT_CONFIGS, official_detection_map  # noqa
from evaluation.eval_model import load_case_gt_mask  # noqa

from evaluation.paths import (  # noqa: E402
    DEMO_DIR, PREDICTIONS_DIR, WEIGHTS_DIR, MARKSHEET_PATH,
)
from xai.methods import (  # noqa: E402
    ensemble_gradcam,
    ensemble_integrated_gradients,
    occlusion_scan,
)
from xai.quality import (  # noqa: E402
    insertion_deletion,
    localization_agreement,
    map_is_degenerate,
    randomization_test,
    randomization_test_ig,
    randomization_test_occlusion,
    stability_jitter,
)
from xai.visualize import explain_figure  # noqa: E402


def choose_cases(n_cases: int, split: str = "val",
                 frozen_name: str | None = None) -> list[dict]:
    """Quartile-stratified + error-type-driven selection of demo cases.

    Preferably reads the frozen long-format score table produced by
    scripts/eval_final.py (experiments/results/final_evaluation/<split>_scores_long.csv);
    otherwise falls back to scoring the split on the selected config.
    """
    path = Path("experiments/results/final_evaluation") / f"{split}_scores_long.csv"
    if not path.exists():
        df = load_and_score_any(split)
    else:
        df = pd.read_csv(path)
    frozen_path = Path("experiments/configs/selected_postproc.json")
    if frozen_name is None and frozen_path.exists():
        frozen_name = json.loads(
            frozen_path.read_text(encoding="utf-8"))["selected_config"]
    cfg = frozen_name or "official_default"

    sub = df[df["config"] == cfg].copy()
    if len(sub) < n_cases:
        n_cases = len(sub)
    take: list[dict] = []
    sub["rank"] = sub["likelihood"].rank(pct=True)
    sub = sub.sort_values("likelihood", ascending=False)
    for q, lo in enumerate([0.0, 0.25, 0.5, 0.75]):
        sel = sub[(sub["rank"] >= lo) & (sub["rank"] < lo + 0.25 if q < 3 else sub["rank"] <= 1.0)]
        if len(sel):
            take.append({
                "case_id": str(sel.iloc[0]["case_id"]),
                "reason": f"quartile{q + 1}_of_score",
            })
    # one FP and one FN at the frozen threshold if available
    thr_line = Path("experiments/configs/selected_postproc.json")
    thr = 0.5
    if thr_line.exists():
        thr = json.loads(thr_line.read_text(encoding="utf-8")).get("threshold_f1_val", 0.5)
    sub["pred"] = (sub["likelihood"] >= thr).astype(int)
    for et in ["FP", "FN"]:
        sel = sub[
            ((sub.y_true == 0) & (sub.pred == 1) & (et == "FP"))
            | ((sub.y_true == 1) & (sub.pred == 0) & (et == "FN"))
        ]
        if len(sel):
            take.append({"case_id": str(sel.iloc[0]["case_id"]), "reason": et})
    seen = set()
    out = []
    for t in take:
        if t["case_id"] not in seen and len(out) < n_cases:
            seen.add(t["case_id"])
            out.append(t)
    return out


def load_and_score_any(split: str) -> pd.DataFrame:
    """Minimal fallback score set on the frozen config for case selection only."""
    from evaluation.eval_model import load_split_cases, score_dataframe
    from evaluation.postproc import OFFICIAL_CONFIG
    df = load_split_cases(split)
    return score_dataframe(df, [OFFICIAL_CONFIG], split=split, verbose=False)


def explain_cases(n_cases: int = 12, split: str = "val", with_occlusion: bool = True,
                  device: str = None, quality_steps: int = 16) -> dict:
    """Run the XAI stack on a case sample; produce figures + per-method quality.

    `with_occlusion` defaults to True: the three complementary method families
    (CAM / gradient-path / perturbation) are computed for every case and get
    their own faithfulness / robustness / model-randomization controls.
    """
    from src.inference import load_models, get_device  # Phase-1 loader
    device = device or get_device()[0]
    sys.path.insert(0, str(DEMO_DIR))
    models, _ = load_models(device=device, weights_dir=WEIGHTS_DIR)
    out_root = Path("experiments/results/xai")
    out_root.mkdir(parents=True, exist_ok=True)
    quality_rows = []

    chosen = choose_cases(n_cases, split)
    frozen_path = Path("experiments/configs/selected_postproc.json")
    frozen = (
        json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen_path.exists() else {"selected_config": "official_default",
                                      "threshold_f1_val": 0.5}
    )
    frozen_cfg = next(c for c in VARIANT_CONFIGS
                      if c["name"] == frozen["selected_config"])
    existing_thr = float(frozen["threshold_f1_val"])

    for t in chosen:
        case_id = t["case_id"]
        t0 = time.time()
        t2w, adc, hbv = load_case_images(case_id)
        preproc = preprocess_images([t2w, adc, hbv])
        tensor = make_tensor(preproc, device).to(device)
        npz_path = PREDICTIONS_DIR / split / f"{case_id}.npz"
        npz = np.load(npz_path, allow_pickle=True) if npz_path.exists() else None
        if npz is not None:
            from evaluation.postproc import fused_native_from_npz
            native_fused = fused_native_from_npz(npz, frozen_cfg)
        else:
            native_fused = _fused_fallback(models, tensor)
        det = official_detection_map(native_fused, frozen_cfg)
        likelihood = float(det.max())

        heatmaps = {
            "gradcam": ensemble_gradcam(models, tensor, device, "gradcam"),
            "gradcampp": ensemble_gradcam(models, tensor, device, "gradcampp"),
            "integrated_gradients": ensemble_integrated_gradients(models, tensor, device,
                                                                  steps=24),
        }
        if with_occlusion:
            heatmaps["occlusion"] = occlusion_scan(models, tensor, device)

        # input-dependence guard: a constant/non-finite map must NOT be presented
        # as an explanation (logged + dropped from figure/artifacts)
        degenerate_flags = {}
        for name in list(heatmaps):
            check = map_is_degenerate(heatmaps[name])
            if check["degenerate"]:
                degenerate_flags[name] = check["reasons"]
                print(f"[xai][WARN] {case_id}: degenerate {name} map "
                      f"({', '.join(check['reasons'])}) -> dropped", flush=True)
                heatmaps.pop(name)

        # ground-truth mask -> to the model grid (agreement metrics per method)
        gt_native = load_case_gt_mask(case_id)
        y_true = _labels_lookup().get(case_id)

        arr_t2w = sitk.GetArrayFromImage(t2w)
        fig_path = out_root / f"{case_id}_xai.png"
        explain_figure(
            arr_t2w, det, heatmaps,
            {"case_id": case_id, "likelihood": likelihood,
             "threshold": existing_thr, "y_true": y_true},
            save_path=fig_path,
        )
        np.savez_compressed(out_root / f"{case_id}_heatmaps.npz", **heatmaps)

        # quality: per-method localization agreement + reliability metrics
        if gt_native is not None:
            g_m = _to_model_grid(gt_native)
            for name, h in heatmaps.items():
                agro = localization_agreement(h, g_m)
                quality_rows.append({
                    "case_id": case_id, "reason": t["reason"], "method": name,
                    "likelihood": likelihood, "y_true": y_true, **agro,
                    "degenerate": name in degenerate_flags,
                })
        # per-method faithfulness + robustness + model-randomization controls.
        # Scores below use DIFFERENT protocols per family and are NEVER mixed
        # in a single cross-method ranking (see reports/XAI_COMPARISON.md).
        for name, h in heatmaps.items():
            idres = insertion_deletion(models, preproc.copy(), h, device,
                                       steps=quality_steps)
            row = {
                "case_id": case_id, "reason": t["reason"],
                "method": f"{name}_summary",
                "likelihood": likelihood, "y_true": y_true,
                "insertion_auc": idres["insertion_auc"],
                "deletion_auc": idres["deletion_auc"],
            }
            if name in ("gradcam", "gradcampp"):
                rand = randomization_test(models, preproc.copy(), device, name)
                stab = stability_jitter(models, preproc.copy(), device, name)
            elif name == "integrated_gradients":
                rand = randomization_test_ig(models, preproc.copy(), device, steps=8)
                stab = stability_jitter(
                    models, preproc.copy(), device, name,
                    explainer=lambda ms, tt, dv: ensemble_integrated_gradients(
                        ms, tt, dv, steps=8))
            elif name == "occlusion":
                rand = randomization_test_occlusion(models, preproc.copy(), device)
                stab = stability_jitter(
                    models, preproc.copy(), device, name,
                    explainer=lambda ms, tt, dv: occlusion_scan(
                        ms, tt, dv, slab=(2, 32, 32), stride=(4, 48, 48)))
            else:
                continue
            row["randomization_pearson"] = rand["pearson_similarity_orig_vs_randomized"]
            row["stability_shift_y_pearson"] = stab["pearson_shift_y"]
            quality_rows.append(row)
        print(f"[xai] {case_id} done in {time.time() - t0:.0f}s -> {fig_path.name}",
              flush=True)

    rows = pd.DataFrame(quality_rows)
    rows.to_csv(out_root / "xai_quality.csv", index=False)
    num_cols = [c for c in rows.columns
                if c not in ("case_id", "reason", "method", "y_true")]
    summary = rows.groupby("method")[num_cols].mean(numeric_only=True).reset_index()
    summary.to_json(out_root / "xai_quality_summary.json", orient="records", indent=2)
    return {"n_cases": len(chosen), "quality_csv": str(out_root / "xai_quality.csv")}


def _fused_fallback(models, tensor):
    """fused model-grid heatmap when npz is not present (recompute locally)."""
    from scipy.ndimage import gaussian_filter
    with torch.no_grad():
        outs = [torch.sigmoid(m(tensor))[:, 1].cpu().numpy() for m in models]
    fused = np.mean([gaussian_filter(o[0], sigma=1.5) for o in outs], axis=0)
    return fused.astype(np.float32)


def _labels_lookup() -> dict[str, int]:
    mk = pd.read_csv(MARKSHEET_PATH)
    mk["case_id"] = mk.patient_id.astype(str) + "_" + mk.study_id.astype(str)
    return dict(zip(mk.case_id, (mk.case_csPCa.astype(str) == "YES").astype(int)))


def _to_model_grid(mask_native: np.ndarray) -> np.ndarray:
    """Nearest-neighbor zoom of the expert mask onto the (20,256,256) grid."""
    from scipy.ndimage import zoom
    scale = [m / n for m, n in zip((20, 256, 256), mask_native.shape)]
    return zoom(mask_native.astype(np.float32), scale, order=0) > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--with-occlusion", action="store_true",
                    help="compat flag: occlusion is now computed by default")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="skip the occlusion scan (fast runs)")
    ap.add_argument("--quality-steps", type=int, default=16,
                    help="insertion/deletion steps for the faithfulness controls")
    args = ap.parse_args()
    res = explain_cases(args.n_cases, args.split,
                        with_occlusion=not args.no_occlusion,
                        quality_steps=args.quality_steps)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
