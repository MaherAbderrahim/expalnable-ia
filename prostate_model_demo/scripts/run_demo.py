"""Run the model demo: PROSTATE AI MODEL DEMO (phase 1 — model only, no XAI).

Usage:
  python scripts/run_demo.py                        # all downloaded demo cases
  python scripts/run_demo.py --offline              # only the official repo test case
  python scripts/run_demo.py --add-case-count 5     # download, then run 5 extra cases
  python scripts/run_demo.py --case 10032_1000032   # one specific case
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402

from src.config import OUTPUTS_DIR  # noqa: E402
from src.data_loader import (  # noqa: E402
    PicaIcase,
    get_case_ground_truth,
    list_available_cases,
    load_case,
    load_lesion_annotation,
    load_marksheet,
)
from src.evaluation import lesion_localization_case  # noqa: E402
from src.inference import get_device, load_models, run_inference, save_result  # noqa: E402
from src.visualization import case_figure  # noqa: E402

LINE = "=" * 62
OFFICIAL_CASE_ID = "10032_1000032"


def run(case_id: str, annotate: bool, models, device: str):
    """Run inference + visualization for one case, attach ground truth if any."""
    case = load_case(case_id)

    # case-level ground truth from the official marksheet, when available
    try:
        mark = load_marksheet()
        case.clinical.update(get_case_ground_truth(mark, case.case_id))
    except Exception as e:
        case.clinical["marksheet"] = f"unavailable ({type(e).__name__}: {e})"

    # expert lesion annotation, when available
    if annotate:
        case.lesion_annotation = load_lesion_annotation(case.case_id)
        case.clinical["expert_annotation"] = (
            "available" if case.lesion_annotation is not None else "not available"
        )

    result = run_inference(case, device=device, models=models)

    saved = save_result(result, OUTPUTS_DIR / "predictions")
    fig = case_figure(
        case,
        result,
        save_path=OUTPUTS_DIR / "figures" / f"{case.case_id}.png",
        title=(f"case {case.case_id} - csPCa likelihood "
               f"{result.case_level_likelihood * 100:.1f} %"),
    )
    return case, result, fig


def print_case_report(case: PicaIcase, result, fig) -> None:
    t2w, adc = case.images["t2w"], case.images["adc"]
    print(f"Patient/case   : {case.case_id}")
    print(f"T2W shape xyz  : {t2w.GetSize()} at spacing "
          f"{tuple(round(s, 2) for s in t2w.GetSpacing())} mm")
    print(f"ADC spacing    : {tuple(round(s, 2) for s in adc.GetSpacing())} mm")
    print(f"Input shape    : {result.preprocessed_shape} (resampled to 3.0x0.5x0.5 mm)")
    print(f"Output shape   : raw heatmap {tuple(result.raw_softmax_heatmap.shape)} -> "
          f"detection map {tuple(result.detection_map.shape)}")

    score = result.case_level_likelihood
    gt_label = case.clinical.get("case_level_csPCa", None)
    isup = case.clinical.get("max_ISUP", None)
    pirads = case.clinical.get("lesion_PIRADS", "")

    print()
    print("Patient prediction:")
    print(f"csPCa probability : {score * 100:5.1f} %   "
          f"(threshold 0.50 -> {'csPCa detected' if score >= 0.5 else 'no csPCa'})")
    if gt_label is not None:
        match = "OK" if bool(gt_label) == (score >= 0.5) else "MISMATCH"
        extra = f" (max ISUP {isup}, PI-RADS {pirads})" if (gt_label and isup is not None) else ""
        print(f"Ground truth      : {'ISUP >= 2' if gt_label else 'no csPCa'}{extra}   -> {match}")
    if case.lesion_annotation is not None:
        loc = lesion_localization_case(result.detection_map, case.lesion_annotation)
        print(f"Lesion localization: IoU(det_map vs expert contour) = {loc['iou_detection_vs_gt']:.2f} "
              f"(GT has lesion: {loc['gt_has_lesion']})")
    print("Detection map    : generated successfully")
    print(f"Timing           : preprocess {result.preprocessing_time_s:.2f}s | "
          f"inference {result.inference_time_s:.2f}s "
          f"({result.n_models_used} ensemble members, device={result.device})")
    if fig is not None:
        print(f"Figure saved     : {fig}")


def golden_test():
    """Compare our run on the official test case with the reference shipped
    by the model authors (case-level likelihood + detection map)."""
    case = load_case(OFFICIAL_CASE_ID)
    ref_likelihood = case.clinical.get("reference_case_level_likelihood")
    ref_map = case.clinical.get("reference_detection_map")

    print(LINE)
    print("GOLDEN TEST — official case 10032_1000032")
    print("(comparison against outputs shipped by the model authors)")

    device, _ = get_device()
    result = run_inference(case, device=device)

    status = []
    if ref_likelihood is not None:
        diff = abs(result.case_level_likelihood - ref_likelihood)
        line = (f"likelihood  : ref {ref_likelihood:.16f} | ours {result.case_level_likelihood:.16f} "
                f"| |diff| {diff:.2e}")
        print(line)
        status.append(diff < 1e-3)

    if ref_map is not None:
        ref_arr = sitk.GetArrayFromImage(ref_map)
        diff_volume = np.abs(result.detection_map - ref_arr)
        n_diff = int((diff_volume > 1e-3).sum())
        frac = n_diff / ref_arr.size
        print(f"detection map: shapes ref {ref_arr.shape} vs ours "
              f"{tuple(result.detection_map.shape)}; |diff| > 1e-3 on "
              f"{n_diff}/{ref_arr.size} voxels ({frac:.3%})")
        print("              (differences, if any, come from platform floating-point")
        print("               rounding in the conv/resampling kernels — see docs)")
        status.append(frac < 1e-3)

    verdict = "PASSED" if all(status) else "NOT IDENTICAL — see notes above"
    print(f"\nverdict      : {verdict}")
    print(LINE)
    return verdict, result


def summary_block(summary: list[dict]) -> None:
    print(LINE)
    print("CASE-LEVEL SUMMARY (pipeline verification only — NOT a clinical evaluation)")
    if not summary:
        print("no cases completed")
        return
    pos = [s for s in summary if s.get("gt_positive") is True]
    neg = [s for s in summary if s.get("gt_positive") is False]
    errors = [s for s in summary if s.get("error")]
    detected = [s for s in summary if s.get("likelihood", -1) >= 0.5]
    print(f"cases run          : {len(summary)} "
          f"(positives: {len(pos)}, negatives: {len(neg)}, "
          f"errors: {len(errors)}, gt unavailable: "
          f"{len(summary) - len(pos) - len(neg) - len(errors)})")
    print(f"score >= 0.50      : {len(detected)} case(s)")
    ok = [s for s in summary if "inference_time_s" in s]
    if ok:
        print(f"mean inference time: {sum(s['inference_time_s'] for s in ok) / len(ok):.2f} s/case")
    annotated = [s for s in summary if s.get("iou") is not None]
    if annotated:
        print(f"mean IoU vs expert : {sum(s['iou'] for s in annotated) / len(annotated):.2f} "
              f"(on {len(annotated)} case(s) with expert contours)")
    print(LINE)
    print("Phase 2 of the project (not implemented yet): Explainable AI")
    print("(Grad-CAM / Integrated Gradients / occlusion on top of these outputs)")
    print(LINE)


def add_cases(n: int) -> None:
    script = PROJECT_ROOT / "scripts" / "download_or_prepare_data.py"
    print(f"[data] picking + downloading {n} additional annotated case(s) ...")
    subprocess.run([sys.executable, str(script), "--add-case-count", str(n)], check=False)


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="PI-CAI baseline U-Net demo runner.")
    ap.add_argument("--case", nargs="*", default=None, help="case ids (default: all downloaded)")
    ap.add_argument("--add-case-count", type=int, default=0,
                    help="download this many extra annotated cases before running")
    ap.add_argument("--offline", action="store_true",
                    help="run only the official repo test case (offline)")
    ap.add_argument("--skip-golden", action="store_true",
                    help="skip the golden test against the authors' reference outputs")
    args = ap.parse_args()

    print(LINE)
    print("PROSTATE MODEL DEMO — PI-CAI reference U-Net (5-fold ensemble)")
    print("(phase 1: model only — no XAI yet)")
    print(LINE)

    device, device_note = get_device()
    print(f"Device         : {device_note}")

    # ---- data selection ---------------------------------------------------
    if args.offline:
        case_ids = [OFFICIAL_CASE_ID]
    else:
        if args.add_case_count > 0:
            add_cases(args.add_case_count)
        case_ids = args.case or list_available_cases()
        if OFFICIAL_CASE_ID in case_ids:
            pass
        else:
            case_ids = [OFFICIAL_CASE_ID] + case_ids

    print(f"Dataset        : PI-CAI Public Training and Development Dataset v2.0 (subset)")
    print(f"Cases          : {', '.join(case_ids)}")
    print(f"Modalities     : T2W + ADC + DWI(HBV) (axial)")
    print(f"Model          : U-Net 3D, 5-fold cross-validation ensemble")

    t0 = time.perf_counter()
    models, loaded_folds = load_models(device=device)
    print(f"Checkpoint     : loaded ({len(models)} folds: {loaded_folds}) in "
          f"{time.perf_counter() - t0:.1f} s")

    # ---- run all selected cases (except the official one — golden test) ----
    other_ids = [c for c in case_ids if c != OFFICIAL_CASE_ID]
    summary = []
    if other_ids:
        for cid in other_ids:
            print("\n" + "-" * 62)
            try:
                case, result, fig = run(cid, annotate=True, models=models, device=device)
            except Exception as e:
                print(f"[ERROR] case {cid}: {type(e).__name__}: {e}")
                summary.append({"case_id": cid, "error": f"{type(e).__name__}: {e}"})
                continue
            print_case_report(case, result, fig)
            loc_iou = None
            if case.lesion_annotation is not None:
                loc = lesion_localization_case(result.detection_map, case.lesion_annotation)
                loc_iou = loc["iou_detection_vs_gt"]
            summary.append({
                "case_id": case.case_id,
                "likelihood": result.case_level_likelihood,
                "gt_positive": case.clinical.get("case_level_csPCa"),
                "iou": loc_iou,
                "inference_time_s": result.inference_time_s,
            })

    # ---- official case + golden test --------------------------------------
    verdict = None
    if not args.skip_golden and OFFICIAL_CASE_ID not in [s.get("case_id") for s in summary]:
        verdict, result = golden_test()
        case = load_case(OFFICIAL_CASE_ID)
        try:
            mark = load_marksheet()
            case.clinical.update(get_case_ground_truth(mark, case.case_id))
        except Exception:
            pass
        case.lesion_annotation = load_lesion_annotation(case.case_id)
        print_case_report(case, result, None)
        loc_iou = None
        if case.lesion_annotation is not None:
            loc = lesion_localization_case(result.detection_map, case.lesion_annotation)
            loc_iou = loc["iou_detection_vs_gt"]
        save_result(result, OUTPUTS_DIR / "predictions")
        case_figure(
            case, result,
            save_path=OUTPUTS_DIR / "figures" / f"{case.case_id}.png",
            title=f"case {case.case_id} - csPCa {result.case_level_likelihood * 100:.1f} %",
        )
        summary.append({
            "case_id": case.case_id,
            "likelihood": result.case_level_likelihood,
            "gt_positive": case.clinical.get("case_level_csPCa"),
            "iou": loc_iou,
            "inference_time_s": result.inference_time_s,
        })
    elif args.skip_golden and OFFICIAL_CASE_ID in case_ids:
        cid = OFFICIAL_CASE_ID
        print("\n" + "-" * 62)
        case, result, fig = run(cid, annotate=True, models=models, device=device)
        print_case_report(case, result, fig)
        loc_iou = None
        if case.lesion_annotation is not None:
            loc = lesion_localization_case(result.detection_map, case.lesion_annotation)
            loc_iou = loc["iou_detection_vs_gt"]
        summary.append({
            "case_id": cid,
            "likelihood": result.case_level_likelihood,
            "gt_positive": case.clinical.get("case_level_csPCa"),
            "iou": loc_iou,
            "inference_time_s": result.inference_time_s,
        })

    summary_block(summary)

    # persist a machine-readable summary
    out = OUTPUTS_DIR / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"golden_test": verdict, "cases": summary}, f, indent=2)
    print(f"summary.json saved: {out}")


if __name__ == "__main__":
    main_cli()
