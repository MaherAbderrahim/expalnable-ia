"""Simple demo interface (Streamlit) — phase 1: MODEL only (no XAI yet).

Run with:
    .venv\\Scripts\\streamlit run scripts/streamlit_app.py

The interface shows: sample selection, prediction run, the input MRI,
the model detection map raw output, the probability, and info blocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import SimpleITK as sitk
import streamlit as st

from src.data_loader import (
    get_case_ground_truth,
    list_available_cases,
    load_case,
    load_lesion_annotation,
    load_marksheet,
)
from src.inference import get_device, load_models, run_inference, save_result
from src.visualization import case_figure

st.set_page_config(page_title="Prostate AI Model Demo", layout="wide")

OFFICIAL_CASE_ID = "10032_1000032"


@st.cache_resource(show_spinner="Loading the 5 U-Net checkpoints ...")
def load_model_cache():
    device, note = get_device()
    models, folds = load_models(device=device)
    return models, folds, device, note


@st.cache_resource(show_spinner=True)
def load_mark():
    return load_marksheet()


# ---------------------------------------------------------------------------
st.title("Prostate AI Model Demo")
st.caption("PI-CAI reference U-Net (5-fold ensemble) — csPCa detection on biparametric MRI")

st.markdown(
    "**PHASE 1 — MODEL** (this demo): DATA → MODEL → PREDICTION.\n\n"
    "**PHASE 2 — EXPLAINABILITY** (future work): PREDICTION → XAI → EXPLANATION → EVALUATION."
)

# ---- sample selection -------------------------------------------------------
case_ids = list_available_cases()
if OFFICIAL_CASE_ID not in case_ids:
    case_ids = [OFFICIAL_CASE_ID] + case_ids
sample = st.selectbox("Choose sample", case_ids, index=0)

if st.button("Run prediction", type="primary"):
    with st.spinner("Running inference (5-model ensemble + TTA)..."):
        case = load_case(sample)
        try:
            case.clinical.update(get_case_ground_truth(load_mark(), case.case_id))
        except Exception:
            pass
        case.lesion_annotation = load_lesion_annotation(case.case_id)

        models, folds, device, device_note = load_model_cache()
        result = run_inference(case, device=device, models=models)
        save_result(result, Path(__file__).resolve().parents[1] / "outputs" / "predictions")
        fig_path = case_figure(
            case, result,
            save_path=Path(__file__).resolve().parents[1] / "outputs" / "figures" / f"streamlit_{case.case_id}.png",
            title=f"case {case.case_id}",
        )

        st.success(f"csPCa probability: {result.case_level_likelihood * 100:.1f} % "
                   f"(threshold 0.50)")
        st.image(str(fig_path), caption="input MRI + model detection map + expert contour (GT)")

        # ---- numeric info ------------------------------------------------------
        gt_label = case.clinical.get("case_level_csPCa", None)
        c1, c2, c3 = st.columns(3)
        c1.metric("csPCa probability", f"{result.case_level_likelihood * 100:.1f} %")
        c2.metric("Ground truth", ("ISUP ≥ 2" if gt_label else "no csPCa")
                  if gt_label is not None else "n/a")
        iou = None
        if case.lesion_annotation is not None:
            from src.evaluation import lesion_localization_case
            iou = lesion_localization_case(result.detection_map, case.lesion_annotation)["iou_detection_vs_gt"]
        if iou is not None:
            c3.metric("IoU vs expert contour", f"{iou:.2f}")
        else:
            c3.metric("IoU vs expert contour", "n/a")

        # ---- info blocks -------------------------------------------------------
        st.divider()
        left, right = st.columns(2)
        with left:
            st.markdown("**Input MRI (what the model sees)**")
            st.write(f"- case: `{case.case_id}`")
            st.write(f"- T2W + ADC + DWI(HBV), axial, co-registered")
            st.write(f"- T2W native shape (x,y,z): {case.images['t2w'].GetSize()}")
            st.write(f"- after preprocessing: (1, 3, 20, 256, 256) tensor")
        with right:
            st.markdown("**Model output**")
            st.write(f"- raw ensemble heatmap: (20, 256, 256)")
            st.write(f"- final detection map on the T2W grid: {result.detection_map.shape}")
            st.write(f"- case-level likelihood = max of the detection map")
            st.write(f"- timings: preprocess {result.preprocessing_time_s:.2f}s, "
                     f"inference {result.inference_time_s:.2f}s ({device_note})")

        st.divider()
        st.markdown("**Model information**")
        st.write("- U-Net 3D (MONAI), channels 32-64-128-256-512-1024, "
                 "strides [(2,2,2),(1,2,2),(1,2,2),(1,2,2),(2,2,2)]")
        st.write("- 5-fold cross-validation ensemble, trained on the PI-CAI public dataset "
                 "(1,295 human-annotated cases, 1,500 total)")
        st.write("- weights: `unet_F0.pt` … `unet_F4.pt` (127 MB each), Apache-2.0, "
                 "from [picai_unet_gc_algorithm](https://github.com/DIAGNijmegen/picai_unet_gc_algorithm)")
        st.write("- published performance (PI-CAI hidden test set): AUROC 0.878, AP 0.558 "
                 "(Saha et al., Lancet Oncology 2024)")

        st.markdown("**Dataset information**")
        st.write("- PI-CAI Public Training and Development Dataset v2.0 "
                 "(Zenodo record 6624726), CC BY-NC 4.0 — research only")
        st.write("- expert lesion contours + clinical labels from "
                 "[picai_labels](https://github.com/DIAGNijmegen/picai_labels)")
        st.write("- this demo runs on a small subset of cases; results shown are "
                 "**pipeline verification, not a clinical evaluation**")

        st.markdown("**Inference information**")
        st.write("- preprocessing (official replication): COM alignment → center "
                 "crop/pad → resampling to 3.0×0.5×0.5 mm → z-score norm (p99.5)")
        st.write("- inference: 5 U-Nets × 2 TTA variants (left-right flip), "
                 "sigmoid, gaussian smoothing, fold-averaging")
        st.write("- postprocessing: reverse resampling, lesion candidate extraction "
                 "('dynamic' threshold), ring-candidate removal")

        if result.case_level_likelihood < 0.5 and "reference" not in case.clinical:
            st.info("Below-threshold scores here are expected: with a threshold of 0.50 the "
                    "official model reaches 70.9% sensitivity / PPV 35.5% — see the docs.")
else:
    st.info("Select a sample and click **Run prediction**.")

st.caption(
    "Phase 2 (not implemented here): explainability (Grad-CAM, Integrated Gradients, "
    "occlusion, non-native detection-map analysis, PI-RADS-style reporting)."
)
