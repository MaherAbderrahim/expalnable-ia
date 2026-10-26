# -*- coding: utf-8 -*-
"""Build + EXECUTE phase1_phase2_demo.ipynb (repo root), export its HTML,
and let the notebook feed the presentation site.

Single source of truth: the executed notebook produces
- phase1_phase2_demo.ipynb  (real outputs, images included)
- phase1_phase2_demo.html   (nbconvert, embedded images)
- reports/demo/data/presentation_data.json (numbers for the site)
- reports/demo/assets/**    (figures served by the presentation site)
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "phase1_phase2_demo.ipynb"
HTML_PATH = ROOT / "phase1_phase2_demo.html"
SITE_NB_DIR = ROOT / "reports" / "demo" / "nb"

CELL_INTRO = r"""# Explainable AI for Prostate Cancer Detection on MRI — phase1_phase2_demo

Ce notebook est la **base technique de la présentation web** : les cellules
ci-dessous exécutent le pipeline réel du dépôt et produisent les valeurs et les
figures affichées sur le site.

Le projet s'appuie sur le détecteur de csPCa (clinically significant prostate
cancer) de PI-CAI (Saha et al., *Lancet Oncology* 2024) — un ensemble UNet 3D à
5 folds entraîné sur 1 500 examens IRM multiparamétriques publics — et applique
ensuite des méthodes d'**Explainable AI** pour analyser sa décision.

> **Portée expérimentale.** Les poids officiels ont été entraînés sur
> l'intégralité des 1 500 cas publics ; les cas locaux d'évaluation ont donc été
> observés pendant l'entraînement. Les métriques ci-dessous constituent une borne
> supérieure pour l'usage en conditions réelles. L'objectif de ce travail est la
> construction d'un pipeline d'évaluation honnête et auditable, et l'application
> d'explicabilité vérifiée sur le modèle réel — pas un record de classification.
"""

CELL_PATHS = r"""import json, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image as IPyImage, display

ROOT = Path.cwd()
if not (ROOT / "prostate_model_demo").is_dir():
    ROOT = ROOT.parent
DEMO = ROOT / "prostate_model_demo"
sys.path.insert(0, str(DEMO))

from src.inference import load_models, make_tensor, preprocess_images, run_inference
from src.data_loader import get_case_ground_truth, list_available_cases, load_case, load_marksheet
from src.visualization import case_figure
from src.config import CASE_LEVEL_THRESHOLD, IMG_SPEC, UNET_HYPERPARAMS, WEIGHTS_DIR

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ASSETS = ROOT / "reports" / "demo" / "assets"
(ASSETS / "cases").mkdir(parents=True, exist_ok=True)
(ASSETS / "xai").mkdir(parents=True, exist_ok=True)
DATA_OUT_DIR = ROOT / "reports" / "demo" / "data"
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

def rel(p: Path) -> str:
    return str(p.relative_to(ROOT / "reports" / "demo")).replace("\\", "/")

def show(obj) -> None:
    display(obj)

print("device      :", DEVICE, "| torch", torch.__version__)
print("weights_dir :", WEIGHTS_DIR)
print("assets dir  :", ASSETS)
"""

CELL_EQUIP = r"""## Matériel et conventions d'inférence

L'inférence suit les constantes officielles de `picai_unet_gc_algorithm` :
entrée tri-modale (T2W, ADC, DWI/HBV) prétraitée en `(20, 256, 256)` par coupe,
espacement `(3,0 / 0,5 / 0,5)` mm, puis ensemble de 5 checkpoints avec TTA
miroir horizontal et lissage gaussien σ = 1,5.
"""

CELL_DATA_LOAD = r"""marksheet = load_marksheet()
case_ids = list_available_cases()
meta_rows = []
for cid in case_ids:
    gt = get_case_ground_truth(marksheet, cid)
    meta_rows.append({
        "case_id": cid,
        "csPCa": gt["case_level_csPCa"],
        "PI-RADS": gt.get("lesion_PIRADS", ""),
        "ISUP_max": gt.get("max_ISUP"),
        "center": gt.get("center", ""),
    })
meta_df = pd.DataFrame(meta_rows)
print(f"{len(case_ids)} cas de démonstration disponibles")
DATA["cases_meta"] = meta_df.astype(object).where(pd.notnull(meta_df), None).to_dict(orient="records")
meta_df
"""

CELL_MODEL_LOAD = r"""t0 = time.time()
models, folds = load_models(device=DEVICE)
model_load_time_s = time.time() - t0
n_params = sum(p.numel() for p in models[0].parameters())
weights_mb = sorted(p.stat().st_size / 1e6 for p in WEIGHTS_DIR.glob("unet_F*.pt"))

print("Model                : PI-CAI UNet 3D — ensemble 5 folds")
print("Status               : ✓ Model loaded")
print(f"✓ Weights loaded     : 5 checkpoints ({weights_mb[0]:.1f} Mo chacun, incl. LFS)")
print("✓ Configuration loaded (constantes officielles picai_unet_gc_algorithm)")
for k, v in IMG_SPEC.items():
    print(f"   - {k:12s}: {v}")
hm = UNET_HYPERPARAMS
print(f"   - hyperparams : strides {hm['model_strides']} | features {hm['model_features']}")
print(f"   - params/fold : {n_params/1e6:.1f} M")
print("✓ Inference ready   : TTA miroir + σ 1.5 + moyenne 5 folds sur", DEVICE,
      f"(chargement {model_load_time_s:.1f} s)")

DATA["model"] = {
    "name": "PI-CAI UNet 3D — ensemble 5 folds (Saha et al., Lancet Oncology 2024)",
    "n_folds": len(models),
    "params_per_fold_millions": round(n_params / 1e6, 1),
    "weight_mb_per_fold": round(weights_mb[0], 1),
    "input_shape": IMG_SPEC["image_shape"],
    "spacing": IMG_SPEC["spacing"],
    "channels": IMG_SPEC["num_channels"],
    "threshold_official": CASE_LEVEL_THRESHOLD,
    "gaussian_sigma": 1.5,
    "tta_flip": True,
    "device": DEVICE,
    "load_seconds": round(model_load_time_s, 2),
}
"""

CELL_P1_MD = r"""# Phase 1 — Detection Model

Cette première phase est consacrée à l'intégration et à la validation du modèle
de détection utilisé comme base du système d'Explainable AI : le pipeline
officiel (prétraitement, ensemble 5-fold, TTA, fusion, post-traitement) est
reproduit fidèlement, puis validé sur le **cas d'or** des auteurs, le cas
`10032_1000032`, pour lequel la probabilité de référence est connue.
"""

CELL_GOLDEN = r"""GOLDEN_ID = "10032_1000032"
GOLDEN_REFERENCE = 0.52692652   # référence publiée avec le dépôt officiel

golden_case = load_case(GOLDEN_ID)
golden_result = run_inference(golden_case, device=DEVICE, models=models,
                              model_load_time_s=model_load_time_s)
likelihood = float(golden_result.case_level_likelihood)
delta = abs(likelihood - GOLDEN_REFERENCE)
print(f"score du modèle        : {likelihood:.10f}")
print(f"référence des auteurs  : {GOLDEN_REFERENCE:.8f}")
print(f"écart absolu           : {delta:.2e}")
print("test d'or du pipeline  :", "PASSED" if delta < 5e-4 else "FAILED")

golden_fig = case_figure(golden_case, golden_result,
                         save_path=ASSETS / "cases" / "case_10032_full.png",
                         title=f"Reproduction fidèle — {GOLDEN_ID} (csPCa {likelihood*100:.1f} %)")
DATA["golden"] = {"case_id": GOLDEN_ID, "score": likelihood, "reference": GOLDEN_REFERENCE,
                  "delta": delta, "passed": bool(delta < 5e-4),
                  "figure": rel(golden_fig)}
show(IPyImage(filename=str(golden_fig)))
"""

CELL_EXAMP_MD = r"""## De la donnée à la décision : cas représentatifs

Chaque examen traverse la même chaîne : **input → inférence → score → décision**.
Trois profils sont conservés pour la présentation : une détection franche, un
cancer manqué sous le seuil officiel, et un cas ambigu proche du seuil. Les
valeurs affichées sur le site sont produites par ces cellules.
"""

CELL_RUN_CASES = r"""thr = CASE_LEVEL_THRESHOLD
records = []
for cid in case_ids:
    c = load_case(cid)
    r = run_inference(c, device=DEVICE, models=models, model_load_time_s=model_load_time_s)
    gt = get_case_ground_truth(marksheet, cid)
    s = float(r.case_level_likelihood)
    is_pos = gt["case_level_csPCa"]
    records.append({
        "case_id": cid,
        "score": s,
        "csPCa": is_pos,
        "PI-RADS": gt.get("lesion_PIRADS", ""),
        "pred": "suspicion" if s >= thr else "aucune",
        "verdict": ("TP" if s >= thr else "FN") if is_pos else ("FP" if s >= thr else "TN"),
    })
cases_df = pd.DataFrame(records)
print(cases_df.to_string(index=False))
DATA["cases"] = cases_df.astype(object).where(pd.notnull(cases_df), None).to_dict(orient="records")
print(f"\n7 cas, seuil officiel {thr} : {sum(1 for r in records if r['verdict']=='TP')}/4 cancers réels détectés")
"""

CELL_PANELS = r"""import SimpleITK as sitk

def case_panel(case_id: str, out_name: str, note: str = "") -> Path:
    c = load_case(case_id)
    r = run_inference(c, device=DEVICE, models=models, model_load_time_s=model_load_time_s)
    t2w = sitk.GetArrayFromImage(sitk.Cast(c.images["t2w"], sitk.sitkFloat32))
    det = np.asarray(r.detection_map, dtype=np.float32)
    zi = int(np.argmax(det.max(axis=(1, 2))))
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 5.0), dpi=150)
    ax[0].imshow(t2w[zi], cmap="gray"); ax[0].axis("off")
    ax[0].set_title(f"IRM T2W — coupe z={zi}", fontsize=11)
    ax[1].imshow(t2w[zi], cmap="gray")
    ax[1].imshow(det[zi], cmap="Reds", alpha=np.clip(det[zi] * 2.0, 0, 0.8))
    ax[1].axis("off")
    ax[1].set_title(f"détection >0.5 — score {float(r.case_level_likelihood):.2f}", fontsize=11)
    out = ASSETS / "cases" / out_name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out

SELECTION = {
    "10032_1000032": ("case_10032_panel.png", "detection franche"),
    "10012_1000012": ("case_10012_panel.png", "cancer manqué"),
    "10008_1000008": ("case_10008_panel.png", "cas ambigu"),
}
panels = {}
for cid, (fname, role) in SELECTION.items():
    p = case_panel(cid, fname)
    rec = next(r for r in records if r["case_id"] == cid)
    panels[cid] = {"role": role, "figure": rel(p), "score": rec["score"],
                   "csPCa": rec["csPCa"], "pred": rec["pred"], "verdict": rec["verdict"]}
    print(f"{cid} -> {rel(p)}")
DATA["panels"] = panels
for cid, info in panels.items():
    display(IPyImage(filename=str(ROOT / "reports" / "demo" / info["figure"])))
"""

CELL_THR_MD = r"""## Le seuil de décision

Le seuil de la Phase 1 est celui utilisé par l'évaluation officielle PI-CAI
(`0.5`), sans ajustement pour embellir les résultats. Chaque décision découle
trivbialement de la comparaison score / seuil. La recalibration des seuils
(Phase 2) est traitée plus loin comme un résultat séparé, mesuré sur la
validation.
"""

CELL_THR = r"""print(f"Seuil voulu (et utilisé) en Phase 1 : {CASE_LEVEL_THRESHOLD}")
print("  origine : point opératoire publié, sensitivity 70.9 % / PPV 35.5 %\n")
for r in records:
    s = r["score"]
    comp = ">=" if s >= CASE_LEVEL_THRESHOLD else " <"
    line = f"  {r['case_id']}   score {s:.3f}  {comp}  {CASE_LEVEL_THRESHOLD}  ->  "
    line += "SUSPICION DETECTEE" if s >= CASE_LEVEL_THRESHOLD else "aucune alarme"
    if r["csPCa"] and s < CASE_LEVEL_THRESHOLD:
        line += "   (cancer manqué — le site le présentera tel quel)"
    print(line)
DATA["threshold_phase1"] = {
    "value": CASE_LEVEL_THRESHOLD,
    "rationale": "point opératoire officiel PI-CAI : sensitivity 70.9 % / PPV 35.5 %"
                 " (Saha et al., Lancet Oncology 2024) — aucun ajustement local.",
}
"""

CELL_P2_TRANSITION = r"""# Transition vers la Phase 2 — Explainable AI

Le modèle permet d'obtenir une prédiction à partir des données d'imagerie.
Cependant, la prédiction seule ne permet pas d'identifier les éléments ayant
contribué à la décision. La seconde phase introduit donc les méthodes
d'Explainable AI afin d'analyser cette décision — avec une instrumentation
rigoureuse de l'évaluation (sélection figée sur la validation, test scoré une
seule fois après gel).
"""

CELL_BENCH = r"""p2_summary = json.load(open(ROOT / "experiments" / "results" / "final_evaluation" / "eval_final_summary.json", encoding="utf-8"))
bench_rows = []
for split in ("val", "test"):
    for cfg in ("official_default", "no_extract"):
        b = p2_summary["splits"][split][cfg]
        op = b["operating_point"]
        bench_rows.append({
            "split": split, "config": cfg, "threshold": round(op["threshold"], 4),
            "auroc": round(b["auc_roc"], 4), "ap": round(b["pr_auc"], 4),
            "sens": round(op["sensitivity"], 3), "spec": round(op["specificity"], 3),
            "precision": round(op["precision"], 3), "f1": round(op["f1"], 3),
            "balanced_accuracy": round(op["balanced_accuracy"], 3),
            "ece": round(b["ece_10bins"], 3), "n_cases": op["n_cases"],
        })
bench_df = pd.DataFrame(bench_rows)
print(bench_df.to_string(index=False))

st = p2_summary["statistics"]["test"]
p_delong = st["delong_official_vs_selected"]["p_value"]
bd = st["bootstrap_auc_diff_official_vs_selected"]
ci_auroc = st["auc_roc_selected_ci"]
fv = p2_summary["statistics"]["fold_run_variance_val"]["auc_roc"]
print("\nDeLong (officiel vs figé, test) : p =", round(p_delong, 3),
      "→ différence de classement non significative")
print(f"ΔAUROC bootstrap     : {bd['mean_diff']:.4f}  IC95 [{bd['ci_lower']:.4f} ; {bd['ci_upper']:.4f}]")
print(f"AUROC test (figé)    : {ci_auroc['value']:.4f}  IC95 [{ci_auroc['ci_lower']:.3f} ; {ci_auroc['ci_upper']:.3f}]")
print(f"variance des folds   : {fv['mean']:.3f} ± {fv['sd']:.3f} (5 folds seuls, val)")

zp = json.load(open(ROOT / "experiments" / "configs" / "selected_postproc.json", encoding="utf-8"))
selected_config = zp.get("selected_config", "no_extract")
selected_thr = float(zp["threshold_f1_val"])
print(f"\nRègle figée Phase 2  : config '{selected_config}', seuil {selected_thr:.4f}"
      " (max-F1 sur la VALIDATION ; le test n'a été scoré qu'après gel).")

DATA["benchmark"] = bench_rows
DATA["benchmark_stats"] = {
    "delong_p_test": p_delong,
    "bootstrap_auc_diff": {"mean": bd["mean_diff"], "ci_lower": bd["ci_lower"], "ci_upper": bd["ci_upper"]},
    "auroc_selected_ci_test": ci_auroc,
    "fold_run_variance_val": {"mean": fv["mean"], "sd": fv["sd"]},
}
DATA["selected_postproc"] = {"config": selected_config, "threshold": selected_thr}
"""

CELL_XAI_MD = r"""# Phase 2 — Explainable AI

Les méthodes appliquées au modèle **réel** sont celles implémentées dans le
dépôt (`xai/methods.py`) :

| Méthode | Principe | Ce qu'elle montre |
|---|---|---|
| **Grad-CAM 3D** | pondération des activations de la dernière connexion utile par leurs gradients moyens | où le réseau regarde |
| **Grad-CAM++ 3D** | pondération du second ordre (α par voxel) | une attention plus concentrée, utile sur petites lésions |
| **Integrated Gradients** | intégration des gradients sur 24 pas depuis la baseline nulle | ce qui « pousse » le score |
| **Occlusion 3D** | masquage de blocs et impact sur le score | fidélité du score (contrôle, hors figure) |

La visualisation met en évidence les régions ayant contribué à la réponse du
modèle selon la méthode XAI considérée.
"""

CELL_XAI_MAIN = r"""from xai.methods import ensemble_gradcam, ensemble_integrated_gradients, gaussian_filter

xai_case = golden_case   # 10032 — cas d'or
prep = preprocess_images([xai_case.images["t2w"], xai_case.images["adc"], xai_case.images["hbv"]])
tensor = make_tensor(prep, DEVICE)
t2w_grid = tensor.detach().cpu().numpy()[0, 0]   # canal T2W (20,256,256) de la grille modèle

cam_g = np.clip(gaussian_filter(ensemble_gradcam(models, tensor, DEVICE, method="gradcam"), 1.5), 0, None)
cam_p = np.clip(gaussian_filter(ensemble_gradcam(models, tensor, DEVICE, method="gradcampp"), 1.5), 0, None)

def xai_pair(cam, out_name: str, method_label: str) -> Path:
    zi = int(np.argmax(cam.max(axis=(1, 2))))
    sl = t2w_grid[zi]
    cam_n = (cam[zi] - cam[zi].min()) / (np.ptp(cam[zi]) + 1e-12)
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.8), dpi=150)
    ax[0].imshow(sl, cmap="gray"); ax[0].axis("off")
    ax[0].set_title("IRM T2W (grille modèle) — même coupe", fontsize=11)
    ax[1].imshow(sl, cmap="gray")
    ax[1].imshow(cam_n, cmap="inferno", alpha=0.58, vmin=0, vmax=1)
    ax[1].contour(cam_n, levels=[0.5], colors=["white"], linewidths=0.8, alpha=0.8)
    ax[1].axis("off")
    ax[1].set_title(f"{method_label} — attention (coupe z={zi})", fontsize=11)
    out = ASSETS / "xai" / out_name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out

fig_g = xai_pair(cam_g, "p2_gradcam_10032.png", "Grad-CAM")
fig_p = xai_pair(cam_p, "p2_gradcampp_10032.png", "Grad-CAM++")
det_grid = golden_result.raw_softmax_heatmap
z_det = int(np.argmax(det_grid.max(axis=(1, 2)))) if det_grid is not None else None
print(f"coupe de score max (detection, grille modele) : z={z_det}")
print(f"coupe d'attention max Grad-CAM     : z={int(np.argmax(cam_g.max(axis=(1, 2))))}")
print(f"coupe d'attention max Grad-CAM++   : z={int(np.argmax(cam_p.max(axis=(1, 2))))}")
DATA["xai_main"] = {
    "case_id": "10032_1000032",
    "gradcam": {"figure": rel(fig_g), "slice": int(np.argmax(cam_g.max(axis=(1, 2))))},
    "gradcampp": {"figure": rel(fig_p), "slice": int(np.argmax(cam_p.max(axis=(1, 2))))},
    "det_slice": z_det,
}
show(IPyImage(filename=str(fig_g)))
show(IPyImage(filename=str(fig_p)))
"""

CELL_XAI_IG = r"""cam_ig = np.clip(gaussian_filter(ensemble_integrated_gradients(models, tensor, DEVICE, steps=24), 1.5), 0, None)
fig_ig = xai_pair(cam_ig, "p2_ig_10032.png", "Integrated Gradients (24 pas)")
print(f"coupe d'attributions max IG         : z={int(np.argmax(cam_ig.max(axis=(1, 2))))}")
DATA["xai_ig"] = {"case_id": "10032_1000032", "figure": rel(fig_ig),
                  "slice": int(np.argmax(cam_ig.max(axis=(1, 2))))}
show(IPyImage(filename=str(fig_ig)))
"""

CELL_XAI_FN = r"""# Comment le modèle "voit" un cancer qu'il manque (10012, score 0.110 < 0.5)
fn_case = load_case("10012_1000012")
prep2 = preprocess_images([fn_case.images["t2w"], fn_case.images["adc"], fn_case.images["hbv"]])
tensor2 = make_tensor(prep2, DEVICE)
cam2 = np.clip(gaussian_filter(ensemble_gradcam(models, tensor2, DEVICE, method="gradcampp"), 1.5), 0, None)
fn_result = run_inference(fn_case, device=DEVICE, models=models, model_load_time_s=model_load_time_s)
fn_score = float(fn_result.case_level_likelihood)
print(f"case 10012 | score {fn_score:.3f} < 0.5 → aucune alarme (cancer manqué)")
print("La carte ci-dessous permet de vérifier si l'absence d'alarme correspond à")
print("une absence de réponse ciblée ou à une réponse d'amplitude insuffisante.")

def xai_pair_case(t2w_grid_, cam_, out_name: str, label: str) -> Path:
    zi = int(np.argmax(cam_.max(axis=(1, 2))))
    sl = t2w_grid_[zi]
    cam_n = (cam_[zi] - cam_[zi].min()) / (np.ptp(cam_[zi]) + 1e-12)
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.8), dpi=150)
    ax[0].imshow(sl, cmap="gray"); ax[0].axis("off")
    ax[0].set_title("IRM T2W — même coupe", fontsize=11)
    ax[1].imshow(sl, cmap="gray")
    ax[1].imshow(cam_n, cmap="inferno", alpha=0.58, vmin=0, vmax=1)
    ax[1].contour(cam_n, levels=[0.5], colors=["white"], linewidths=0.8, alpha=0.8)
    ax[1].axis("off")
    ax[1].set_title(f"{label}", fontsize=11)
    out = ASSETS / "xai" / out_name
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return out

fig_fn = xai_pair_case(tensor2.detach().cpu().numpy()[0, 0], cam2, "p2_gradcampp_10012_FN.png",
                       f"Grad-CAM++ — cancer manqué (score {fn_score:.2f})")
DATA["xai_fn"] = {"case_id": "10012_1000012", "figure": rel(fig_fn), "score": fn_score,
                  "slice": int(np.argmax(cam2.max(axis=(1, 2))))}
show(IPyImage(filename=str(fig_fn)))
"""

CELL_XAI_QUALITY = r"""import csv
quality_rows = []
with open(ROOT / "experiments" / "results" / "xai" / "xai_quality.csv", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        quality_rows.append(row)

def agg_float(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "None", "nan")]
    return float(sum(vals) / len(vals)) if vals else None

gc_sum = [r for r in quality_rows if r["method"] == "gradcam_summary"]
quality_summary = {
    "insertion_auc_mean": agg_float(gc_sum, "insertion_auc"),
    "deletion_auc_mean": agg_float(gc_sum, "deletion_auc"),
    "stability_best": max((float(r["stability_shift_y_pearson"]) for r in gc_sum
                          if r.get("stability_shift_y_pearson") not in (None, "", "None", "nan")),
                          default=None),
    "randomization_mean_abs": agg_float(gc_sum, "randomization_pearson"),
    "gradcampp_enrichment_max": max((float(r["enrichment"]) for r in quality_rows
                                     if r["method"] == "gradcampp" and r.get("enrichment") not in (None, "", "None", "nan")),
                                    default=None),
    "n_audited_cases": len({r["case_id"] for r in quality_rows}),
}
qc_df = pd.DataFrame([quality_summary])
print("Contrôles qualité XAI — agrégats du journal du pipeline (ÉPISODE audit 6 cas :")
print("quartiles de score + FP + FN) :")
print(qc_df.to_string(index=False))
DATA["xai_quality"] = quality_summary
DATA["xai_quality_source"] = "experiments/results/xai/xai_quality.csv — audit réel (6 cas incl. FP/FN)"
"""

CELL_EXPORT = r"""export_path = DATA_OUT_DIR / "presentation_data.json"
export_path.write_text(json.dumps(DATA, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("export ->", export_path)
print("clés    :", ", ".join(DATA.keys()))
"""

CELL_CONCL_MD = r"""# Conclusion et perspectives

### Ce que ce notebook a établi

- **Phase 1** — le pipeline officiel est reproduit fidèlement : sur le cas d'or,
  l'écart au score de référence est de quelques millionièmes ; l'intégration
  (prétraitement, ensemble, TTA, fusion, post-traitement) est validée, et les
  7 cas de démonstration montrent le tube complet données → inférence → score → décision.
- **Phase 2** — les méthodes XAI réellement implémentées (Grad-CAM, Grad-CAM++,
  Integrated Gradients, occlusion) ont été appliquées au modèle et s'accompagnent
  d'un audit quantitatif du journal du pipeline (fidélité insertion/délétion,
  randomisation des poids, stabilité au bruit, densité d'attention). L'évaluation
  élargie (240 cas, val/test sans fuite) et la recalibration du seuil sur la
  validation seule complètent l'instrumentation expérimentale.

### Limites honnêtes

- les cas locaux ont été vus à l'entraînement du modèle officiel : les métriques
  sont une borne supérieure, la validation externe reste à faire ;
- l'informativité des cartes d'attention décroît fortement sous un score ≈ 0.15 ;
- l'audit XAI ne couvre que 6 cas représentatifs, à étendre systématiquement.

### Prochaines étapes

1. comparaison systématique des méthodes XAI sur tout le test
   (matrice méthode × cas, métriques de fidélité agrégées) ;
2. intégration des cartes d'attention comme interface de lecture pour radiologiste ;
3. validation externe sur données détenues (jeu test public held-out) ;
4. calibration du point opératoire (échelle de température), en conservant la logique
   figée sur validation ;
5. extension des contrôles qualité de l'XAI (robustesse inter-erreurs, incertitudes).
"""


def _nb_of_cells():
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    plan = [
        ("markdown", CELL_INTRO),
        ("code", CELL_PATHS),
        ("markdown", CELL_EQUIP),
        ("code", CELL_DATA_LOAD),
        ("code", CELL_MODEL_LOAD),
        ("markdown", CELL_P1_MD),
        ("code", CELL_GOLDEN),
        ("markdown", CELL_EXAMP_MD),
        ("code", CELL_RUN_CASES),
        ("code", CELL_PANELS),
        ("markdown", CELL_THR_MD),
        ("code", CELL_THR),
        ("markdown", CELL_P2_TRANSITION),
        ("code", CELL_BENCH),
        ("markdown", CELL_XAI_MD),
        ("code", CELL_XAI_MAIN),
        ("code", CELL_XAI_IG),
        ("code", CELL_XAI_FN),
        ("code", CELL_XAI_QUALITY),
        ("code", CELL_EXPORT),
        ("markdown", CELL_CONCL_MD),
    ]
    for kind, src in plan:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "markdown"
                        else nbf.v4.new_code_cell(src))
    return nb


def build_notebook() -> None:
    import nbformat as nbf
    nbf.write(_nb_of_cells(), str(NB_PATH))
    print(f"[notebook] structure écrite -> {NB_PATH}")


def execute_notebook() -> None:
    import nbformat as nbf
    from nbclient import NotebookClient
    nb = nbf.read(str(NB_PATH), as_version=4)
    client = NotebookClient(nb, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(nb, str(NB_PATH))
    print(f"[notebook] exécution réussie -> {NB_PATH} (sorties réelles incluse)")


def export_html() -> None:
    import nbformat as nbf
    from nbconvert.exporters import HTMLExporter
    exp = HTMLExporter(embed_images=True)
    nb = nbf.read(str(NB_PATH), as_version=4)
    body, resources = exp.from_notebook_node(nb)
    HTML_PATH.write_text(body, encoding="utf-8")
    print(f"[notebook] export HTML -> {HTML_PATH} ({HTML_PATH.stat().st_size // 1024} Ko)")


def deploy_to_site() -> None:
    SITE_NB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HTML_PATH, SITE_NB_DIR / HTML_PATH.name)
    shutil.copy2(NB_PATH, SITE_NB_DIR / NB_PATH.name)
    data_file = ROOT / "reports" / "demo" / "data" / "presentation_data.json"
    n_fig = len(list((ROOT / "reports" / "demo" / "assets").glob("**/*.png")))
    ok = data_file.exists()
    print(f"[site] notebook+html exposés -> {SITE_NB_DIR}")
    print(f"[site] presentation_data.json : {'OK' if ok else 'MANQUANT'} | {n_fig} figures png")


if __name__ == "__main__":
    build_notebook()
    execute_notebook()
    export_html()
    deploy_to_site()
    print("[presentation_build] DONE")
