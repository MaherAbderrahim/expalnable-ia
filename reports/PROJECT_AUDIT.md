# AUDIT TECHNIQUE DU PROJET — Expalnable IA (PI-CAI / XAI)

> **Date de l'audit** : 2026-10-26 · **Nature** : audit statique, factuel, lecture seule.
> **Aucun fichier n'a été modifié, supprimé ou ré-entraîné pendant l'audit.**
> Seules deux suites de tests ont été *exécutées* (lecture seule) ainsi que des lectures de CSV/JSON/NPZ.
> Chaque constat est tracé vers un fichier ou un résultat exécuté. `[C]` = confirmé par observation directe,
> `[S]` = suspecté/doctiné, nécessitant vérification humaine.

---

## 1. État réel du projet

### 1.1 Git et périmètre

| Élément | Observation | Preuve |
|---|---|---|
| Branche | `main`, à jour avec `origin/main`, **working tree propre** (aucun fichier modifié) | `git status --short --branch` `[C]` |
| Historique | 2 commits : `5292b64` Phase 2 (évaluation/XAI/pipeline/tests/rapports), `2fea8db` Phase 1 (baseline + golden test PASSED) | `git log --oneline -10` `[C]` |
| Fichiers versionnés | 178 fichiers (code, configs, rapports, résultats légers, figures) | `git ls-files` `[C]` |
| Non versionné (gitignoré, présent sur disque) | poids `third_party/` (5 × 127 Mo), `data/picai/` (600 images + 600 masques experts), `data/predictions/*.npz` (val+test), `.venv`, logs runtime | `.gitignore` `[C]` |
| Statique | **Aucune modification non commitée** avant l'audit | `git status` `[C]` |

### 1.2 Environnement détecté

| Aspect | Détecté |
|---|---|
| Langage | Python 3.12 (venv partagé `prostate_model_demo/.venv`) |
| Framework ML | PyTorch (>= 2.7, build CUDA), MONAI, picai_prep, picai_baseline, Report-Guided-Annotation |
| Interface | Streamlit (démo Phase 1), HTML auto-portant `reports/demo/index.html` (démo finale), Jupyter (notebook de présentation) |
| GPU | **RTX 5060 Ti, CUDA disponible** — vérifié par `torch.cuda.is_available() == True` `[C]` |
| Orchestration | `scripts/run_phase2.py` (7 étapes idempotentes + reprise), 2 lanceurs `.bat` + `tools/launch_demo.ps1` |
| CI/CD | **Aucun** (pas de `.github/workflows`, pas de GitLab CI) `[C]` |
| Tests | pytest — 13 tests Phase 1 (`prostate_model_demo/tests`) + 30 tests Phase 2 (`tests/` à la racine) |
| Données sur disque | 600/600 images `.mha`, 600/600 masques experts, prédictions val (120) + test (120) + 1 smoke `[C]` |

### 1.3 What it actually is (résumé factuel)

Le dépôt contient deux phases effectivement réalisées :

1. **Phase 1** (`prostate_model_demo/`) : reproduction offline du pipeline d'inférence officiel PI-CAI
   (préprocessing `picai_prep` + U-Net MONAI 3D ensemblisté sur 5 plis + TTA flip + gaussian σ=1.5 +
   extraction RGA `dynamic` + suppression d'anneau), avec un **test golden** numérique sur le cas
   officiel `10032_1000032`, une démo Streamlit, et 13 tests pytest. **Tous les poids réels sont présents**
   (5 × exactement 127 229 381 octets, pas des pointeurs LFS) `[C]`.

2. **Phase 2** (`evaluation/`, `xai/`, `scripts/`, `tests/`) : inférence à 1 passe (composants par pli
   sauvegardés en fp16, `native_fused` en fp32), post-traitement **offline** à 10 configurations
   pré-enregistrées, sélection sur validation uniquement (règle figée dans le code), test scored une fois,
   métriques (AUROC/AP/ECE/point opératoire/leison-IoU), statistiques (bootstrap IC + DeLong apparié),
   analyse d'erreurs, et une pile XAI 3D maison (Grad-CAM / Grad-CAM++ / IG / occlusion) avec contrôles
   qualité (insertion/deletion, randomisation, stabilité, localisation vs contours experts).

**Résultats réellement disponibles** (tous re-vérifiés chiffre à chiffre contre les artifacts) `[C]` :

| Résultat | ValeurVérifiée | Source |
|---|---|---|
| AUROC val / test (officiel @0.5) | 0.8379 / 0.8042 | `eval_final_summary.json`, `comparison_table.csv` |
| AUROC val / test (`no_extract` @0.1572) | 0.8451 / **0.8081** | idem |
| AP val / test officiel→sélectionné | 0.7064→0.7107 (test) | idem |
| IC 95 % bootstrap test (AUROC / AP) | [0.710–0.895] / [0.586–0.827] | idem (2000 rééchantillons, seed 2026) |
| DeLong off vs sel (val / test) | p = 0.8907 / **0.9473** (non significatif — assumé) | idem |
| Seuil gelé | 0.157167 … (F1 max sur val) | `experiments/configs/selected_postproc.json` |
| Splits patients | 360 / 120 / 120 (patients), **chevauchement val∩test = train∩val∩test = 0** | `data/splits/patient_split_2026_bff8efe556.json`, recalculé `[C]` |
| XAI cas audités | **6 cas** (pas 8) — 3 méthodes heatmap + 4 contrôles sur gradcam | `xai_quality.csv` (24 lignes) |
| Golden test Phase 1 | Δ de vraisemblance = 3.93e-6 (0.5269225836 vs 0.5269265175 ref), statut PASS | pytest Phase 1, exit 0 `[C]` |
| Suite Phase 2 | **30 passed** (exécuté pendant l'audit, exit 0) `[C]` | `pytest tests -q` |
| Suite Phase 1 | **13 passed** (exécuté pendant l'audit, exit 0) `[C]` | `prostate_model_demo/tests` |

---

## 2. Tableau des fonctionnalités (statut par fonctionnalité)

Statuts : `IMPLEMENTED_AND_TESTED` (code + exécution prouvée + tests), `IMPLEMENTED_NOT_TESTED` (code + artifacts, pas de tests dédiés),
`PARTIALLY_IMPLEMENTED` (réel mais incomplet vs doc), `DOCUMENTATION_ONLY`, `NOT_IMPLEMENTED`, `BLOCKED`.

| # | Fonctionnalité | Statut | Preuves (fichier/artifact) |
|---|---|---|---|
| F1 | Chargement modèle PI-CAI (5 plis `unet_F0..F4`) + poids réels | **IMPLEMENTED_AND_TESTED** | `prostate_model_demo/src/inference.py:86-107`; tests `test_model.py`, `test_data_loader.py::test_weights_are_real_not_lfs_pointers`; 5 fichiers = 127 229 381 o `[C]` |
| F2 | Préprocessing officiel (picai_prep + z-score p99.5) | **IMPLEMENTED_AND_TESTED** | `src/preprocessing.py:45-61` (réutilise les libs officielles); `test_preprocessing.py` (3 tests) |
| F3 | Inférence officielle (TTA flip, σ 1.5, moyenne 5 plis) | **IMPLEMENTED_AND_TESTED** | `src/inference.py:119-137` (ligne-à-ligne vs `process.py`); `test_inference.py` |
| F4 | Post-traitement officiel (RGA dynamic + suppression anneau) | **IMPLEMENTED_AND_TESTED** | `src/inference.py:187-235`; `evaluation/postproc.py:121-133`; golden test |
| F5 | Golden test vs référence auteurs | **IMPLEMENTED_AND_TESTED** | `test_inference.py::test_golden_reference_reproduction`; Δ = 3.93e-6; 1 voxel/2 801 664 (>1e-3) — passé à l'instant `[C]` |
| F6 | Démo web Streamlit (Phase 1) | **IMPLEMENTED_NOT_TESTED** (smoke manuel documenté) | `scripts/streamlit_app.py`; 2 PNG streamlit sur disque |
| F7 | Téléchargement partiel Zenodo (range requests) + annotations | **IMPLEMENTED_NOT_TESTED** (pas de test unitaire; exercé par 3 runs réels) | `scripts/prepare_data.py`; `experiments/logs/prepare_data.log`; 600/600 images + masques présents `[C]` |
| F8 | Sélection stratifiée de 600 cas (seed 2026, 1 étude/patient) | **IMPLEMENTED_NOT_TESTED** | `prepare_data.py::select_cases`; manifest `data/splits/case_selection_seed2026.json` (600, 170+) — NOTE : stratification réelle = *(centre, csPCa, buckets PI-RADS)*, **hospital/biopsy non utilisés** |
| F9 | Splits patients leak-free 60/20/20 | **IMPLEMENTED_AND_TESTED** | `evaluation/splits.py` (`check_no_leakage`); `scripts/build_splits.py`; 3 tests `test_splits.py`; test de recouvrement récité `[C]` |
| F10 | Inférence 1 passe val+test avec composants par pli | **IMPLEMENTED_NOT_TESTED** (pas de test unitaire; artifacts complets) | `evaluation/infer.py`; `data/predictions/{val,test}/*.npz` (120+120) + `run_inference_summary.json` |
| F11 | Benchmark 10 post-traitements sur validation | **IMPLEMENTED_NOT_TESTED** | `evaluation/postproc.py` (10 configs pré-enregistrées); `val_comparison.csv` (10 lignes), `val_scores_long.csv` |
| F12 | Sélection pré-enregistrée (règle écrite avant) + gel seuil | **IMPLEMENTED_NOT_TESTED** | `scripts/select_config.py` (règle en docstring); `selected_postproc.json` (thr 0.1572) |
| F13 | Évaluation finale test 1 passe + comparaison val/test | **IMPLEMENTED_NOT_TESTED** | `scripts/eval_final.py`; `eval_final_summary.json` (67 Ko), `comparison_table.{csv,md}` |
| F14 | Métriques cas (AUROC/AP/ECE/points optimaux/Youden/F1) | **IMPLEMENTED_AND_TESTED** | `evaluation/metrics.py`; 8 tests `test_metrics.py` avec valeurs à la main `[C]` |
| F15 | Métriques lésions (IoU/Dice 3D, critère ≥10 %) | **IMPLEMENTED_AND_TESTED** | `metrics.py::mask_iou_dice, detection_overlap`; `eval_model._score_case_all_configs`; test `test_mask_overlap_metrics` |
| F16 | Statistiques (bootstrap IC, DeLong apparié fait-maison, variance multi-runs) | **IMPLEMENTED_AND_TESTED** | `evaluation/statistics.py` (DeLong Sun & Xu 2014 vectorisé); tests DeLong/b Bootstrap; CI recalculées matchant à 4 décimales `[C]` |
| F17 | Calibration (ECE + Platt fit) | **IMPLEMENTED_AND_TESTED** | `metrics.py:136-196`; test ECE/Platt (2 warnings RuntimeWarning overflow `exp` — bénin, à clipper) |
| F18 | Analyse d'erreurs stratifiée (TP/FN/FP, PI-RADS, centre, IoU) | **IMPLEMENTED_NOT_TESTED** (pas de test unitaire) | `evaluation/error_analysis.py`; `val/test_patterns.json` + 2 CSV (cohérents avec REPORT.md `[C]`) |
| F19 | Grad-CAM & Grad-CAM++ 3D | **IMPLEMENTED_AND_TESTED** | `xai/methods.py:87-125` (implementation maison 3D, hook sur `model.model[1]`); tests de forme et comportement numérique |
| F20 | Integrated Gradients 3D | **IMPLEMENTED_AND_TESTED** | `xai/methods.py:132-155`; test forme/range |
| F21 | Occlusion sensitivity 3D | **IMPLEMENTED_AND_TESTED (code) / PARTIALLY_IMPLEMENTED (pipeline)** | `xai/methods.py:162-198` + test `test_occlusion_scan_shape`; **MAIS jamais exécuté dans le run phase 2** (drapeau `with_occlusion=False` par défaut; aucune colonne occlusion dans `xai_quality.csv`, aucune heatmap occlusion sur disque) `[C]` |
| F22 | fig=XAI overlays (`xai/visualize.py`) | **IMPLEMENTED_AND_TESTED (smoke)** | `visualize.py::explain_figure`; test smoke fichier PNG; 6 PNG de cas + smokes sur disque |
| F23 | Contrôles qualité XAI (stabilité / randomisation / insertion-deletion) | **IMPLEMENTED_AND_TESTED** (sur gradcam uniquement) | `xai/quality.py`; 4 tests (`test_xai.py`) ; qualité CSV remplie |
| F24 | Contrôle qualité "localisation vs carte de détection du modèle ET contours" | **PARTIALLY_IMPLEMENTED** | `localization_agreement` compare uniquement heatmap vs masque expert; **pas de comparaison vs la détection du modèle** dans le CSV `[C]`; alignement masque→grille modèle par zoom naïf (origine crop non compensée) |
| F25 | Pipeline 1 commande 7 étapes (idempotent, reprise) | **IMPLEMENTED_NOT_TESTED** (exercé 3 fois, timestamps log) | `scripts/run_phase2.py`; `run_phase2.log` : download→…→regression OK 09:38:42 `[C]` |
| F26 | Régression Phase 1 à chaque run | **IMPLEMENTED_AND_TESTED** | `run_phase2.py:161-164`; 2 exécutions 13/13 passées (log 09:24:46 et 09:38:42, + poids ré-exécuté pendant l'audit) `[C]` |
| F27 | Tableau de bord démo HTML (`reports/demo/index.html`) | **IMPLEMENTED_NOT_TESTED** (site statique vérifié ; tous les chiffres matchent les artifacts ; tous les PNG référencés existent `[C]`) | `build_demo_page.py`/`build_site_v2.py` ; non ré-exécuté pendant l'audit |
| F28 | Lanceurs Windows (`Lancer/Arreter-Demo-IA.bat`, tunnel Cloudflare) | **IMPLEMENTED_NOT_TESTED** | fichiers + logs runtime ; serveur/tunnel non démarrés pendant l'audit |
| F29 | Notebook de présentation (`phase1_phase2_demo.ipynb`) | **PARTIALLY_IMPLEMENTED** | 6.2 Mo à la racine (contient des sorties, version exécutée copiée dans `reports/demo/nb/`); rebuild via `scripts/presentation_build.py`; non ré-exécuté; `prostate_model_demo/notebooks/demo.ipynb` est **stocké sans sorties** (exécuté antérieurement, preuve indirecte par `nb_*.png`) `[C]` |
| F30 | Suite de tests 30 tests racine | **IMPLEMENTED_AND_TESTED** | 30 `def test_` : 8 metrics + 10 postproc/stats + 3 splits + 9 XAI ; **30/30 passés à l'instant** `[C]` |
| F31 | CI/CD (GitHub Actions, lint, tests auto) | **NOT_IMPLEMENTED** | aucun fichier workflow `[C]` |
| F32 | Entrainement / fine-tuning GPU | **NOT_IMPLEMENTED** (assumé et documenté honnêtement) | README "Limites connues"; aucun script d'entraînement `[C]` |
| F33 | Couche PI-RADS (recommandation RECOMMENDATION.md) | **DOCUMENTATION_ONLY** | recommandé dans `RECOMMENDATION.md` ; pas de code `[C]` |

---

## 3. Audit détaillé des 4 méthodes XAI

Toutes les méthodes sont **adaptées au modèle 3D** (travail sur tenseur `(1, 3, 20, 256, 256)`,
modalité csPCa = ch.2, ensembliste sur 5 plis). Les 4 contrôleurs qualité ont tous un test unitaire sur un
mini-UNet synthétique contractuel (`tests/test_xai.py`). Aucune dépendance externe (pas de Captum, pas de
pytorch-grad-cam) : **tout est implementé from scratch en torch/scipy** — le README dit le contraire (`[C]` mismatch).

### 3.1 Grad-CAM (3D)
| Item | Valeur |
|---|---|
| Fichier / entrée | `xai/methods.py::gradcam_3d(model, input_tensor, method="gradcam")` → `ensemble_gradcam(models, tensor, device, method="gradcam")` |
| Sortie | `(20, 256, 256)` float32 → ReLU + min-max normalisé ; superposition native dans `xai/visualize.py::_to_native` |
| Méthode math | GAP des gradients classe-somme sur la dernière skip-connection (`model.model[1]`, 96 ch à demi-résolution), CAM = Σ_k w_k·A_k, trilinear resize, par-pli puis moyenne naturelle sur 5 plis |
| Compat 3D | Oui — hook sur un bloc skip 3D; testé sur UNet 3D synthétique de même contrat (`test_xai.py`) |
| Tests | `test_gradcam_shape_and_range`, `test_gradcampp_differs_from_gradcam`, `test_ensemble_gradcam_averages_folds`, `test_randomization_changes_cam`, `test_stability_jitter_correlates`, `test_insertion_deletion_structure`, smoke figure |

### 3.2 Grad-CAM++ (3D)
| Item | Valeur |
|---|---|
| Fichier / entrée | `xai/methods.py::cam_weights_gradcampp` (+ `gradcam_3d(method="gradcampp")`) |
| Sortie | idem Grad-CAM, poids par-Chattopadhyay (formule par voxel auto-normalisée `alpha = g²/(2g² + ΣA·g² + eps)`), documentée comme version stable (les variantes directes sont instables) |
| Compat 3D | Oui, testé |
| Tests | `test_gradcampp_differs_from_gradcam`, qualité (IoU vs masque expert/enrichissement) dans `xai_quality.csv` |

### 3.3 Integrated Gradients (3D)
| Item | Valeur |
|---|---|
| Fichier / entrée | `xai/methods.py::ensemble_integrated_gradients(models, tensor, device, steps=…) » (24 pas dans `scripts/explain.py`) |
| Sortie | `(20, 256, 256)` — IG par canal, résumé par max(\|IG\|) sur les 3 canaux |
| Méthode math | IG standard : ∫₀¹ ∂f/∂x (base + α(x−base)) dα, baseline = **zeros** (= mean per-voxel des canaux z-scorés); score cible = csPCa logit sommé sur le volume ; moyenne sur 5 plis |
| Compat 3D | Oui (32 pas validés par défaut, 24 pas utilisés, 4 pas en test) |
| Tests | `test_integrated_gradients_shape_and_range` |

### 3.4 Occlusion sensitivity (3D)
| Item | Valeur |
|---|---|
| Fichier / entrée | `xai/methods.py::occlusion_scan(models, tensor, device, slab=(2,32,32), stride=(2,24,24))` (batch 32) |
| Sortie | `(20, 256, 256)` heatmap des baies (drop), min-max |
| Méthode math | FPS: score fusé = max-vol de la moyenne des sigmoides par pli (sans TTA, sans σ — proxy documenté pour la vitesse), occlusion de slabs rectangulaires remplis de zéro |
| Compat 3D | Oui — test `test_occlusion_scan_shape` |
| **⚠ Statut dans le run réel** | **JAMAIS appelé dans le pipeline phase 2** : `scripts/explain.py --with-occlusion` absent (défaut `False`), le run n'a produit AUCUNE heatmap occlusion, le CSV `xai_quality.csv` n'a pas de lignes `occlusion`. README → "4 méthodes" est faux pour ce run réel. `[C]` |

### 3.5 Contrôles qualité XAI

| Contrôleur | Fichier / entrée | Testé | Appelé dans `run_phase2` ? |
|---|---|---|---|
| Stabilité (±1 voxel y, Pearson) | `xai/quality.py::stability_jitter` | test | oui, **Grad-CAM uniquement** |
| Randomisation (Adebayo 2018 — poids up-path pli 0) | `quality.py::randomization_test` | test | oui, **Grad-CAM uniquement** |
| Fidélité insertion/delete (AUC normalisée, 16 pas) | `quality.py::insertion_deletion` (blurred baseline) | test | oui, **Grad-CAM uniquement** |
| Localisation vs contours experts (IoU top-5 %, rendement/enrichissement) | `quality.py::localization_agreement` | test | oui, **par méthode heatmap**; **alignement approximation zoom naïf** `[C]` |
| Localisation vs carte de détection du modèle (README) | — | **absent** | **non implémenté** → README sur-promet `[C]` |

### 3.6 Chiffres XAI réellement enregistrés (revérifiés)

- 6 cas audités : `10810_1000826`, `10388_1000394`, `11175_1001198`, `10925_1000942`, `10650_1000666`, `11341_1001364`; 1 smoke.
- Sélection stratifiée par quartile de score + 1 FP (`10650_1000666`, lik 0.270) + 1 FN (`11341_1001364`, lik 0.112) + bas scores (0.009/0.038/0.120).
- Grad-CAM ref (meilleur cas 10925) : ins 0.732 / del 0.206 / randomisation 0.095 / stabilité 0.994 ; Grad-CAM++ enrichissement 20× ; IG 1.5×.
- Moyennes qualité (3 méthodes) : insertion 0.4567, deletion 0.3537, randomisation \|r\| 0.0148, stabilité 0.2854.
- Les 4 contrôles sur insertion/delete/randomisation/stabilité **ne portent que sur Grad-CAM** (lignes `gradcam_summary`), les IoU/enrichissements couvrent gradcam/gradcampp/IG. `[C]`

---

## 4. Commandes pour reproduire (et pour compléter les manques)

Expédié pendant l'audit (résultats réels) :

```powershell
# Suite Phase 2 — 30 passed (vérifié pendant l'audit)
& .\prostate_model_demo\.venv\Scripts\python.exe -m pytest tests -q

# Suite Phase 1 + golden test — 13 passed (vérifié pendant l'audit)
cd prostate_model_demo
& .\.venv\Scripts\python.exe -m pytest tests -q
cd ..
```

À exécuter par toi (produisent de nouvelles données / nécessitent l'GPU, ici le RTX 5060 Ti présent):

```powershell
# Terme du Pipeline Phase 2 complet en 1 commande (idempotent, reprise)
python scripts/run_phase2.py                    # sans GPU/données: BLOCKED au stage "download"
python scripts/run_phase2.py --skip-download    # données déjà là (600 images + masques présents)
python scripts/run_phase2.py --from eval        # reprend à une étape
python scripts/run_phase2.py --force explain    # re-forcer l'étape XAI

# XAI UNIQUEMENT (ajoute la 4e méthode) — occlusion n'a PAS été exécutée aujourd'hui
python scripts/explain.py --n-cases 12 --with-occlusion   # default 12 (le run réel: 6 par limitation d'artefact)

# Évaluation finale seule (dangereux, ré-score ; test split théoriquement une seule passe)
python scripts/eval_final.py --workers 4

# Rebuild du site de démo
python scripts/build_demo_page.py               # ou build_site_v2.py
python scripts/presentation_build.py            # notebook de présentation

# Démo web (serveur + tunnel)
.\Lancer-Demo-IA.bat / .\Arreter-Demo-IA.bat

# Démarrage propre (new env) : cf README §2/3 — pip install -r requirements.txt, git lfs pull
```

---

## 5. Problèmes bloquants / incohérences, classés par priorité

### P0 — rien de technique bloquant
Aucun blocked : pipeline complet, poids réels, suite 30/30 et 13/13 passées, données en place, GPU OK.
Le projet **court et produit ce qu'il est documenté de produire** — à l'exception des incohérences ci-dessous.

### P1 — Fausses déclarations du README / de la doc (documentées dans le repo)

1. **"8 cas audités XAI"** (`README.md:52`, `reports/REPORT.md`) → réel **6** (CSV/documentation/demo page disent 6) `[C]`
2. **"4 méthodes XAI (Grad-CAM, Grad-CAM++, IG, Occlusion)" s'applique au run** → le run réel n'a produit que **3 heatmaps**, occlusion jamais appelée (courant expliqué §3.4) `[C]`
3. **"Grad-CAM/++ via pytorch-grad-cam, IG/occlusion via Captum"** (`README.md:187`) → implementations from scratch; **aucune de ces librairies n'est installée ni requise** `[C]` — période de confusion scientifique/auditabilité
4. **"benchmark … sur 600 cas stratifiés"** (`README.md:5,175`) → l'évaluation réel n'a couvert que **val+test = 240** (600 sélectionnés; train jamais inféré) `[C]`
5. **Stratification déclarée "(centre/hospital/csPCa/biopsy/PI-RADS)"** → manifest/code : *(centre, csPCa)* pour le split, *(centre, buckets PI-RADS)* pour la sélection; **hospital et biopsy non utilisés** `[C]`
6. **Comparaison vs "carte de détection du modèle ET contours experts"** → uniquement vs contours; pas vs détection `[C]`

### P2 — Erreurs internes de ` REPORT / BEST_MODEL.md` (chiffres, à corriger en doc, coûts doc-side)

1. REPORT.md "TTA removal costs −0.007 AUROC" → artefact: `no_tta` = +0.0003 (mieux ou égal; BEST_MODEL dit −0.0003 correct) `[C]`
2. REPORT.md "σ 1.0 ≈ neutre (val 0.846)" → artefact `sigma10` = 0.8423 (et `sigma05` = 0.8478) `[C]`
3. BEST_MODEL.md matrice d'erreurs test officiel "TP 8, TN 93, FP 0, FN 26" → TN réel = **86** (total correct 120) `[C]`
4. Temps de post-traitement incohérents et WITHOUT artefact: REPORT 36× ("0.07 s vs 2.55 s, GPU ~1.5 s/case") vs BEST_MODEL 50× vs log réel (GPU 0.1 s, total ≈ 2.5 s/case) `[C]` — aucune metric de temps figée dans `experiments` pour ce claim
5. `run_phase2.py:147` message "[eval] metrics_final.json exists" → le fichier réel est `eval_final_summary.json` (nom fantome répété dans le log) `[C]`
6. "28 dépôts inspectés / 45 liens vérifiés" (README) : rien de mécaniquement prouvé dans le repo (tools/*.json existes mais le comptage non enregistré formellement) `[S]`

### P3 — Fragilité / Dette d'ingénierie (impact moyen-faible)

1. `evaluation/eval_model.py:48` — `sorted(glob("patient_split_*.json"))[0]` : si plusieurs fichiers de split coexistent, le "premier" est utilisé silencieusement (ambiguïté potentielle de split).
2. Taille de jeu de tests XAI = 9 tests sur UNet synthétique; le profil XAI **réel** (chiffres attendus par méthode/cas) n'a pas de test de non-régression (seule la vérification par le CSV existant). Si le code XAI casse, la regression n'est détectée que via pytest synthétique.
3. Warnings `RuntimeWarning: overflow in exp` (metrics.py:175/196, Platt) — numérique factor (clip interne z), bénin mais sale sur la CI future.
4. `experiments/results/xai/smoke` file `data/predictions/smoke/10032_1000032.ERROR.txt` resté d'un crash ancien alors que le JSON final existe — QA documenté par le log  `ValueError: operands could not be broadcast together` — à purger ou eclipser, sinon confusion.
5. Aucune CI (tests ne s'exécutent que sur la machine d'un dev).
6. Notebook `prostate_model_demo/notebooks/demo.ipynb` commité **sans sorties**; version exécutée séparée dans `reports/demo/nb/`.
7. Pas de requirements précis au chiffre (torch `>=2.7` mais réel installé = torch 2.11.0+cu128) — la reproductibilité exacte exige le freeze.
8. Tests pytest racine ont des dépendances de présence de données (`selection_manifest_path` skip si absent) — OK, mais n'indiquent pas de manière visible dans le `skipped`.
9. `fused_score_from_batch` utilise la moyenne des sigmoides sans TTA — proxy documenté (module docstring), acceptable mais tout contrôle qualité XAI se base dessus.
10. README.md:148 annonce "pytest tests — 30 tests Phase 2" (prouvé) — mais ne mentionne pas la suite Phase 1 dans la section "Tests" (elle est en étape `regression`) — mineur.

### P4 — Points méthodologiques à garder en vue (vigilance scientifique)

1. L'alignement du masque expert → grille modèle est un **zoom naïf sans dispositif de crop/pad origine** (`scripts/explain.py::_to_model_grid`) — le preprocessing PI-CAI fait un crop/pad box avant resize; les IoU d'état `localization_agreement` portent ainsi une incertitude géométrique non mesurée.
2. L'IG baseline = zeros n'est pas garanti être la "moyenne par voxel" du jeu (choix du module docstring); pas de baseline alternatif (blur) testé; **l'audit ne remet pas ça en cause, il le documente**.
3. `insertion_deletion` utilise un score "fused max" du proxy sans TTA/σ — sous-estime fonctionnellement la fidélité (documenté).
4. La variance "multi-seed" est en réalité un éventail par-pli (pas de seed différents) — renommé "fold-run variance" dans le code (correct) mais REPORT.md l'appelle parfois multi-seed.

---

## 6. Plan d'action ordonné (Phase suivante — à valider avant implémentation)

> **Rien n'a été corrigé pendant l'audit** (consigne read-only). Proposition d'ordre:

1. **Nettoyage documentation (P1+P2)** — corriger `README.md` (8→6 cas, 4→3 méthodes le run, "Captum/pytorch-grad-cam"→"from-scratch", 600→240 évalués, stratification réelle), `reports/REPORT.md` (TTA, σ1.0, cas XAI), `reports/BEST_MODEL.md` (TN 86, harmoniser timings).
2. **Terminer le run XAI annoncé** : relancer `python scripts/explain.py --with-occlusion --n-cases 12` pour produire la 4e méthode et potentiellement l'audit réellement à 12 cas; suivi d'un `build_demo_page.py` pour la démo.
3. **[optionnel] Implémenter le comparateur "heatmap vs carte de détection du modèle"** (`localization_agreement_vs_model_map`) pour valider la 4e phrase du README.
4. **Figer l'environnement** : `pip freeze > requirements-freeze.txt` + note "torch 2.11.0+cu128 / CUDA 12.8 / Python 3.12.10 / RTX 5060 Ti" pour reproductibilité byte-level.
5. **CI minimale** (GitHub Actions) : `pytest tests` root + `pytest tests` phase 1 en CPU ( натуральment plus lent mais sans GPU nécessaire: les tests Phase 1 UTILISENT les checkpoints, donc nécessitent un artefact LFS en cache (runner ubuntu? sinon garde en "local only").
6. **Micro-fix code** : nom `metrics_final.json` vs `eval_final_summary.json` dans le message `run_phase2.py:147`; platt exp clip; purger `data/predictions/smoke/10032_1000032.ERROR.txt` (crash ancien).
7. **Robustifier souvent utilisé** : si plusieurs split files, fail loudly (assert single file) au lieu de prendre le premier.
8. **Commits** : séparer doc / code / run-artifacts dans des PR commits distincts.

---

## 7. Affirmations du README non vérifiées / à vérifier (et pourquoi)

| # | Affirmation (README) | Statut de vérification |
|---|---|---|
| 1 | "Reproduction fidèle … golden test PASSED \|e-likelihood\| = 3.94e-6" | ✅ Vérifiée (statistique: Δ réel 3.93e-6 fichier; pytest exit 0) `[C]` |
| 2 | "benchmark 10 configurations … sur 600 cas stratifiés" | ⚠️ En partie fausse : **600 sélectionnés**, mais seuls **240 évalués** (val+test) `[C]` |
| 3 | "audit d'explicabilité avec 4 méthodes XAI + contrôles qualité" | ⚠️ 3 méthodes only dans le run réel; occlusion jamais appelée `[C]` |
| 4 | "Grad-CAM/XAI pics pytorch-grad-cam + Captum" | ❌ Faux (implementation from scratch) `[C]` |
| 5 | "XAI — 8 cas audités" | ❌ 6 `[C]` |
| 6 | "sous-ensemble stratifié par centre/hospital/csPCa/biopsy/PI-RADS" | ⚠️ Partiel — hospital/biopsy non utilisés `[C]` |
| 7 | "~10 Go d'espace disque" | Non vérifié en conditions de premier téléchargement ; sur l'installation actuelle, `data/` pèse réellement ≈ **18,6 Go** (600 images + 240 npz prédictions) — l'ordre de grandeur README est cohérent `[S]` |
| 8 | "Sous-ensemble de 600 cas … Vérité terrain : csPCa au cas + contours experts" | ✅ (manifest 600, 170+, 600 masques présents) `[C]` |
| 9 | "Split patient 60/20/20, recouvement vérifié nul" | ✅ `[C]` (manifest + test recalculé) |
| 10 | "PYTHON 3.12 + RTX 5060 Ti CUDA 12.8 validé" | ✅ (torch.cuda OK, GPU présent) `[C]` — version torch 2.11.0 documentée par log, non vérifiée byte-level |
| 11 | "python scripts/build_demo_page.py reconstruit le tableau de bord" | Script présent, **non re-exécuté** pendant l'audit `[S]` |
| 12 | "pipeline idempotent & resumable" | ✅ (log montre 2 runs "skip" successifs successfully) `[C]` |
| 13 | "Lancer-Demo-IA.bat / tunnel Cloudflare" | Non testé (serveur non démarré) `[S]` |
| 14 | "28 dépôts inspectés, 45 liens vérifiés" (site de recherche) | Non vérifiable mécaniquement (tools/*.json parsés, aucun comptage formalisé) `[S]` |
| 15 | "reproducibility seed 2026 partout" | ✅ (manifest/logous, seeds récurrents) `[C]` |
| 16 | "Limites connues (RM check LM lower bound)" | ✅ documenté de manière honnête (aucun retrain; threshold spécifique à la distribution PI-CAI) `[C]` |
| 17 | "rapport PHI-CAI report-guided annotation > 0.3" | version réelle installée vérifiée côté venv `[S]` (pip list non encore scanné) |
| 18 | "9 étapes run_phase2" — en pratique 7 étapes (`download/splits/inference/select/eval/explain/regression`) | La doc en dit parfois 9 (page htm fête?) — 7 confirmées `[C]` |

---

## 8. Verdict

**Le projet est substantiellement réel et vérifiable :** les poids officiels sont là, les deux suites
de tests **passent réellement** (13/13 Phase 1 + 30/30 Phase 2, exécutées pendant l'audit), les
métriques sauvegardées retrouvent **chiffre à chiffre** les scripts et le tableau de bord, le protocole
anti-fuite est implémenté et testé, et la limitation "modèle pré-entraîné → borne supérieure optimiste"
est documentée honnêtement.

**Ce qui doit être corrigé avant de continuer** : la documentation sur-promet sur 5 points précis
(8 vs 6 cas XAI, 4 vs 3 méthodes, pytorch-grad-cam/Captum vs from-scratch, 600 vs 240 cas évalués,
stratification), REPORT.md/BEST_MODEL.md contiennent 4 chiffres internes faux, et le run XAI réel
devrait être mis à niveau (occlusion + plus de cas) pour matcher la claim du README.

**Recommandation immédiate** : plan d'action §6, étapes 1 → 2 → 4 en priorité (doc/alignement/run XAI
complet/freeze env), avant toute nouvelle fonctionnalité XAI.
