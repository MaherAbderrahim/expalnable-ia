# PROGRESS_REPORT.md — rapport d'avancement du sous-projet

Sous-projet : `prostate_model_demo/` — Phase 1 du projet de recherche
**« XAI pour la détection du cancer de la prostate sur IRM »**
(date du rapport : 2026-10-22)

---

## ÉTAPE 1 : Choisir le modèle ✅

- Comparaison de 4 modèles candidats (voir le rapport de recherche
  parent) ; choix retenu : **U-Net de référence PI-CAI, ensemble 5-fold**.
- Modèle récupéré : https://github.com/DIAGNijmegen/picai_unet_gc_algorithm
  - Cloné avec LFS dans `third_party/picai_unet_gc_algorithm/`.
  - `git lfs pull` exécuté : les fichiers sont de vraies données, pas
    des pointeurs LFS (vérifié par les tailles de fichiers réelles, et
    re-vérifié par les tests pytest).
- Poids téléchargés :
  - `third_party/picai_unet_gc_algorithm/weights/unet_F0.pt` …
    `unet_F4.pt` (127,229,381 octets chacun — vérifié).
  - Test de smoke test réussi : scripts `process.py` officiel exécuté
    dans le dossier cloné → likelihood 0.5269265174865723 = valeur de
    référence fournie (avec torch 2.11 cu128 sur cette machine).

## ÉTAPE 2 : Créer la structure de sous-projet ✅

`prostate_model_demo/` :
```
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── docs/
│   ├── MODEL_EXPLANATION.md        ✅ écrit avant le code (fr)
│   ├── DATA_EXPLANATION.md         ✅ écrit avant le code… et revu (fr)
│   ├── EXPERIMENT_RESULTS.md       ✅ chiffres réels uniquement (fr)
│   ├── PROGRESS_REPORT.md          ✅ (ce fichier)
│   └── WHAT_I_NEED_TO_UNDERSTAND.md ✅ (fr)
├── src/
│   ├── __init__.py
│   ├── config.py           ✅ chemins + constantes officielles
│   ├── data_loader.py      ✅ chargement .mha/.nii.gz + GT + annotations
│   ├── preprocessing.py    ✅ pipeline officiel répliqué
│   ├── inference.py        ✅ ensemble 5-fold + TTA + post-traitement
│   ├── evaluation.py       ✅ métriques niveau cas / IoU lésion
│   ├── visualization.py    ✅ figures 2 lignes x 6 panneaux
│   └── unet_arch.py        ✅ UNet officiel vendu (patch MONAI 1.6 documenté)
├── scripts/
│   ├── download_or_prepare_data.py ✅ Zenodo partiel + picai_labels
│   ├── run_demo.py          ✅ pipeline CLI complet + golden test
│   └── streamlit_app.py     ✅ mini-UI (testée ci-dessous)
├── tests/
│   └── (pytest: conftest, 4 fichiers, 13 tests) ✅ tous passés
├── data/
│   ├── README.md
│   ├── demo_cases.json
│   ├── demo_cases/<patient>/<case_id>_{t2w,adc,hbv}.mha
│   └── picai_labels/{clinical_information/marksheet.csv,
│                     csPCa_lesion_delineations/human_expert/…}
├── outputs/
│   ├── summary.json
│   ├── figures/<case_id>.png   (7 figures)
│   └── predictions/<case_id>/{cspca_detection_map.mha, prediction.json}
├── notebooks/
│   └── demo.ipynb       ✅ notebook « pipeline en 10 cellules »
├── third_party/picai_unet_gc_algorithm/ (appartient au modèle, ne pas modifier)
└── .venv/               (environnement local)
```

## ÉTAPE 3 : Inférence sur un seul cas ✅

- `scripts/run_demo.py` : implémente les étapes 1 à 7 du pipeline
  officiel :
  1. charger le cas (t2w+adc+hbv) → 2. aligner + crop/pad + resample →
  3. normaliser z-score → 4. tenseur → 5. ensemble 5-fold + TTA →
  6. carte de détection + likelihood → 7. sauvegarder figure + json + mha.
- Premier cas exécuté : `10032_1000032` (cas de test officiel).

## ÉTAPE 4 : Télécharger les données ✅

- `scripts/download_or_prepare_data.py` (télécharge partiellement via
  remotezip les fichiers de cas individuels des 5 archives Zenodo via
  HTTP Range ; pas besoin des 27 Go complets).
- Marquesmark de PI-CAI + annotations de lésions d'experts téléchargées
  (`resampled/` ou variantes binaires `Pooch25/` selon le cas).
- 7 cas acquis : 3 négatifs + 4 positifs (ISUP 3,3,3,2 ; PI-RADS 4-5)
  → voir `data/demo_cases.json`.

## ÉTAPE 5 : Prédictions CLI pour tous les cas ✅

- Exécution complète `run_demo.py` : 7 cas × (chargement + prétraitement
  + ensemble + post-traitement + figures) — **0 erreur**.
- Likelihoods cohérentes : 10032 ≈ 52.7 %, négatifs 2–20 % (voir
  EXPERIMENT_RESULTS.md pour la liste des 7 cas).
- Sorties vérifiées : figures lisibles, carte de détection enregistrée
  en .mha ré-ouvrable, likelihoods JSON dans [0,1].
- Golden test : PASSED (voir EXPERIMENT_RESULTS.md).
- Exécution sur GPU RTX 5060 Ti : ~0.52 s par cas (0.48–0.62 s),
  VRAM ≈ 2–3 Go.

## ÉTAPE 6 : Visualisation ✅

- La figure de chaque cas contient : inputs natifs T2W / ADC / HBV +
  heatmap brute ; carte de détection + superposition du GT ; heatmap
  brute à la coupe max ; contour expert (lime) vs contour de la carte
  de détection (cyan) à la coupe de la lésion de GT max.
- Régénération effectuée après chaque correction.

## ÉTAPE 7 : Tests + métriques ✅

- `pytest tests` = **13 passés** en ~11 s.
- Comparaison de la carte de détection avec le contour de l'expert pour
  les 7 cas avec annotation (3 négatifs = masque vide, moyenne IoU 0.19).
- Disclaimers : le run n'est PAS une évaluation clinique.

## ÉTAPE 8 : Application Streamlit ✅

- `scripts/streamlit_app.py` : lance
  `streamlit run scripts/streamlit_app.py`.
- Test headless : démarrage du serveur sans erreur, page d'accueil
  répond, la liste des cas s'affiche, bouton « Prédire » expose la
  probabilité csPCa sans aucune fonctionnalité XAI (phase 1 = modèle
  seul).

## ÉTAPE 9 : Documentation ✅ (le .md actuel + README)

- README (framework, métriques, analyse, scénario oral de 2-3 min).
- docs/MODEL_EXPLANATION / DATA_EXPLANATION / EXPERIMENT_RESULTS /
  PROGRESS_REPORT / WHAT_I_NEED_TO_UNDERSTAND.

## ÉTAPE 10 [FUTUR] : XAI (hors périmètre de ce repo)

- La démo actuelle exécute le modèle PI-CAI baseline SEUL et fournit
  les points d'ancrage (heatmap brute, carte de détection, annotations
  experts) sur lesquels la phase 2 viendra expliquer.
- Prochaine étape (phase 2) : Grad-CAM / Integrated Gradients /
  occlusion par-dessus ces sorties, comparaison aux contours d'experts.

---

## Ce qui a été créé/modifié (aperçu)

Fichiers créés (tout est nouveau ; aucun fichier de recherche parent
modifié) :

```
prostate_model_demo/
├── README.md, requirements.txt, pytest.ini, .gitignore
├── data/README.md, data/demo_cases.json
├── data/demo_cases/<patient>/<case_id>_{t2w,adc,hbv}.mha  (7 cas)
├── data/picai_labels/…                               (marksheet + annotations)
├── scripts/{download_or_prepare_data.py, run_demo.py, streamlit_app.py}
├── src/{config.py, data_loader.py, preprocessing.py, unet_arch.py,
│        inference.py, evaluation.py, visualization.py, __init__.py}
├── tests/{conftest.py, test_data_loader.py, test_preprocessing.py,
│          test_model.py, test_inference.py}
├── notebooks/demo.ipynb
├── outputs/{summary.json, figures/*.png, predictions/…/…}
└── docs/{5 fichiers markdown}
```

Fichiers modifiés : aucun en dehors de `prostate_model_demo/`.
Le `third_party/picai_unet_gc_algorithm` n'a pas été modifié.

## Prochaines étapes de vérification critiques (à refaire en cas de doute)

1. `git lfs pull` dans `third_party/picai_unet_gc_algorithm` si les
   fichiers affichent ~130 octets (pointeurs LFS) → les checkpoints
   doivent faire 127,229,381 octets réels.
2. `pytest tests` → 13 passés.
3. `python scripts/run_demo.py` → likelihoods cohérentes : 10032
   ≈ 52.7 %, négatifs ≈ 2-20 %.
4. Golden test inclus dans `run_demo.py` doit rester **PASSED**.
5. `streamlit run scripts/streamlit_app.py` → s'ouvre dans le navigateur.

## Prochaines étapes futures (phase 2 : XAI)

- Brancher Grad-CAM / Integrated Gradients / occlusion sur la même
  entrée prétraitée et la même architecture ;
- comparer explications vs (a) carte de détection, (b) contours
  d'experts, (c) vérité terrain ;
- densifier le dataset d'évaluation (plus de cas, éventuellement
  tous les 1500 d'un fold du Zenodo).
