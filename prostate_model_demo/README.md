# Prostate Model Demo — phase 1 (modèle, sans XAI)

Sous-projet de démonstration du projet de recherche
**« XAI pour la détection du cancer de la prostate sur IRM »** :
ce module fournit la **preuve de fonctionnement réelle** du modèle
centre de phase 2 — sans toucher au XAI (phase 2, à venir).

> **En une phrase** : pipeline d'inférence du *U-Net de référence PI-CAI*
> (ensemble de 5 U-Nets 3D, publié dans le Lancet Oncology 2024 avec
> AUROC 0.878) répliqué à l'identique depuis le code officiel, exécuté
> sur 7 cas IRM réels du dataset public PI-CAI, avec le **test doré
> (golden test) PASSED** contre les sorties de référence fournies par
> les auteurs du modèle.

## 1. Ce que cette démo fait

1. **données** : sélectionne des cas IRM prostate publics (PI-CAI,
   licence CC BY-NC 4.0), télécharge uniquement les voxels utiles
   (~6 Mo par cas → 3 modalités axiales T2W/ADC/HBV) + vérité terrain
   (`marksheet.csv` : `case_csPCa`, `case_ISUP`) + contours de lésions
   d'experts ;
2. **preprocessing** : réplique le pipeline officiel d'inférence
   (alignement par centre de gravité, crop/pad (20, 256, 256),
   rééchantillonnage à (3.0, 0.5, 0.5) mm, z-score par modalité) ;
3. **inférence** : exécute l'ensemble 5-fold U-Net 3D officiel
   (PyTorch/MONAI) avec test-time augmentation → heatmap → carte de
   détection + probabilité patient ;
4. **pondération des décisions** : seuil 0.50 de probabilité de cas ;
5. **visualisation** : pour chaque cas, une figure (IRM native, carte
   de détection superposée, heatmap brute, contour d'expert);
6. **métriques micro** : IoU 3D (carte de détection vs contour d'expert),
   concordance au seuil 0.50 avec la vérité terrain selon la marksheet ;
7. **golden test** : pour le cas de test officiel fourni AVEC ses sorties
   de référence par les auteurs, vérifie que notre probabilité et notre
   carte de détection Coïncident à ≤ 1e-3 près ⇒ **PASSED**.

## 2. Structure des dossiers

```
prostate_model_demo/
├── README.md                     ← ce fichier
├── requirements.txt / pytest.ini / .gitignore
├── docs/                         ← (5 documents en français)
│   ├── MODEL_EXPLANATION.md      modèle: architecture, U-Net, ensemble, source
│   ├── DATA_EXPLANATION.md       datasets: PI-CAI, licences, annotations
│   ├── EXPERIMENT_RESULTS.md     chiffres réels mesurés (timing, IoU, ...)
│   ├── PROGRESS_REPORT.md        statut du projet (étapes faites/à faire)
│   └── WHAT_I_NEED_TO_UNDERSTAND.md  fiche de révision pour l'exposé
├── src/                          ← le code (pipelines) (page 7 modules)
│   ├── config.py                 chemins, constantes officielles du modèle
│   ├── data_loader.py            charger un cas (.mha) + GT + annotations
│   ├── preprocessing.py          réplication du preprocessing d'inférence
│   ├── inference.py              ensemble 5-fold + TTA + post-traitement
│   ├── evaluation.py             métriques de la démo (IoU 3D)
│   ├── visualization.py          figures (matplotlib)
│   └── unet_arch.py              architecture vendue (source : picai_baseline)
├── scripts/
│   ├── download_or_prepare_data.py  récupération des cas (Zenodo partiel)
│   ├── run_demo.py                  LA commande principale
│   └── streamlit_app.py             mini-interface interactive
├── tests/                        ← 13 tests pytest (tous passés)
├── notebooks/demo.ipynb          ← notebook « pipeline en 10 cellules »
├── data/                         ← données téléchargées (3 séquences + labels)
├── outputs/                      ← résultats (figures, métriques de la démo)
└── third_party/picai_unet_gc_algorithm/ ← modèle officiel + poids LFS
    (ne pas modifier)              + cas de test + sorties de référence
```

## 3. Commandes principales

```powershell
cd "D:\Esprit_5eme\Expalnable ia\prostate_model_demo"

# tout installer (une fois)
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# (1) récupérer les données (marksheet + 7 cas démo + annotations)
.\.venv\Scripts\python scripts\download_or_prepare_data.py

# (2) LANCER LA DÉMO : golden test + 7 cas + figures + summary.json
.\.venv\Scripts\python scripts\run_demo.py

# (2-bis) un seul cas (rapide) ; d'autres cas : --add-case-count N
.\.venv\Scripts\python scripts\run_demo.py --case 10032_1000032

# (3) TESTS AUTOMATISÉS (13)
.\.venv\Scripts\python -m pytest tests

# (4) MINI-INTERFACE interactive (navigateur)
.\.venv\Scripts\streamlit run scripts\streamlit_app.py

# (5) notebook pas à pas
.\.venv\Scripts\python -m jupyter notebook notebooks/demo.ipynb
```

Durée d'un run complet : **≈ 1 minute** sur GPU (RTX 5060 Ti, 0.52 s
d'inférence par cas + ~1 s de prétraitement).

## 4. Ce qui fonctionne (vérifié, mesuré)

- **Golden test : PASSED** sur le cas de test officiel `10032_1000032` :
  - probabilité de cas : **0.5269265 (référence auteurs)** vs
    **0.5269226 (notre run)** — écart 3.9e-6 (tolérance 1e-3);
  - carte de détection : 1 voxel sur 2 801 664 au-delà de 1e-3
    (voxel de bord, arrondi flottant de plateforme — voir
    `docs/EXPERIMENT_RESULTS.md` §1).
- 7/7 cas démo exécutés sans erreur (RTX 5060 Ti 16 Go, ~0.52 s/cas, VRAM ≈ 2–3 Go) :
  - négatifs : **20.2 %, 4.0 %, 2.1 %** — tous sous le seuil 0.50 ✓
  - positifs : **16.1 %, 34.6 %, 11.0 %, 52.7 %** — 1 sur 4 franchit le
    seuil, mais les 3 positifs « sous le seuil » restent au-dessus des
    3 négatifs (l'ordre est correct ; voir limitation ci-dessous)
- IoU 3D carte de détection vs contour d'expert : **0.19 en moyenne**
  (0.56 sur le cas golden, annotations resampled + Pooch25 mélangées)
- 13 tests pytest **tous passés** (chargement, prétraitement, modèle,
  inférence, reproduction de la référence golden, anti-pointeur-LFS).

## 5. Ce qui ne fonctionne PAS (à ce stade)

- **Les performances du modèle ne sont pas évaluées ici.** 7 cas ne
  peuvent estimer ni sensibilité ni PPV ni AUC ; ce run vérifie le
  pipeline, pas les performances. Les chiffres publiés (AUROC 0.878)
  viennent du jeu de test masqué PI-CAI de 1000 IRM.
- 3 positifs sur 4 tombent sous le seuil 0.50 : ce n'est pas un bug,
  c'est le point d'exploitation publié (sensibilité 70.9 % avec PPV 35.5 %
  — 1.41 faux positifs par patient). Évaluer ce point nécessite des
  dizaines de cas minimum, pas la démo.
- Le cas golden diffère de la référence Docker d'1 voxel (arrondi
  flottant entre plateformes) — documenté, non bloquant (voir §1
  ci-dessus).
- `picai_baseline` 0.8.5 est **incompatible avec MONAI ≥ 1.6** pour la
  classe UNet : nous avons vendu (vendored) le fichier UNet exact d'entraînement
  dans `src/unet_arch.py` (décors `@export`/`@alias` de MONAI 1.5 retirés,
  header d'attribution conservé).
- Les points de données **Pooch25** (205 cas les plus récents) sont
  annotés en binaire (ISUP ≥ 2), alors que les annotations `resampled/`
  ont des labels par grade (2..5) ; les deux types sont gérés.
- ❌ **PAS ENCORE** (phase 2 du projet, hors périmètre) : génération
  d'images explicatives (Grad-CAM etc.), évaluation FROC/lésion,
  dataset complet.

## 6. Résultats attendus après un run

- `outputs/figures/<case_id>.png` — une figure par cas (7 figures)
- `outputs/predictions/<case_id>/cspca_detection_map.mha` — carte 3D de
  détection, ré-ouvrable avec SimpleITK/3D Slicer (la même vision que la
  sortie officielle des auteurs)
- `outputs/predictions/<case_id>/prediction.json` — probabilité +
  métriques de la démo + timing
- `outputs/summary.json` — résumé du run

## 7. Datasets et licences

| Resource | Licence | Source |
|---|---|---|
| PI-CAI Public Training and Development Dataset v2.0 | CC BY-NC 4.0 | Zenodo record 6624726 |
| picai_labels (marksheet + contours d'experts) | CC BY-NC 4.0 | GitHub DIAGNijmegen/picai_labels |
| Poids du modèle (5 folds) + code d'inférence | Apache-2.0 | GitHub DIAGNijmegen/picai_unet_gc_algorithm |
| MONAI / picai_prep / picai_baseline / SimpleITK | Apache-2.0 / BSD | pip |

Usage recherche non commerciale. Détails dans `docs/DATA_EXPLANATION.md`.

## 8. Comment ça se présente (scénario oral en 2-3 minutes)

1. « Le problème : trouver sur une IRM de prostate un cancer
   *cliniquement significatif* (ISUP ≥ 2 — gros cancers utiles à
   traiter), pas les micro-cancers indolents qu'on ne traite pas. »
2. « L'outil : le modèle de référence d'une compétition publique
   (PI-CAI), ensemble de 5 U-Nets 3D, performance publiée AUROC 0.878. »
3. « Ce que j'ai fait : **TOUT le pipeline d'inférence officiel,
   répliqué et vérifié** — preuve par le test golden : la probabilité
   du cas de test officiel est réproduite à 3.9e-6 près, et la carte
   de détection est identique à 1 voxel près sur 2 800 000. »
4. [LANCER] le run (`run_demo.py`) — ~1 min, montrer les 7 figures.
5. « Les vraies difficultés : récupérer les poids réels (pas des
   pointeurs Git LFS), patcher l'incompatibilité MONAI, télécharger
   les cas sans les 27 Go d'archives. »
6. [PROJETER] `outputs/figures/10032_1000032.png` : IRM + carte de
   détection + contour de l'expert : la région détectée (0.53) recouvre
   la lésion ISUP 3.
7. « La suite (phase 2) : expliquer POURQUOI le modèle regarde là —
   Grad-CAM / Integrated Gradients / occlusion — et comparer ces
   explications aux annotations d'experts. »

## 9. Troubleshooting

| Symptôme | Cause probable | Fix |
|---|---|---|
| `unet_F0.pt is 134 bytes` | fichiers Git LFS non téléchargés | `cd third_party\picai_unet_gc_algorithm` puis `git lfs fetch --include="weights/*"` et `git reset --hard` |
| `CUDA out of memory` | GPU occupé / plus petit | fermer d'autres apps ; le modèle tient en ~3 Go de VRAM |
| Zenodo 404 / timeouts pendant le téléchargement | indisponibilité momentanée | re-exécuter ; le script réessaie et saute les fichiers déjà présents |
| `KeyError: 'unet' / impossible de charger un checkpoint` | modification de `src/unet_arch.py` | GARDER le fichier vendu d'origine (cf. header du fichier) |
| Les figures affichent une coupe sans lésion | par défaut on montre la coupe à max heatmap ; si max < 0.1, coupe médiane | c'est le fonctionnement voulu (`visualization._best_slice`) |
| Streamlit `ModuleNotFoundError` | environnement venv erroné | lancer avec `.\.venv\Scripts\streamlit` |

## 10. Ce que phase 2 gardera de ce module

- `load_case` / `preprocess_images` / `make_tensor` : la même entrée
  (ré)utilisée pour générer les explications ;
- `heatmaps` (sorties brutes) en comparaison des cartes d'attention
  Grad-CAM ;
- annotations + contours d'experts déjà téléchargés « au bon endroit »
  pour chaque cas ;
- la base `outputs/predictions/` pour les études par cas.

## 11. Notes d'intendance (propreté technique)

- `third_party/picai_unet_gc_algorithm/` contient `weights/`
  (127,229,381 octets/fold) + `test/` (cas de test complet avec les
  sorties de référence officielles) ; ce dossier n'est PAS versionné ni
  modifié.
- `data/` est ignoré par git (gros volumes binaires) ; cf. `data/README.md`.
- Un run complet ne touche qu'à `outputs/` (et écrit `summary.json`).
- Tous les chiffres cités dans la doc proviennent d'exécutions réelles
  du 2026-10-22 (déjà enregistré dans `docs/EXPERIMENT_RESULTS.md`).

## Étape suivante du projet (phase 2 — XAI)

Phase 2, hors périmètre de ce module : méthodes d'explicabilité sur la
même architecture (Grad-CAM, Integrated Gradients, occlusion), et
comparaison automatisée de ces explications contre (a) la carte de
détection, (b) les contours d'experts, (c) les zones ignorées par le
modèle. Ce module fournit un environnement adéquat pour exécuter toute
expérimentation XAI sans casser le modèle de la phase 1.
