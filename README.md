# IA Explicable pour le Cancer de la Prostate (bpMRI — PI-CAI)

Projet d'**Intelligence Artificielle Explicable (XAI)** appliquée à la détection du **cancer de la prostate cliniquement significatif (csPCa, ISUP ≥ 2)** sur **IRM biparamétrique (bpMRI)**, avec une phase de recherche bibliographique et une implémentation complète d'inférence / évaluation / explicabilité du modèle de référence **PI-CAI**.

> **Résumé en une phrase :** reproduction fidèle du détecteur csPCa officiel PI-CAI (U-Net 5 plis), benchmark honnête de 10 configurations de post-traitement sur 600 cas stratifiés, et audit d'explicabilité avec 4 méthodes XAI + contrôles qualité.

---

## Sommaire

- [Contenu du projet](#contenu-du-projet)
- [Résultats principaux](#résultats-principaux)
- [Démarrage rapide](#démarrage-rapide)
- [Architecture du dépôt](#architecture-du-dépôt)
- [Pipeline Phase 2 en une commande](#pipeline-phase-2-en-une-commande)
- [Explicabilité (XAI)](#explicabilité-xai)
- [Tests](#tests)
- [Démonstration web](#démonstration-web)
- [Recherche & rapports](#recherche--rapports)
- [Données & poids du modèle](#données--poids-du-modèle)
- [Reproductibilité](#reproductibilité)
- [Limites connues](#limites-connues)
- [Licence & avertissements](#licence--avertissements)

---

## Contenu du projet

Le projet combine trois livrables :

1. **Une recherche vérifiée** sur l'IA explicable en imagerie prostatique (28 dépôts inspectés, 45 liens vérifiés) → `RESEARCH_REPORT.md`, `TOP_MODELS.md`, `RECOMMENDATION.md` + site interactif `index.html`.
2. **Une reproduction de Phase 1** : inférence byte-fidèle du modèle officiel PI-CAI `picai_unet_gc_algorithm` (golden test **PASSED**, |e-likelihood| = 3.94e-6 vs référence auteurs) → `prostate_model_demo/`.
3. **Une Phase 2 complète** : benchmark de post-traitement, métriques médicales, analyse d'erreurs, statistiques (bootstrap + DeLong) et audit XAI sur 600 cas → `evaluation/`, `xai/`, `scripts/`, `experiments/`, `reports/`.

---

## Résultats principaux

### Phase 2 — test split (1er et unique run après gel de la config, 2026-10-23)

| Métrique | officiel @ seuil 0.50 | config gelée `no_extract` @ 0.157 |
|---|---|---|
| AUROC | 0.8042 | **0.8081** (IC 0.710–0.895) |
| AP / PR-AUC | 0.7064 | **0.7107** (IC 0.586–0.827) |

- **Sélection pré-enregistrée** : meilleure PR-AUC sur *validation* uniquement (tie-break AUROC) ; seuil = F1 max sur validation ; test scoré **une seule fois** après gel (`experiments/configs/selected_postproc.json`).
- Le gain vient de la **couche décision/post-traitement** (aucun ré-entraînement) : le seuil officiel 0.50 était quasi-aveugle (17–24 % de sensibilité), et l'extraction RGA de candidates éliminait de vraies lésions.
- **Non-significativité assumée** : DeLong p = 0.95, ΔAUROC ≈ 0 — le classement par le modèle n'a pas changé, seul l'**opérant point** s'améliore.

### XAI — 8 cas audités

4 méthodes (Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion) + 4 contrôles qualité (stabilité, déplacement, fidélité, localisation vs contours experts) → `experiments/results/xai/xai_quality.csv`.

---

## Démarrage rapide

### Prérequis

- Windows (scripts de démo `.bat`/`.ps1`), Python **3.12**
- GPU NVIDIA CUDA (validé : RTX 5060 Ti, CUDA 12.8) — l'inférence fonctionne aussi en CPU (lent)
- ~10 Go d'espace disque (données PI-CAI partielles + poids)

### 1. Cloner le dépôt

```powershell
git clone https://github.com/MaherAbderrahim/expalnable-ia.git
cd expalnable-ia
```

### 2. Installer l'environnement Python

L'environnement partagé se trouve dans `prostate_model_demo/.venv` (le pipeline le détecte automatiquement) :

```powershell
cd prostate_model_demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
```

### 3. Récupérer le modèle de référence (Phase 1)

Les poids officiels (5 checkpoints U-Net × 127 Mo) sont dans le dépôt tiers `DIAGNijmegen/picai_unet_gc_algorithm` via **Git LFS** :

```powershell
git lfs install
git clone https://github.com/DIAGNijmegen/picai_unet_gc_algorithm.git prostate_model_demo/third_party/picai_unet_gc_algorithm
cd prostate_model_demo/third_party/picai_unet_gc_algorithm
git lfs pull   # sans git-lfs, on récupère des pointeurs de 134 octets et l'inférence échoue silencieusement
cd ../../..
```

### 4. Lancer la démo web (le plus simple)

```powershell
.\Lancer-Demo-IA.bat      # sert le tableau de bord reports/demo/ (+ tunnel optionnel)
.\Arreter-Demo-IA.bat     # arrête le serveur
```

---

## Architecture du dépôt

```
expalnable-ia/
├── README.md                     # ce fichier
├── index.html / script.js /      # site interactif du rapport de recherche
│   style.css / research-data.*
├── RESEARCH_REPORT.md            # rapport de recherche vérifié ([V]/[R]/[!])
├── TOP_MODELS.md                 # top 5 des modèles, notés sur 10 critères
├── RECOMMENDATION.md             # recommandation finale argumentée
│
├── prostate_model_demo/          # PHASE 1 — reproduction fidèle du modèle officiel
│   ├── demo.ipynb                # notebook de démonstration
│   ├── requirements.txt          # dépendances (torch cu128, monai, picai_prep…)
│   ├── third_party/              # poids PI-CAI (gitignoré — via Git LFS)
│   └── data/                     # données téléchargées (gitignoré)
│
├── evaluation/                   # PHASE 2 — moteur d'évaluation
│   ├── paths.py                  # chemins/constants partagés
│   ├── splits.py                 # splits au niveau patient (fuite zéro)
│   ├── infer.py                  # inférence 1 passage, ensemble 5 plis + TTA
│   ├── postproc.py               # post-traitement offline (officiel + variantes)
│   ├── eval_model.py             # évaluation des composants sauvegardés
│   ├── metrics.py                # métriques cas + lésion (AUROC, AP, Dice, FROC…)
│   ├── statistics.py             # IC bootstrap + DeLong apparié + variance multi-seed
│   └── error_analysis.py         # analyse d'erreurs stratifiée (centre, PI-RADS…)
│
├── xai/                          # audit d'explicabilité
│   ├── methods.py                # Grad-CAM / Grad-CAM++ / IG / Occlusion (3D)
│   ├── quality.py                # contrôles qualité des explications
│   └── visualize.py              # superpositions image / détection / heatmap
│
├── scripts/                      # orchestration
│   ├── run_phase2.py             # pipeline complet en 1 commande (7 étapes)
│   ├── prepare_data.py           # sélection stratifiée + téléchargement robuste
│   ├── build_splits.py           # splits patient leak-free
│   ├── run_inference.py          # inférence val+test (1 passe GPU)
│   ├── select_config.py          # sélection SUR VALIDATION uniquement
│   ├── eval_final.py             # évaluation finale (test scoré une fois)
│   ├── explain.py                # pile XAI complète
│   ├── build_demo_page.py /      # génération du tableau de bord démo
│   │   build_site_v2.py
│   └── presentation_build.py     # exécution + export HTML du notebook
│
├── tests/                        # suite pytest (30 tests)
├── experiments/                  # configs, logs et résultats bruts (CSV/JSON)
├── reports/
│   ├── REPORT.md                 # rapport Phase 2 détaillé
│   ├── BEST_MODEL.md             # modèle gelé + justification honnête
│   └── demo/                     # tableau de bord web auto-portant
├── data/                         # splits + prédictions (lourds, gitignorés)
├── tools/launch_demo.ps1         # lanceur de démo PowerShell
├── Lancer-Demo-IA.bat            # lancer la démo (Windows)
└── Arreter-Demo-IA.bat           # arrêter la démo (Windows)
```

---

## Pipeline Phase 2 en une commande

```powershell
python scripts/run_phase2.py                # télécharge, coupe, infère, sélectionne, évalue, explique
python scripts/run_phase2.py --skip-download # données déjà sur disque
python scripts/run_phase2.py --from eval     # reprendre à une étape donnée
python scripts/run_phase2.py --force <stage> # forcer une étape
```

Les 7 étapes (`download → splits → inference → select → eval → explain → regression`) sont **idempotentes et reprenables** ; chaque étape est ignorée si sa sortie existe déjà. Tout est journalisé dans `experiments/logs/run_phase2.log`.

### Données et découpage

- Sous-ensemble **stratifié de 600 cas** (sur 1 500 publics) par centre / hospital / csPCa / biopsy / PI-RADS — seed **2026**.
- Split **patient** 60/20/20 (360/120/120), recouvement inter-splits **vérifié nul** (`check_no_leakage`).
- Vérité terrain : csPCa au cas + **contours de lésion par experts humains** (`picai_labels`).

---

## Explicabilité (XAI)

```powershell
python scripts/explain.py   # appelé automatiquement par run_phase2.py
```

- **4 méthodes** : Grad-CAM, Grad-CAM++ (pytorch-grad-cam), Integrated Gradients et Occlusion (Captum) — toutes adaptées au pipeline 3D (fold-flip TTA, dé-normalisation du crop/pad).
- **4 contrôles qualité** (ne pas faire confiance à une belle heatmap) :
  1. **Stabilité** — cohérence entre folds de l'ensemble.
  2. **Déplacement** — la heatmap suit-elle la lésion quand on la compare au contour expert ?
  3. **Fidélité** — chute de probabilité après suppression de la région expliquée.
  4. **Localisation** — recouvrement heatmap vs carte de détection du modèle ET vs contours experts (LocAcc).
- Sorties : `experiments/results/xai/` (`.npz` bruts + figures) et figures intégrées au tableau de bord démo.

---

## Tests

```powershell
pytest tests          # 30 tests Phase 2 (métriques, post-traitement, stats, splits, XAI)
```

La régression Phase 1 (golden test) fait partie de l'étape `regression` du pipeline.

---

## Démonstration web

| Commande | Effet |
|---|---|
| `.\Lancer-Demo-IA.bat` | Sert `reports/demo/index.html` en local, ouvre le navigateur, propose un tunnel Cloudflare public |
| `.\Arreter-Demo-IA.bat` | Arrête le serveur et le tunnel |

Le tableau de bord (`reports/demo/`) est **auto-portant** (HTML + assets embarqués) : métriques, courbes, cas par cas, panneaux XAI. Il est reconstruit par `python scripts/build_demo_page.py` (ou `build_site_v2.py`).

Le notebook de présentation `phase1_phase2_demo.ipynb` (racine) se rebuild et s'exécute via :

```powershell
python scripts/presentation_build.py
```

---

## Recherche & rapports

| Fichier | Contenu |
|---|---|
| `RESEARCH_REPORT.md` | Rapport de recherche vérifié — chaque affirmation marquée **[V]** (vérifiée), **[R]** (rapportée) ou **[!]** (signalée) |
| `TOP_MODELS.md` | Top 5 des modèles notés sur 10 critères (réutilisation, qualité, poids, licence…) |
| `RECOMMENDATION.md` | Recommandation finale : U-Net PI-CAI + Grad-CAM/IG confrontés aux contours experts + couche PI-RADS |
| `reports/REPORT.md` | Rapport Phase 2 complet (benchmark, significativité, erreurs, XAI) |
| `reports/BEST_MODEL.md` | Config gelée du meilleur modèle + décomposition honnête du gain |
| `index.html` | Site web interactif du rapport de recherche |

**Recommandation en une ligne** : U-Net de référence PI-CAI (poids déjà publiés) expliqué par Grad-CAM + Integrated Gradients, cartes de chaleur confrontées à la carte de détection du modèle ET aux contours d'expert, avec une couche de signes PI-RADS en prolongement de Hamm et al. (*Radiology* 2023).

---

## Données & poids du modèle

| Élément | Source | Licence | Taille |
|---|---|---|---|
| PI-CAI Public Training v2.0 (1 500 IRM bpMRI, 1 295 contours experts) | [Zenodo 6624726](https://zenodo.org/records/6624726) | **CC BY-NC 4.0** (non commercial) | ~7 Go (sous-ensemble 600 cas) |
| Poids U-Net 5 plis (`unet_F0..F4.pt`) | [DIAGNijmegen/picai_unet_gc_algorithm](https://github.com/DIAGNijmegen/picai_unet_gc_algorithm) | Apache-2.0 | 5 × 127 Mo (**Git LFS**) |
| Baseline PI-CAI | [DIAGNijmegen/picai_baseline](https://github.com/DIAGNijmegen/picai_baseline) | Apache-2.0 | code |

> ⚠️ **Piège Git LFS** : les poids sont stockés via Git LFS. Un clone sans `git-lfs` installé récupère des pointeurs de 134 octets et **l'inférence échoue silencieusement**. Vérifiez la taille des fichiers (~127 Mo) après le `git lfs pull`.

Les données et poids ne sont **pas** inclus dans ce dépôt (trop lourds) ; ils sont regénérés par `scripts/prepare_data.py` et le clonage LFS ci-dessus.

---

## Reproductibilité

- **Graines fixes** : 2026 partout (sélection, splits, bootstrap).
- **Sélection pré-enregistrée** : la règle est écrite dans `scripts/select_config.py` *avant* de toucher au test — aucun métrique n'a été réglé sur le test.
- **Une seule passe test** : le test n'a été scoré qu'une fois, après gel.
- Pipeline idempotent + log complet : `experiments/logs/run_phase2.log`.
- Résultats figés dans `experiments/results/` (CSV/JSON) et `experiments/configs/selected_postproc.json`.

---

## Limites connues

- Les checkpoints officiels ont été entraînés sur **tous** les 1 500 cas publics PI-CAI : toute métrique PI-CAI calculée ici est un **borne supérieure optimiste** (les cas « validation/test » du sous-ensemble ont été vus à l'entraînement par le modèle de référence).
- Ce projet peut revendiquer un **benchmark de calibration/post-traitement** et un **audit d'explicabilité** du modèle publié — pas une mesure de généralisation sur un jeu nouveau.
- Aucun entraînement GPU n'a été effectué (contrainte projet) : l'amélioration vit entièrement dans la couche post-traitement/décision.
- Le seuil gelé (0.157) est spécifique à la distribution PI-CAI et doit être recalibré sur un nouveau centre.

---

## Licence & avertissements

- Le **code** de ce dépôt est un projet académique (École Esprit — 5e année) ; le modèle officiel PI-CAI est sous **Apache-2.0**.
- Les **données PI-CAI** sont sous **CC BY-NC 4.0** : usage **non commercial uniquement**. Respectez cette contrainte en toute réutilisation.
- **Avertissement médical** : outil de recherche uniquement, non validé cliniquement, ne doit en aucun cas guider une décision médicale.

---

*Projet réalisé dans le cadre du cursus ingénieur (5e année) — Esprit, Tunis. Démonstration : `Lancer-Demo-IA.bat`.*
