# IA explicable pour le cancer de la prostate — Rapport de recherche

**Projet :** IA explicable appliquée à la détection / l'analyse du cancer de la prostate en imagerie
**Focus :** IRM prostatique biparamétrique (csPCa, ISUP ≥ 2) et histopathologie H&E (grading de Gleason / ISUP)
**Date :** 21 octobre 2026
**Fichiers associés :** `research-data.json` / `research-data.js` (base structurée), `index.html` (site interactif), `TOP_MODELS.md`, `RECOMMENDATION.md`

---

## 0. Comment ce rapport a été produit, et ce que « vérifié » signifie ici

Le cahier des charges demandait une recherche qui vérifie plutôt que de résumer. Chaque affirmation factuelle ci-dessous tombe dans l'un des trois états suivants :

| Marque | Signification |
|---|---|
| **[V]** | **Vérifié directement.** J'ai récupéré la page, le fichier, l'enregistrement d'API ou les octets bruts. |
| **[R]** | **Rapporté.** La source l'affirme, mais je ne l'ai pas reproduit indépendamment. |
| **[!]** | **Signalé.** L'affirmation de la source entre en conflit avec ce que j'ai trouvé, ou elle est structurellement suspecte. |

### L'appareil de vérification

Quatre mécanismes indépendants, parce que chacun attrape une classe d'erreur différente :

1. **API REST GitHub** — métadonnées des dépôts (étoiles, identifiant SPDX de licence, branche par défaut).
2. **`git clone --filter=blob:none --no-checkout` + `git ls-tree -r -l HEAD`** — ne télécharge que les objets de commit et d'arborescence, ce qui fournit la liste de fichiers *faisant autorité* et les tailles de blobs. C'est cet outil qui a établi qu'un dépôt de modèle prostatique contient réellement 5 points de contrôle de 127 Mo.
3. **Sondes HTTP de fichiers bruts** — chaque fichier candidat a été demandé sur `raw.githubusercontent.com` et sa taille `Content-Length` enregistrée. Un 404 ne laisse aucune ambiguïté.
4. **API REST Zenodo** — tailles exactes en octets et identifiants de licence pour chaque archive de modèle publiée.

**Résultat :** 21 dépôts candidats inspectés en profondeur (28 catalogués au total), 6 fichiers de poids sondés, 13 licences confirmées, 3 dépôts signalés comme **n'ayant aucune licence exploitable**, 45 liens vérifiés.

### Les erreurs attrapées dans mes propres hypothèses

- **Un dépôt annoncé « sans README » l'était seulement parce que j'interrogeais la mauvaise branche.** Trois dépôts `DIAGNijmegen` ont `master` comme branche par défaut, pas `main`. Une vérification superficielle les aurait déclarés vides à tort.
- **Un clone blobless a expiré** sur le dépôt de la solution 1er prix du PANDA, et un autre ne renvoie ni README ni LICENSE à la racine. Plutôt que de les présenter comme utilisables, ils sont marqués explicitement comme non vérifiés (§4).
- **Un piège technique majeur propre à ce projet** : les poids du modèle PI-CAI de référence sont stockés via **Git LFS**. Le dépôt contient des pointeurs de 134 octets ; les fichiers réels font 127 229 381 octets. Un clone sans `git-lfs` récupère des pointeurs et **l'inférence échoue silencieusement**. C'est le détail que je signale en premier à quiconque embarkant sur ce projet.

---

## 1. Synthèse

Le résultat principal est **meilleur** que celui du dossier cérébral, et il tient en une phrase : **en imagerie prostatique, une IA explicable validée cliniquement existe déjà.** Ce n'est plus un champ à explorer, c'est un standard à mesurer et à étendre.

Cinq conclusions, par ordre d'importance pratique :

1. **Le modèle de référence PI-CAI est disponible avec ses poids, dans le dépôt, sous licence Apache-2.0.** `DIAGNijmegen/picai_unet_gc_algorithm` contient `weights/unet_F0.pt` à `unet_F4.pt` (les 5 plis de la validation croisée), vérifiés par `git ls-tree` et par sonde HTTP : 134 octets de pointeur LFS pointant vers un fichier réel de 127 229 381 octets. C'est l'équivalent prostatique exact de ce que l'on cherchait côté cerveau, et il est plus facile à obtenir.

2. **PI-CAI est le jeu de données de référence, avec 1 500 IRM de 1 476 patients et 1 295 contours de lésion d'expert** — et, depuis juillet 2026, des scores PI-RADS pour les 1 500 cas (Pooch et al. 2025). Le jeu est en **CC BY-NC 4.0** : usage non commercial uniquement. C'est la contrainte structurante du projet, et elle n'a pas d'équivalent en imagerie cérébrale où BRISC est en CC BY.

3. **La prostate offre deux vérités terrain indépendantes pour la même image** — la carte de détection continue du modèle, et les contours d'expert. Vous pouvez donc *mesurer* une carte de chaleur, ce qui est la principale faiblesse de la littérature XAI prostatique.

4. **Hamm et al. (*Radiology* 2023) est la référence à battre.** Leur modèle n'explique pas par des pixels mais par les **14 signes PI-RADS** : exactitude des explications 80 % (1 080/1 352) confirmée par des experts, prédiction des signes à 98 %, et — decisive — une étude de lecteurs où la confiance des non-experts passe de 3,4 à 4,1 sur 5 (p = 0,007) avec 58 s de temps de lecture en moins (p = 0,009). L'éditorial qui l'accompagne, « Don't Trust, Verify », formule la posture du projet.

5. **En histopathologie, l'explicabilité est native et ne demande aucune méthode post-hoc.** L'attention MIL (CLAM, ABMIL, TransMIL) *est* l'explication ; et la frontière actuelle (StructMIL, grading prototype adaptatif) remplace l'attention opaque par des **prototypes histologiques nommés**, dont on vérifie la localisation contre les masques annotés.

**Recommandation en une ligne :** *le U-Net de référence PI-CAI (poids déjà publiés) sur PI-CAI, expliqué par Grad-CAM + Integrated Gradients, avec la carte de chaleur confrontée à la fois à la carte de détection du modèle et aux contours experts, et une couche de signes PI-RADS en prolongement de Hamm et al.* Raisonnement complet dans `RECOMMENDATION.md`.

---

## 2. Taxonomie des tâches : la prostate a trois niveaux, pas deux

### 2.1 Détection au niveau du patient
**Entrée :** un examen bpMRI complet. **Sortie :** une probabilité que le patient ait un cancer cliniquement significatif (csPCa, ISUP ≥ 2).
**Ce que cela dit au clinicien :** faut-il biopsier ?
**Ce que l'IA explicable doit ajouter :** *où* et *sur quels signes*. Une probabilité seule n'est pas actionnable.

### 2.2 Détection au niveau de la lésion
**Entrée :** le même examen. **Sortie :** une carte de détection voxel à voxel — une **probabilité de csPCa par voxel** — plus une liste de lésions candidates avec leur score.
**Ce que cela dit au clinicien :** quelle lésion, et à quel niveau de suspicion.
**C'est le niveau décisif pour l'explicabilité**, et il a une propriété que la classification n'a pas : **la carte de détection est déjà une explication spatiale continue.** Elle dit où et à quel degré. Une carte de chaleur superimposed sur elle est donc directement comparable, voxel par voxel.

### 2.3 Grading (ISUP 0-5)
**Entrée :** une lame entière H&E (PANDA) ou un core de micro-puce (Gleason 2019). **Sortie :** un grade ordinal.
**Ce que cela dit au clinicien :** l'agressivité, qui conditionne le traitement.
**L'IA explicable ici est radicalement différente** : l'unité n'est plus le voxel mais le **patch**, et le mécanisme d'attention du MIL *est* l'explication. Un modèle d'attention indique les glandes qui portent le grade.

### 2.4 Ce que cela change par rapport à l'imagerie cérébrale

Trois différences structurantes, qu'il faut assumer :

| | Tumeur cérébrale | Cancer de la prostate |
|---|---|---|
| **Métrique naturelle** | Dice (segmentation) | AUROC patient + AP + FROC lésion (détection), QWK (grading) |
| **Unité de vérité terrain** | le voxel d'un masque | le voxel d'un contour expert **et** la probabilité continue du modèle |
| **Lexique clinique disponible** | aucun | **PI-RADS** (14 signes) et **patterns de Gleason** |
| **Validation clinique disponible** | quasi inexistante | étude de lecteurs PI-CAI (50+ radiologues) ; Hamm et al. validé |
| **Contrainte de licence des données** | CC BY (BRISC) | **CC BY-NC** (PI-CAI, PANDA) |

Le tableau ci-dessous est la vraie nouvelle du dossier prostatique : **le problème n'est pas de construire une explicabilité, mais de mesurer si celle qui existe est fidèle, et de l'étendre aux niveaux qu'elle ne couvre pas encore** (la lésion, l'ISUP, l'histopathologie).

---

## 3. Modèles — catalogue complet

27 entrées de modèles sont cataloguées dans `research-data.json`. Cette section détaille ceux qui comptent.

### 3.1 U-Net de référence PI-CAI — le point de départ

| Champ | Valeur |
|---|---|
| Architecture | Ensemble de 5 U-Net 3D (validation croisée 5 plis) ; détecte et classe les lésions csPCa |
| Tâche | Détection + diagnostic de csPCa, au niveau du patient et de la lésion |
| Jeu de données | PI-CAI Public Training and Development Dataset v2.0, 1 295 cas avec annotations humaines ISUP ≥ 2 |
| Classes | 2 par lésion (csPCa / non), plus une probabilité au niveau du cas |
| Entrée | 3 canaux : T2W axial, carte ADC, DWI à haut b-value (HBV) |
| Performance | PI-CAI, phase de test fermé : score 0,718 · AUROC 0,878 · AP 0,558 · Sens3 0,693 · Sens4 0,636 · Sens5 0,472 (IC 95 % AUROC 0,855-0,900). Version semi-supervisée : score 0,734 · AUROC 0,881 · AP 0,586 **[R]** |
| Code | `github.com/DIAGNijmegen/picai_unet_gc_algorithm` — Apache-2.0 **[V]** |
| Poids | **OUI, DANS LE DÉPÔT** : `weights/unet_F0.pt` à `unet_F4.pt` **[V]** |
| Format | `.pt` (PyTorch), **via Git LFS** |
| XAI | Aucune intégrée — mais la sortie EST une carte de détection, donc la vérité terrain de l'explication |
| Avantages | Le seul artefact prostate qui livre code + poids entraînés + déploiement Docker, sous licence permissive |
| Limites | Pas d'IA explicable intégrée ; poids en LFS ; entrée volumique 3D |
| Difficulté de réutilisation | Facile à moyen |

**Contenu vérifié** `[V]` (par `git ls-tree -r -l HEAD`, 33 fichiers) : `weights/unet_F0.pt` à `unet_F4.pt` · `Dockerfile` (859 o) · `LICENSE` (11 357 o) · `process.py` (11 253 o) · `requirements.txt` (257 o) · `setup.cfg` · `build.sh` / `export.sh` / `test.sh` · `test/` avec des exemples `.mha` (T2, ADC, HBV, coronal, sagittal) et `clinical-information-prostate-mri.json`. Le fichier `.gitattributes` déclare le filtre LFS pour `*.pt`, `*.mha` et `*.nii.gz`.

**Le piège, énoncé précisément** : `weights/unet_F0.pt` renvoie un contenu de 134 octets, qui est un pointeur LFS :
```
version https://git-lfs.github.com/spec/v1
oid sha256:d21d31bb8609ae12272da93a8c8556613cdaa0834c2fe1451f59557f814ce69d
size 127229381
```
Le fichier réel fait donc **127 229 381 octets**. Avec `git lfs pull` (ou `GIT_LFS_SKIP_SMUDGE=0`), on l'obtient. Sans cela, on obtient un pointeur, et l'inférence échoue sans message clair.

### 3.2 picai_baseline — la boîte à outils officielle

Apache-2.0, 142 fichiers, installable par `pip install git+https://github.com/DIAGNijmegen/picai_baseline`. Fournit **trois** références supportées — U-Net, nnU-Net, nnDetection — partageant la même préparation de données, avec une documentation par modèle (`unet_baseline.md`, `nnunet_baseline.md`, `nndetection_baseline.md`) et surtout les **découpages 5 plis prêts à l'emploi**, qui sont l'élément le plus difficile à reproduire soi-même. C'est la façon la plus correcte de reproduire les chiffres publiés.

### 3.3 PI-CAI : nnU-Net semi-supervisé

Apache-2.0, branche `master`, 33 fichiers. Le dépôt apporte trois entraîneurs nnU-Net personnalisés — `nnUNetTrainerV2_focalLoss.py`, `nnUNetTrainerV2_Loss_FL_and_CE.py`, `nnUNetTrainerV2_Loss_CE_checkpoints.py` — qui gèrent le bruit des pseudo-étiquettes. Score de test fermé 0,734, le meilleur du dépôt. Deux des cinq meilleurs algorithmes du défi étaient basés sur nnU-Net.

### 3.4 AIMI Task788_Prostate — les poids permissifs

| Champ | Valeur |
|---|---|
| Architecture | nnU-Net **v1**, 3D, segmentation de la glande prostatique sur T2 |
| Jeu de données | Entraîné sur PI-CAI |
| Poids | **Zenodo 8290093**, fichier `Task788_Prostate.zip` = **1 675 047 442 octets (1,67 Go)**, licence **CC BY 4.0** **[V via l'API Zenodo]** |
| Code | `github.com/bamf-health/aimi-prostate-mr` — MIT (LICENSE 1 068 o) **[V]** |
| Avantages | Les poids prostatiques les plus permissivement licenciés trouvés, entraînés sur PI-CAI donc dans le même domaine que le projet |
| Limites | nnU-Net v1 (ancienne version) ; segmentation de la glande seulement, pas des lésions ; le dépôt GitHub est mince (README 5 586 o) |
| Difficulté | Moyen |

Indispensable en pratique : sans segmentation de la prostate, la détection de lésion n'a pas de région d'intérêt.

### 3.5 GleasonXAI — l'explicabilité intrinsèque

| Champ | Valeur |
|---|---|
| Architecture | U-Net qui segmente des **descriptions** de patterns de Gleason en terminologie d'anatomopathologiste |
| Jeu de données | 1 015 cores de micro-puces annotés par un groupe international de **54 anatomopathologistes** |
| Performance | Dice 0,713 ± 0,003 (entraîné sur les **explications**) contre 0,691 ± 0,010 (entraîné directement sur les patterns). Entraînement sur labels souples pour capter l'incertitude inter-annotateur **[R]** |
| Code | `github.com/DBO-DKFZ/GleasonXAI` — MIT (LICENSE 1 066 o) **[V]** |
| Poids | **NON** — aucun point de contrôle dans l'arborescence |
| Difficulté | Moyen |

**Contenu vérifié** `[V]` (80+ fichiers) : `src/gleasonxai/` (`gleason_data.py` 18 980 o, `lightning_modul.py` 21 798 o, `jdt_losses.py`, `loss_functions.py`, `model_utils.py`, `tree_loss.py`, `augmentations.py`) · `configs/` pour **7 architectures** de segmentation (unet, unetplusplus_efficientnet, densenet, efficientnet, deeplabv3p_efficientnet, fcn, resnet) plus pertes, dataset, optimiseur, trainer, dataloader, augmentations, logger · `scripts/` (`train.py`, `test.py`, `run_gleasonXAI.py`, `calculate_fleiss_kappa.py`, `calculate_dataset_characteristics.py`, `create_metric_visualization.py`, `evaluate_paper_results.ipynb` 896 610 o, `feature_visualization.ipynb` 106 444 o) · `tests/` unitaires et d'intégration · `pyproject.toml`, `uv.lock`, `CHANGELOG.md`.

C'est le **code de recherche le plus complet du dossier** : configs complètes, tests, notebooks d'évaluation et de visualisation de features, et calcul du kappa de Fleiss pour quantifier la variabilité inter-annotateur. C'est le patron à copier.

### 3.6 Modèle XAI à signes PI-RADS (Hamm et al.) — la référence

| Champ | Valeur |
|---|---|
| Architecture | CNN détectant la lésion index, classant PCa / csPCa / bénin, ET justifiant par **14 signes PI-RADS** (T2W, DWI, ADC) ; explication textuelle ET visuelle |
| Jeu de données | 1 224 hommes consécutifs (2012-2017), **3 260 lésions** avec analyse anatomopathologique |
| Performance | AUROC **0,89** (interne) / **0,87** (externe, PROSTATEx) ; sensibilité 93 % (IC 95 % 87-98) ; ~1 faux positif par patient. Exactitude des explications **80 %** (1 080/1 352) confirmée par des experts. Prédiction des 14 signes : exactitude 98 %, sensibilité 98 %, spécificité 99 %. Lecture assistée : confiance des non-experts **4,1 vs 3,4** (p = 0,007), temps de lecture **−58 s** (p = 0,009). Traitement de 600-900 ms par lésion. **[R]** |
| Code / poids | Non localisés dans cette recherche **[!]** |
| XAI | **Par concepts** : l'explication est un compte rendu radiologique, pas une carte de chaleur |
| Difficulté | Difficile à reproduire / Facile comme benchmark |

**C'est l'artefact le plus important du dossier prostatique**, pour une raison qu'aucun autre n'a : c'est le seul où l'explicabilité a été **validée par des experts et testée sur le comportement de lecteurs**.

### 3.7 MIL d'attention et attribution par prototypes

| Modèle | Description | Performance | XAI |
|---|---|---|---|
| **ABMIL / CLAM / DSMIL / TransMIL** | Une WSI = un sac de patchs ; l'attention apprend un poids par patch | Standard de fait en pathologie computationnelle | **Natif** : la carte d'attention EST l'explication |
| **StructMIL** (npj Digit Med 2026) | Pooling prototypique + a priori topologique de graphe | Camelyon16 : +3,2 % d'AUC inter-centre (AUC 0,967). PANDA : **+7,4 %** de gain de Cohen's kappa en robustesse inter-scanner | Prototypes **nommés** = motifs histologiques réels |
| **DSPA-U-MIL** (Med Image Anal 2026) | Jetons d'agrégation de patterns de Gleason, guidage par concepts d'expert, distillation, contrainte d'incertitude | Jusqu'à **+6,7 % de QWK** sur CLAM, sur 5 jeux (10 809 WSI + 1 133 TMA) | Les poids de gating de l'élève sont **alignés sur les annotations d'anatomopathologistes** |
| **Grading prototype adaptatif** (arXiv:2603.04947) | Prototype-based faiblement supervisé | Les prototypes à attention > 0,5 localisent des structures glandulaires bien définies, **en accord avec les masques de vérité terrain** | Attribution par prototypes |

La critique documentée de l'attention MIL, qu'il faut citer : « *les cartes d'attention indiquent quelles régions ou caractéristiques le modèle met en avant, mais elles ne désignent pas des cas prototypiques* ». Les travaux à prototypes y répondent directement.

### 3.8 Le reste, brièvement

| Modèle | Tâche | Données | Poids | Licence | Remarque |
|---|---|---|---|---|---|
| **Radiologie 2024** (DL 3D csPCa) | Détection csPCa | Cohorte mono-établissement, test externe PROSTATEx | Non | — | Performance **non différente** des radiologues (test de DeLong). Grad-CAM sur la lésion dans 35/38 et 56/58 vrais positifs |
| **ProstAttention-Net** (Med Image Anal 2022) | Seg multi-étiquette par grade | IRM clinique | Non | Elsevier | L'attention par grade *est* l'explication |
| **U-Net / AH-Net** (Abdom Radiol 2022) | Détection + segmentation | 525 patients, biopsie IRM-TRUS | Non | Springer | Sensibilité 70,9 % / 74,4 %, **PPV 35,5 % / 47,8 %**, ~1,4 faux positif/patient. Protocole de données exemplaire |
| **nnU-Net annotation guidée par compte rendu** (Bosma et al.) | Détection csPCa | 7 756 IRM dont 3 050 annotés, 300 externes | Non | Apache-2.0 (pipeline) | Explique les DONNÉES, pas le modèle : compter les lésions du compte rendu radiologique pour guider l'annotation automatique |
| **Langkilde et al. 2025** (G2 dépistage) | Détection csPCa | Population de DÉPISTAGE | Non | — | **AUROC 0,83** contre ~0,88 en population de suspicion |
| **MRI-invisible PCa** (2025) | Détection du cancer invisible | 150 patients PI-RADS ≤ 3, 1 475 images | Non | — | 70 % avec Gleason ≥ 7. L'angle clinique le plus fort |
| **Dictionnaire radiomique PM1.0** (2025) | Stratification du risque | Lésions T2W/DWI/ADC | n/a | **IJIM exclusive** | Exactitude 0,78 ± 0,01 ; relie chaque caractéristique radiomique à un signe PI-RADS. Pas de redistribution du dictionnaire sans accord |
| **Dai et al. 2026** | Grading (≤3+4 vs ≥4+3) | 225 patients multicentriques | n/a | Elsevier | AUC 0,975 entraînement / 0,854 test interne / **0,795 validation externe** — l'écart de 0,18 est la norme |
| **Ordinal multimodal** (CVPRW 2026) | Grading ISUP ordinal | 1 700 patients (test 255) | Non | — | AUC 0,817, QWK 0,612, Brier 0,168 ; évite 19,2 biopsies/100 patients à ~95 % de sensibilité |
| **Gleason2019 / SICAPv2 / TCGA-PRAD** | Grading histologie | Voir §5 | Non | Variable | Voir §5 |
| **MedSAM / MedSAM2** | Seg pilotable | 15 modalités | Oui (Apache-2.0 / restreint) | Apache-2.0 | Production rapide d'un masque de lésion pour valider une carte de chaleur |
| **ImageNet backbones** | Classification par transfert | ImageNet-1k | Oui | BSD-3 / Apache-2.0 | Chemin le plus rapide vers un prototype |

---

## 4. Dépôts GitHub — ce qu'ils contiennent réellement

### 4.1 Classification

| Catégorie | Définition |
|---|---|
| **A** | Code **+** poids **+** documentation — utilisable immédiatement |
| **B** | Code **ou** poids présent, mais il manque quelque chose d'essentiel |
| **C** | Code partiel ou notebook, incomplet, ou sans les poids |
| **D** | Référence scientifique uniquement — aucun artefact exploitable |

### 4.2 Le catalogue vérifié

**Catégorie A — très facilement réutilisable**

| Dépôt | Licence | Contenu vérifié | Poids |
|---|---|---|---|
| [`DIAGNijmegen/picai_unet_gc_algorithm`](https://github.com/DIAGNijmegen/picai_unet_gc_algorithm) | Apache-2.0 (11 357 o) | 33 fichiers : `weights/unet_F0.pt`→`F4.pt`, Dockerfile, `process.py` (11 253 o), scripts build/export/test, `test/` avec exemples `.mha` | **OUI, dans le dépôt** (127 229 381 o par pli, via LFS) |
| [`bamf-health/aimi-prostate-mr`](https://github.com/bamf-health/aimi-prostate-mr) | MIT (1 068 o) | README (5 586 o) + LICENSE | **OUI** — Zenodo 8290093, `Task788_Prostate.zip` = 1 675 047 442 o, **CC BY 4.0** |
| [`MIC-DKFZ/nnUNet`](https://github.com/MIC-DKFZ/nnUNet) | Apache-2.0, 8 981 étoiles | Pipeline complet, CLI, documentation | **OUI** pour la prostate (AIMI) |
| [`bowang-lab/MedSAM`](https://github.com/bowang-lab/MedSAM) | Apache-2.0 (code **et** poids) | `MedSAM_Inference.py`, `gui.py`, `tutorial_quickstart.ipynb`, scripts d'entraînement | **OUI** — `medsam_vit_b` + HF `flaviagiammarino/medsam-vit-base` |
| [`bowang-lab/MedSAM2`](https://github.com/bowang-lab/MedSAM2) | Apache-2.0 (code) | `sam2/`, `app.py`, `notebooks/`, `training/` | **OUI** (poids : recherche/enseignement) |
| [`jacobgil/pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) | MIT, 12 998 étoiles | 25 fichiers XAI, 7 notebooks | n/a |
| [`pytorch/captum`](https://github.com/pytorch/captum) | BSD-3-Clause, 5 712 étoiles | 25 fichiers XAI, 20 notebooks | n/a |
| [`shap/shap`](https://github.com/shap/shap) | MIT | Implémentation de référence de l'article SHAP | n/a |
| [`tensorflow/tcav`](https://github.com/tensorflow/tcav) | Apache-2.0 | `tcav/`, 2 notebooks | n/a |
| [`pytorch/vision`](https://github.com/pytorch/vision) (+ timm) | BSD-3 / Apache-2.0 | Catalogue de modèles | **OUI** (ImageNet) |

**Catégorie B — réutilisable**

| Dépôt | Licence | Contenu | Ce qui manque |
|---|---|---|---|
| [`DIAGNijmegen/picai_baseline`](https://github.com/DIAGNijmegen/picai_baseline) | Apache-2.0 | 142 fichiers : 3 références (U-Net, nnU-Net, nnDetection), découpages 5 plis, docs par modèle | Poids ; mais les découpages publiés sont l'élément le plus dur à reproduire |
| [`DIAGNijmegen/picai_labels`](https://github.com/DIAGNijmegen/picai_labels) | **CC BY-NC 4.0** (19 895 o) | README (18 002 o) documentant contours experts, contours IA, scores PI-RADS, répartition PI-RADS et ISUP | n/a — c'est la référence d'annotation |
| [`DIAGNijmegen/picai_nnunet_semi_supervised_gc_algorithm`](https://github.com/DIAGNijmegen/picai_nnunet_semi_supervised_gc_algorithm) | Apache-2.0 | 33 fichiers : 3 entraîneurs nnU-Net personnalisés, Dockerfile, `process.py` | Poids. Branche `master` |
| [`Project-MONAI/model-zoo`](https://github.com/Project-MONAI/model-zoo) | Apache-2.0 | Bundles (dont `brats_mri_segmentation`) | **Aucun bundle prostate** |
| [`Project-MONAI/MONAI`](https://github.com/Project-MONAI/MONAI) | Apache-2.0 | UNETR, SwinUNETR, SegResNet | Idem |
| [`mahmoodlab/CLAM`](https://github.com/mahmoodlab/CLAM) | MIT (à vérifier) | Référence MIL | Non inspecté en détail |
| [`Kaggle PANDA`](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment) | CC BY-SA-NC 4.0 | Compétition + jeu de développement complet | Test final détenu par Kaggle |

**Catégorie C — partiellement réutilisable**

| Dépôt | Licence | Contenu | Limites |
|---|---|---|---|
| [`DBO-DKFZ/GleasonXAI`](https://github.com/DBO-DKFZ/GleasonXAI) | MIT (1 066 o) | 80+ fichiers : code de recherche complet, 7 configs de modèle, tests, notebooks d'évaluation, calcul du kappa de Fleiss | **Aucun poids** ; le jeu de 1 015 TMA est à obtenir |
| [`AlexNmSED/DSPA-MIL`](https://github.com/AlexNmSED/DSPA-MIL) | MIT | 44 fichiers | Aucun poids |
| [`Kentaroy47/Kaggle-PANDA-1st-Place-Solution`](https://github.com/Kentaroy47/Kaggle-PANDA-1st-Place-Solution) | LICENSE de 19 346 o (type non vérifié en détail) | README (7 594 o) | **[!] Clone blobless non concluant** : l'arborescence n'a pas pu être vérifiée. À contrôler avant de s'y fier |

**Catégorie D — référence uniquement**

| Dépôt | Réalité |
|---|---|
| [`NabaviLab/Prostate-Cancer`](https://github.com/NabaviLab/Prostate-Cancer) | **[!] VÉRIFICATION NÉGATIVE** : ni README ni LICENSE à la racine sur `main`. Cité par arXiv:2212.12844 comme fournisseur de code, mais rien n'est vérifiable. **Ne construisez pas votre projet dessus.** |
| StructMIL (npj Digit Med 2026) | Aucun dépôt de code localisé. Article seulement |
| Modèle PI-RADS de Hamm et al. | Ni code ni poids identifiés. **La référence clinique** |
| Dictionnaire PM1.0 | Aucun dépôt ; licence d'article exclusive |
| ProstAttention-Net | Aucun dépôt de code ni de poids |
| U-Net / AH-Net (Abdom Radiol 2022) | Aucun dépôt de code ni de poids |
| Corrélation IRM ↔ histologie macroscopique (PCA-Rad) | Aucun dépôt ; fournit la vérité histologique |

### 4.3 Synthèse des licences

| Licence | Nombre | Réutilisation dans un projet étudiant |
|---|---|---|
| MIT | 3 | Sans restriction |
| Apache-2.0 | 6 | Sans restriction + octroi de brevet |
| BSD-3-Clause | 1 | Sans restriction |
| **CC BY 4.0 (poids AIMI prostatique)** | 1 | Permissive avec attribution — **les seuls poids prostatiques ouverts** |
| **CC BY-NC 4.0 (PI-CAI)** | 1 | **NON COMMERCIAL** — structurant |
| **CC BY-SA-NC 4.0 (PANDA)** | 1 | **NON COMMERCIAL + partage à l'identique** |
| CC BY 3.0 (PROSTATEx, TCGA-PRAD, PROMISE12) | 3 | Permissive avec attribution |
| **Aucune licence exploitable** | 3 | Public ≠ ouvert : lecture seule |

---

## 5. Jeux de données

### 5.1 PI-CAI — le jeu de référence

| Champ | Valeur |
|---|---|
| Type | IRM biparamétrique prostatique (T2W axial, ADC, DWI à haut b-value), volumes 3D |
| Volume | **1 500 examens** de **1 476 patients**, 3 centres néerlandais (Radboud, UMC Groningen, Ziekenhuis Groep Twente), 2012-2021 |
| Classes | 2 au niveau patient : **1 075** bénins/indolents, **425** csPCa |
| Masques | **OUI** — contours de lésions csPCa par expert humain pour **1 295/1 500 (86 %)** ; contours automatisés par IA (Bosma et al. 2022) de la glande entière et des lésions pour les 1 500 cas |
| Annotations | Contours experts + **scores PI-RADS** attribués en pratique courante pour les **1 500 cas** (Pooch et al. 2025, publiés le 13 juillet 2026) |
| Répartition lésions | 1 087 lésions positives : PI-RADS 3 → 246 (23 %), PI-RADS 4 → 438 (40 %), PI-RADS 5 → 403 (37 %). 776 lésions classées ISUP : ISUP 1 → 311 (40 %), 2 → 260 (34 %), 3 → 109 (14 %), 4 → 41 (5 %), 5 → 55 (7 %) |
| Split | 5 plis de téléchargement officiels ; **328 cas proviennent de ProstateX** |
| Licence | **CC BY-NC 4.0** (fichier LICENSE de 19 917 o) |
| Accès | Gratuit — Zenodo, 113 277 téléchargements, 38 585 uniques |
| Taille | 5 archives de 5,16 à 5,56 Go, ~27 Go au total |
| Format | NIfTI (.nii.gz) + annotations JSON |
| Recommandé | **OUI** |

**Les deux réserves à reproduire dans votre rapport :**
1. **Non commercial** (CC BY-NC 4.0).
2. « *Le jeu de données original ne fournissait qu'un découpage entraînement/test et n'incluait aucune information sur les patients, les séquences ou les tranches, ce qui rend impossible de relier de façon définitive les images au même sujet… l'indépendance complète au niveau du sujet ne peut pas être garantie.* » Les cas de ProstateX sont inclus : ne pas utiliser ProstateX en plus.

### 5.2 PROSTATEx — déprécié, à ne pas utiliser en parallèle

346 sujets, 349 études, 18 321 séries, 309 251 images, 16,67 Go, **CC BY 3.0**. La TCIA le déclare explicitement : « *PROSTATEx has been superseded by PI-CAI: The ProstateX dataset (both training and testing cases) have been included in the PI-CAI Public Training and Development dataset. As such, ProstateX as a benchmark has been deprecated and is superseded by the PI-CAI challenge.* » Utile comme cohorte externe, mais contenu dans PI-CAI. Collections dérivées : PROSTATEx-Seg-Zones (98 sujets), PROSTATEx-Seg-HiRes (66), ProstateX-Targets.

### 5.3 PANDA — le plus grand jeu de grading prostatique

10 616 WSI H&E de **2 113 patients**, deux centres (Radboud 5 160 / Karolinska 5 456), ISUP 0-5 + Gleason primaire/secondaire, et surtout les **`train_label_masks`** : annotations pixel partielles. Radboud annote les glandes individuellement (0 fond, 1 stroma, 2 épithélium bénin, 3/4/5 épithélium cancéreux) ; Karolinska annote des régions (0 fond, 1 bénin, 2 cancéreux). **CC BY-SA-NC 4.0**, métrique : kappa quadratique pondéré, test final détenu par Kaggle (545 biopsies en validation interne aveugle).

Ces masques d'annotation sont une **ressource d'explicabilité fournie par les données** : ils disent quelles parties de l'image ont conduit au grade, ce qui permet de valider une carte d'attention MIL quantitativement. Few-shot : la convention d'annotation diffère entre les deux centres, ce qui rend l'harmonisation non triviale.

### 5.4 Les autres

| Jeu | Contenu | Licence | Recommandé |
|---|---|---|---|
| **Gleason 2019** | 244 cores d'entraînement + 88 de test ; 6 anatomopathologistes en annotation pixel, étiquette finale par vote majoritaire ; 4 des 6 ont annoté les 331 images, les 2 autres seulement 191 et 91 | CC BY 4.0 | PEUT-ÊTRE — petit mais annoté au pixel par 6 experts ; 244 images seulement, donc surapprentissage quasi garanti |
| **SICAPv2** | ~10 340 patchs 512×512, score global + grade par patch | CC BY 4.0 | PEUT-ÊTRE — bon complément de Gleason 2019 |
| **TCGA-PRAD** | 723 lames de cancer, 490 patients, **sans lame bénigne** | CC BY 3.0 | MAYBE — utile en validation externe de robustesse |
| **Prostate-MRI-US-Biopsy** | 842 sujets, IRM + échographie + biopsie | CC BY 4.0 | MAYBE — appariement lourd |
| **PROMISE12 / PCA-Rad** | IRM + **lame macroscopique de prostatectomie** appariées | CC BY 3.0 | **OUI** comme vérité histologique : la seule façon de prouver qu'une carte de chaleur pointe sur la bonne lésion |
| **AIMI Annotations** | Segmentations DICOM-SEG prostatiques + cotations Likert par radiologue | CC BY 4.0 | **OUI** comme ressource d'évaluation |
| **QIN-PROSTATE** | 15 sujets avant/après ablation | CC BY 4.0 | NON — trop petit |
| **DiagSet** | WSI prostatiques annotés par grade de Gleason | non vérifiée | MAYBE — licence et accès à clarifier |
| **TCGA-PRAD (IRM)** | 164 cas avec IRM sur 500 | CC BY 3.0 | MAYBE — sans vérité de lésion |

---

## 6. IA explicable — la section centrale

### 6.1 La distinction à critically analyser

#### Niveau 1 — « où le modèle regarde » (localisation)

**Grad-CAM, Grad-CAM++, LayerCAM, Score-CAM, Eigen-CAM, cartes d'attention, Seg-Grad-CAM.**

Mécanisme : rétropropager le gradient du score cible dans une couche convolutionnelle, moyen-pooler, pondérer les activations, ré-échantillonner, ReLU.

**Ce que cela établit :** une *corrélation* entre une région et la sortie.

**Ce que cela n'établit pas**, et la prostate fournit la formulation la plus nette de cette limite. L'étude de 2022 sur la segmentation tumorale corrélée à l'histologie macroscopique écrit :

> « *Les cartes Grad-CAM décrivent la localisation de l'attention du CNN, alors que la segmentation CNN ne considère que l'activation de classe résultante, qui peut provenir de n'importe où dans le champ réceptif.* »

Donc : **une carte de chaleur et un masque ne disent pas la même chose**, même quand ils se recouvrent. Une carte qui surligne la lésion ne prouve pas que le réseau a décidé *grâce à* elle. Et l'éditorial de *Radiology* qui accompagne Hamm et al. le formule comme un impératif : « *Don't Trust, Verify* ».

**Le mode de défaillance clinique spécifique :** en IRM prostatique, une carte de chaleur qui met en évidence le **bord de l'image**, la **vessie** ou la **zone périphérique** saine est plausible et ne dit rien. Un radiologue le voit immédiatement ; une galerie de cartes de chaleur naïve, non.

**Mais la prostate a aussi répondu à cette objection**, et c'est important : l'étude de 2019 a été la première à **quantifier** le recouvrement entre zones focées par le réseau et cibles identifiées par les radiologues (PI-RADS) *et* par les anatomopathologistes. Ce protocole est directement réutilisable.

#### Niveau 2 — « pourquoi il a décidé » (attribution)

**Integrated Gradients, SHAP, DeepLIFT, LIME, occlusion / ablation de caractéristiques.**

Deux propriétés à vérifier :
1. **Complétude** — les attributions somment à la différence de score. IG et SHAP le garantissent ; Grad-CAM et LayerCAM non.
2. **Fidélité** — supprimer la zone mise en évidence change-t-elle la prédiction ? *Testable*, donc falsifiable.

**La prostate rend ce test réalisable, et c'est son avantage décisif.** Le modèle PI-CAI de référence produit une **carte de détection continue**, et 1 295 cas portent des **contours d'expert**. Vous avez donc deux vérités terrain indépendantes pour la même image :

- **LocAcc** contre la carte de détection (le modèle est-il cohérent avec lui-même ?)
- **LocAcc** contre les contours experts (le modèle et son explication designate-ils la bonne lésion ?)
- **Faith** : supprimer la zone la plus saillante fait chuter la probabilité de csPCa de combien ?

C'est le protocole de 2022, transposé de l'histologie macroscopique à PI-CAI, et il n'a jamais été fait à cette échelle.

#### Niveau 3 — « quels concepts a-t-il utilisés » (niveau concept)

**Signes PI-RADS, patterns de Gleason, attribution par prototypes.**

C'est ici que la prostate devance nettement tout le reste, et il faut comprendre pourquoi : **le lexique clinique standardisé existe déjà**. PI-RADS v2 définit 14 signes observables ; le système de Gleason définit 5 patterns architecturaux. Ce ne sont pas des concepts à inventer, ce sont des concepts que les médecins utilisent déjà.

Hamm et al. ont démontré que la machine peut écrire dans ce lexique, et l'évaluation a été faite par des experts sur 1 352 cas. C'est le seul endroit de tout ce dossier où l'IA explicable a Improve la **décision** plutôt que la compréhension.

**Le niveau 4 — l'explication intrinsèque** (histopathologie). Le réseau ne prédit pas un grade, il prédit une **description**. L'explication est la sortie. GleasonXAI obtient de meilleurs résultats de segmentation en étant entraîné sur les explications (0,713) qu'en étant entraîné sur le grade (0,691) — parce que les labels souples encodent l'incertitude inter-annotateur, que le grade ne contient pas.

**Le niveau 5 — l'explication par prototypes** (frontière actuelle). L'attention MIL a une faiblesse documentée : un poids d'attention élevé ne désigne pas un prototype représentatif. Le pooling prototypique y répond, et la validation est faite : les prototypes à attention > 0,5 localisent des structures glandulaires bien définies, **en accord avec les masques annotés**.

### 6.2 Recommandations pratiques

1. **Montrez les trois niveaux.** Une carte Grad-CAM (où), une attribution Integrated Gradients (pourquoi), et les signes PI-RADS (compte rendu). Un modèle prostatique qui n'explique qu'en pixels n'a pas fait le travail.
2. **Mesurez, n'affichez pas.** Sans LocAcc et sans Faith, une carte de chaleur n'est qu'une illustration. La prostate est le seul domaine où c'est faisable à grande échelle.
3. **Attention ≠ prototype.** Si vous travaillez en histologie, preferez le pooling prototypique, ou au moins validez l'alignement de l'attention avec les annotations d'experts — c'est exactement ce que fait DSPA-U-MIL.
4. **N'utilisez pas le score brut sans la population.** Un AUROC de 0,88 en population de suspicion (PI-CAI) devient 0,83 en population de dépistage (G2). C'est l'argument le plus fort que vous tarik face à un lecteur critique.

### 6.3 Tableau complet des méthodes

| Méthode | Famille | Principe | Compatible | Visualisation | Avantages | Limites | Coût | Pertinence prostatique | Code |
|---|---|---|---|---|---|---|---|---|---|
| **Grad-CAM** | Gradient | Gradient du score cible dans la dernière couche conv, moyenné, somme pondérée des activations | CNN (ResNet, EfficientNet, U-Net, AH-Net, DA-UNet) ; ViT/Swin via couche cible | Carte de chaleur grossière sur l'IRM axiale T2W | 1 fwd+bwd ; **la méthode de référence prostatique** ; vérification quantitative existante | Grossière ; saturante avec la perte de Dice ; **non causale** | 1 fwd+bwd | **TRÈS ÉLEVÉE** — 4 travaux prostatiques l'utilisent, dont 2 avec vérification terrain | [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) |
| **Explication par signes PI-RADS** | Par concepts | Le modèle écrit sa justification dans le lexique PI-RADS (14 signes) | CNN de détection de lésion + classifieurs de signes par couche | Carte de chaleur **+ compte rendu textuel** | **Validée cliniquement** ; 80 % d'explications confirmées ; confiance des non-experts 4,1 vs 3,4 | Données d'annotation de signes coûteuses ; 14 signes et couches non entièrement publiés ; pas de code | Faible en inférence, élevé en annotation | **TRÈS ÉLEVÉE** — l'état de l'art clinique | [Radiology 2023](https://pubs.rsna.org/doi/10.1148/radiol.222276) |
| **Explication par terminologie d'anatomopathologiste** | Par concepts (intrinsèque) | Le U-Net segmente des DESCRIPTIONS de patterns, pas le grade | U-Net de segmentation de patterns | Carte + vocabulaire | **Intrinsèque** : pas de méthode post-hoc à contester ; labels souples → incertitude inter-annotateur ; Dice 0,713 > 0,691 | Histopathologie ; pas de poids ; jeu de données à obtenir | Faible | **TRÈS ÉLEVÉE** (histopathologie) | [GleasonXAI](https://github.com/DBO-DKFZ/GleasonXAI) (MIT) |
| **Cartes d'attention MIL** | Attention (intrinsèque) | Une WSI = un sac de patchs ; l'attention pondère les instances ; ces poids *sont* l'explication | MIL ABMIL/CLAM/DSMIL/TransMIL | Superposition WSI, glandes en évidence | **Natif** ; seul mode viable à l'échelle WSI ; standard de la pathologie computationnelle | Un poids d'attention élevé ne désigne pas un prototype ; pas de validation clinique prospective | Élevée (IO) | **ÉLEVÉE** — dominant en histologie | [DSPA-MIL](https://github.com/AlexNmSED/DSPA-MIL) ; [CLAM](https://github.com/mahmoodlab/CLAM) |
| **Attribution par prototypes** | Par prototypes (intrinsèque) | Pooling par similarité cosinus au prototype le plus proche | MIL à pooling prototypique | Prototypes **nommés**, superposables | Répond à la critique de l'attention ; localise des structures en accord avec les masques | Plus complexe ; pas de dépôt pour le grading adaptatif | Comparable à MIL | **ÉLEVÉE** — la frontière actuelle | [npj 2026](https://www.nature.com/articles/s41746-025-02289-4) ; [arXiv:2603.04947](https://arxiv.org/abs/2603.04947) |
| **Integrated Gradients** | Gradient (axiomatique) | Intègre les gradients le long du chemin droit de la référence vers l'entrée | Tout modèle différentiable, U-Net de segmentation inclus | Attribution signée par voxel | **Garantie de complétude** : « ce voxel a relevé le logit csPCa de 0,8 » est arithmétiquement vrai | Sensible à la référence et au chemin ; lent en 3D ; bruité | n_steps fwd+bwd | **ÉLEVÉE** — le complément naturel de la carte de détection continue | [captum](https://github.com/pytorch/captum) |
| **SHAP** | Perturbation + axiomatique | Valeurs de Shapley ; unicité dans la classe additive | Tout modèle via un wrapper | Superposition diffuse | Garanties théoriques les plus fortes ; **déjà utilisé** pour le grading prostatique (Dai et al. 2026) | Exponentiel à l'exact ; lent ; hors de portée sur WSI | Centaines d'évaluations | **MOYENNE À ÉLEVÉE** | [shap/shap](https://github.com/shap/shap) |
| **LIME** | Perturbation | Surrogate interprétable local | Toute boîte noire | Segmentaire | Indépendant du modèle ; lisible | Superpixels sans correspondance clinique en IRM ; lent | ~1000+ fwd | FAIBLE à MOYENNE | [lime](https://github.com/marcotcr/lime) ; Captum |
| **Occlusion / Feature Ablation** | Perturbation | Masque des régions, mesure la chute du score | Tout modèle, tout rang | Grille de deltas | **L'outil de validation** : supprimer la zone Grad-CAM et mesurer la chute de probabilité csPCa | Lent ; fenêtre et valeur de remplissage dominent | N_fenetres fwd | **TRÈS ÉLEVÉE** grâce aux deux vérités terrain | [captum](https://github.com/pytorch/captum) |
| **RISE** | Perturbation | Masques binaires aléatoires, régression | Modèle différentiable | Carte lisse | Robuste, lisse | Milliers de passes ; né de la classification 2D | Très élevé | MOYENNE | [arXiv:1806.07421](https://arxiv.org/abs/1806.07421) |
| **Seg-Grad-CAM** | Gradient (adaptation segmentation) | Gradient du score sommé sur la région prédite | U-Net, nnU-Net, DA-UNet, AH-Net, ProstAttention-Net | Moyenne, superposée au masque | **Aligné sur la segmentation** → accord mesurable | Variantes concurrentes ; pas de protocole universellement adopté | 1 fwd+bwd | **ÉLEVÉE** — avec la distinction conceptuelle de l'article de 2022 | [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) ; [PMC8976981](https://pmc.ncbi.nlm.nih.gov/articles/PMC8976981/) |
| **Attention rollout / flow** | Attention | Propagation multiplicative de l'attention à travers les couches | ViT, Swin, TransUNet, SwinUNETR, ProstAttention-Net | Grille de patchs | Aucune modification, aucun gradient ; la seule explication sensée pour un transformateur | Poids d'attention non validés causalement ; grossier | Bon marché | MOYENNE | [arXiv:2005.00928](https://arxiv.org/abs/2005.00928) |
| **TCAV** | Par concepts | CAV par concept et par couche + test statistique | Tout réseau | Score signé par concept | Les signes PI-RADS forment un jeu de concepts **prêt à l'emploi** | TF ; statistiques exigeant assez d'exemples ; hors de portée sur WSI | Modéré | MOYENNE — le principe est validé par Hamm et al. | [tcav](https://github.com/tensorflow/tcav) |
| **Dictionnaire radiomique PM1.0** | Par concepts / features | Chaque caractéristique radiomique rattachée à un signe PI-RADS | Modèle tabulaire + classifieur simple | Tableau de contributions | Les variables du modèle sont des **signes cliniques** ; features retenues cliniquement sensibles (90e percentile T2W, run entropy ADC) | Pas de code ; licence d'article **exclusive** ; exactitude 0,78 | Faible | **ÉLEVÉE** — l'explicabilité la plus lisible, mais avec un handicap de licence | [PMID 40608191](https://pubmed.ncbi.nlm.nih.gov/40608191/) |
| **Validation par étude de lecteurs** | Validation | Étude multi-lecteurs : confiance, temps de lecture, justesse | Tout modèle destiné au clinique | n/a | **Le seul niveau qui prouve qu'une explication change une décision.** Protocole PI-CAI fourni | Coûteux en effort humain | Élevée | **TRÈS ÉLEVÉE** | [pi-cai.grand-challenge.org](https://pi-cai.grand-challenge.org/) |

### 6.4 Le paysage XAI prostatique en un coup d'œil

- **Côté IRM** : Grad-CAM domine, avec deux études qui l'ont **quantitativement vérifiée** (2019, 2022) et une validation clinique complète (Hamm 2023, Radiology 2024).
- **Côté histologie** : l'attention MIL et le pooling prototypique dominent, avec une exigence de validation plus élevée (alignement sur les annotations d'experts).
- **Le niveau concept** existe et est validé en IRM (PI-RADS) ; il est en développement en histologie (terminologie d'anatomopathologiste, prototypes).
- **Ce qui manque** : une validation quantitative unifiée à l'échelle de PI-CAI. C'est exactement là que se situe votre contribution.

---

## 7. Modèles déjà explicables (MODÈLE → PRÉDICTION → XAI → SORTIE)

| # | MODÈLE | PRÉDICTION | XAI | SORTIE | État |
|---|---|---|---|---|---|
| 1 | **U-Net de référence PI-CAI** | csPCa par lésion + probabilité au cas | Grad-CAM + IG + occlusion (à écrire) | Carte de chaleur sur la carte de détection + LocAcc + Faith | **A — poids dans le dépôt** |
| 2 | **Modèle PI-RADS de Hamm et al.** | PCa / csPCa / bénin + 14 signes | **Explication textuelle en signes PI-RADS** + carte interactive | Un compte rendu radiologique | **Référence clinique** — code/poids non localisés |
| 3 | **Radiologie 2024 (DL 3D)** | csPCa au patient | Grad-CAM | Activation sur la lésion dans 35/38 et 56/58 vrais positifs | D (article) |
| 4 | **Étude 2019 (307 IRM)** | Cancer / non-cancer | Grad-CAM | Zones focées comparées **quantitativement** aux cibles PI-RADS et aux localisations d'anatomopathologiste | D (article) |
| 5 | **PCA-Rad 2022 (U-Net, 122 patients)** | Masque de lésion | Grad-CAM | Corrélation avec l'histologie macroscopique, IoU max à δ = 0,5 | D (article) |
| 6 | **GleasonXAI (U-Net)** | Descriptions de patterns de Gleason | **Intrinsèque** | Carte de descriptions en terminologie d'anatomopathologiste | **C — code complet MIT, pas de poids** |
| 7 | **CLAM / ABMIL / TransMIL** | Grade ISUP 0-5 | **Attention MIL native** | Carte d'attention sur la WSI, glandes en évidence | C (poids à réentraîner) |
| 8 | **StructMIL / DSPA-U-MIL** | Grade ISUP | **Prototypes nommés / gating guidé par concepts** | Prototypes localisés, alignement vérifié avec les annotations | C (MIT) |
| 9 | **ProstAttention-Net** | Masque par grade d'agressivité | Attention multi-étiquette | Une carte par grade | D (article) |
| 10 | **Dictionnaire PM1.0** | Risque faible / élevé | Caractéristiques radiomiques rattachées à PI-RADS | Tableau de contributions lisibles | D (article, licence exclusive) |
| 11 | **Dai et al. 2026** | Gleason ≤3+4 vs ≥4+3 | SHAP + corrélation + médiation | Contribution de chaque caractéristique | D (article) |
| 12 | **MedSAM** | Masque piloté | l'invite | Masque + boîte tracée | **A — poids Apache-2.0** |
| 13 | **nnU-Net à annotation guidée (Bosma)** | Détection csPCa voxel | — | — | B (pipeline Apache-2.0) |
| 14 | **AIMI Task788_Prostate** | Glande prostatique | — | — | **A — poids CC BY 4.0** |
| 15 | **ProstateX-2 (Gleason Grade Group)** | Grade GG | Biomarqueurs quantitatifs IRM | Score ordinal | D (défi clos) |
| 16 | **Langkilde 2025 (G2)** | csPCa | — | — | D (cohorte non publique) |
| 17 | **MRI-invisible PCa 2025** | csPCa invisible | — | — | D (article) |
| 18 | **Ordinal multimodal 2026** | ISUP ordinal | Variables cliniques + radiomiques zonales | Score ordinal + QWK | D (article) |

**Lecture de ce tableau.** Sur les 18 lignes, **quatre** offrent une chaîne complète et immédiatement exécutable : ① (poids dans le dépôt), ⑥ et ⑧ (code complet sous licence permissive, à entraîner), ⑫ (MedSAM). Les lignes ② à ⑤ sont des **références scientifiques** — à citer, à mesurer, à dépasser, pas à exécuter. C'est la situation inverse de l'imagerie cérébrale : ici, la littérature XAI de qualité est en avance sur les artefacts publiés, ce qui est une bonne nouvelle pour un projet de recherche et une contrainte pour un projet d'ingénierie.

---

## 8. Littérature scientifique

28 articles sont catalogués dans `research-data.json` avec auteurs, revue, modèle, jeu de données, métriques, méthode XAI, lien de code, état des poids et limites déclarées. Les plus structurants :

### 8.1 Fondamentaux de l'IA explicable

| Travail | Année | Pourquoi il compte ici |
|---|---|---|
| **Grad-CAM** (Selvaraju et al., ICCV 2017) | 2017 | La méthode réellement utilisée en imagerie prostatique. L'extension 2019 ajoute les sanity checks |
| **SHAP** (Lundberg & Lee, NIPS 2017) | 2017 | Unicité des attributions additives — la méthode « pourquoi » du grading prostatique |
| **LIME** (Ribeiro et al., KDD 2016) | 2016 | Substituts locaux |
| **Integrated Gradients** (Sundararajan et al., ICML 2017) | 2017 | Complétude |
| **TCAV** (Kim et al., ICML 2018) | 2018 | Le cadre par concepts — dont les signes PI-RADS sont l'instance prostatique |

### 8.2 Fondamentaux de la segmentation

| Travail | Année | Rôle |
|---|---|---|
| **nnU-Net** (*Nature Methods* 18:203) | 2021 | Le framework utilisé par PI-CAI et AIMI ; s'auto-configuré en U-Net 3D à 5 niveaux pour la prostate |
| **U-Net** (arXiv:1505.04597) | 2015 | L'ancêtre |
| **PI-RADS v2** (PMID 31627826) | 2019 | **Le lexique qui rend possible l'explication par concepts** |

### 8.3 Les articles prostatiques qui structurent le projet

| Travail | Modèle | XAI | Point clé |
|---|---|---|---|
| **Saha et al., Lancet Oncol 2024** | 5 algorithmes d'IA réentraînés sur 9 107 IRM | Carte de détection + score de risque | L'étude de référence ; **non-infériorité des IA par rapport aux radiologues** ; étude de lecteurs 50+ radiologues sur 400 cas |
| **Hamm et al., Radiology 2023** | CNN + 14 signes PI-RADS | **Explication textuelle en concepts** | **LA référence clinique** : 80 % d'explications confirmées, +58 s gagnés, confiance 3,4 → 4,1 |
| **Chapiro, Radiology 2023** (éditorial) | — | — | « Don't Trust, Verify » : la posture critique à adopter |
| **Radiology 2024** | DL 3D | Grad-CAM | Performance égale aux radiologues ; Grad-CAM sur la lésion dans 35/38 et 56/58 |
| **Bosma et al. 2021 / 2023** | nnU-Net, DA-UNet | Annotation guidée par compte rendu | Explicabilité des DONNÉES : compter les lésions du compte rendu pour guider l'annotation |
| **Bulten et al., Nature Medicine 2022** | MIL | Attention native | PANDA ; le référentiel de grading |
| **Mittmann et al. 2024/2025** | U-Net | **Intrinsèque** | Explications en vocabulaire d'anatomopathologiste ; 1 015 TMA annotés par 54 experts |
| **Étude 2019** (PMC6920905) | CNN | Grad-CAM | **Première vérification quantitative** du recouvrement avec PI-RADS et l'histologie |
| **PCA-Rad 2022** (PMC8976981) | U-Net | Grad-CAM | Corrélation à l'histologie macroscopique ; **la distinction conceptuelle carte ≠ masque** |
| **Langkilde et al. 2025** | nnU-Net | — | AUROC 0,83 **en population de dépistage** |
| **Dai et al. 2026** | ML interprétable | SHAP | AUC 0,975 → 0,795 en validation externe : l'écart normal |
| **PM1.0 (2025)** | Classifieurs interprétables | Concepts | Relie radiomique et PI-RADS ; licence exclusive |
| **StructMIL / DSPA-U-MIL (2026)** | MIL à prototypes | Prototypes, gating | +7,4 % de kappa inter-scanner ; alignement vérifié avec les annotations |
| **Étude 2019/2022 (histopathologie)** | MIL faiblement supervisé | Attention | Diagramme 3 niveaux |

### 8.4 Surveys et références de domaine

| Référence | Lien | Usage |
|---|---|---|
| Gipiškis et al., *XAI in Image Segmentation* | `arXiv:2405.01636` | Taxonomie des variantes Seg-* et lacune d'évaluation |
| Getamesay et al., *XAI in Biomedical Image Analysis* | `arXiv:2507.07148` | Quelle méthode pour quelle modalité |
| Bhati et al., *XAI techniques for medical imaging* | PMC11508748 | Le manque de protocole d'évaluation standard |
| **PI-CAI (page du défi)** | `pi-cai.grand-challenge.org` | Protocoles de données, d'évaluation et de lecteur |
| Gleason2019 (défi) | `gleason2019.grand-challenge.org` | Référentiel de grading |

---

## 9. Audit de performance — quels chiffres croire

### 9.1 L'écart de population : la première règle

| Population | Contexte | AUROC |
|---|---|---|
| Suspicion (PI-CAI) | Patient biopsié pour suspicion | ~0,88 |
| **Dépistage (G2)** | Patient d'une population de dépistage | **0,83** |

Tout chiffre d'IA prostatique doit être rapporté avec sa population. C'est l'argument le plus solide face à un lecteur critique, et il est documenté.

### 9.2 Verdicts sur des chiffres précis

| Affirmation | Verdict | Raisonnement |
|---|---|---|
| **U-Net de référence PI-CAI : AUROC 0,878, score 0,718** | **FIABLE** | Classement officiel sur une cohorte de test cachée de 1 000 scans, réentraînement par les organisateurs sur 9 107 examens, avec intervalles de confiance publiés. C'est un **référentiel**. |
| **Version semi-supervisée : score 0,734, AUROC 0,881** | **FIABLE** | Idem, même protocole. |
| **nnU-Net : 0,81 / 0,888 / 0,732** | FIABLE avec réserve | Trois variantes de post-traitement sur le même modèle ; le choix de configuration change le résultat de 0,08. |
| **Hamm 2023 : AUROC 0,89 / 0,87, explications à 80 %** | **FIABLE, et c'est le plus solide** | Validation croisée 5 plis, **validation externe**, et surtout **explications confirmées par des experts** avec étude de lecteurs. Le niveau de preuve le plus élevé du dossier. |
| **Radiologie 2024 : performance = celle des radiologues** | **FIABLE** | Test de DeLong, test externe PROSTATEx, Grad-CAM vérifié sur 35/38 et 56/58. Réserve : mono-établissement à l'entraînement. |
| **Saha 2022 : sensibilité 70,9 %, PPV 35,5 %** | FIABLE | Vérité par biopsie IRM-TRUS, deux institutions, découpage strict. **C'est le rappel reality** : la détection de lésions prostatiques a une PPV faible. |
| **Langkilde 2025 : AUROC 0,83** | FIABLE | Mais **population de dépistage** : ne pas le comparer aux chiffres de suspicion. |
| **Dai 2026 : AUC 0,975 (entraînement) → 0,795 (externe)** | FIABLE | L'écart de 0,18 est **la norme**, pas l'exception. Citer le chiffre externe, jamais celui d'entraînement. |
| **Étude 2019 : recouvrement quantifié** | FIABLE et rare | Première vérification quantitative du recouvrement Grad-CAM / cibles expertes. |
| **PCA-Rad 2022 : IoU max à δ = 0,5** | FIABLE mais sur 15 patients | Cohorte d'histologie de 15 patients seulement. |
| **Mittmann 2024 : Dice 0,713 (explications) > 0,691 (grade)** | FIABLE | Résultat contre-intuitif et solide : entraîner sur l'explication est meilleur. |
| **Bulten 2022 (PANDA) : QWK** | FIABLE | Étude pré-enregistrée, validation interne aveugle de 545 biopsies. Mais test final détenu par Kaggle : ne comparez pas hors soumission. |

### 9.3 Les quatre niveaux de preuve

| Niveau | Signification | Exemples prostatiques |
|---|---|---|
| **Référentiel** | Test caché, réentraînement par les organisateurs, IC publiés | Classements PI-CAI (0,718 / 0,734) |
| **Reproductible** | Code, poids et découpage publics et rejouables | U-Net de référence PI-CAI, picai_baseline, AIMI Task788_Prostate |
| **Expérimental** | Une exécution sur un découpage privé | La plupart des articles de grading ; Saha 2022 |
| **Validation clinique** | Étude de lecteurs, validation externe, expertise | **Hamm 2023 (PI-CAI et la presse), Radiology 2024** |

### 9.4 Le protocole que ce projet devrait suivre

1. **Toujours rapporter la population** (suspicion, dépistage, biopsie ciblée).
2. **Ne jamais splitter par tranche.** Sur PI-CAI, le découpage officiel est fourni : respectez-le.
3. **Ne pas mélanger ProstateX et PI-CAI** : 328 cas sont communs, donc chevauchement de patients.
4. **Rapporter la sensibilité ET la PPV ET le nombre de faux positifs par patient.** Une sensibilité de 75 % avec 1,9 faux positif par patient n'est pas un bon outil clinique, et le dire est plus honnête que de l'omettre.
5. **Ajouter la validation externe** si possible (PROSTATEx, ou une population de dépistage).
6. **Mesurer ses propres explications** : LocAcc contre la carte de détection *et* contre les contours experts, plus la fidélité par suppression.

---

## 10. Licences et réutilisation

| Licence | Dépôts / données | Ce que cela signifie pour un projet étudiant / recherche |
|---|---|---|
| **MIT** | `GleasonXAI`, `DSPA-MIL`, `CLAM` | Sans restriction, y compris commercial |
| **Apache-2.0** | `picai_unet_gc_algorithm`, `picai_baseline`, `picai_nnunet_semi_supervised_gc_algorithm`, `MIC-DKFZ/nnUNet`, `Project-MONAI/*`, `bowang-lab/MedSAM` | Sans restriction + octroi de brevet ; conserver LICENSE et NOTICE |
| **BSD-3-Clause** | `pytorch/captum`, `pytorch/vision` | Sans restriction |
| **CC BY 4.0** | **Poids AIMI Task788_Prostate (Zenodo 8290093)**, AIMI Annotations, Gleason 2019, SICAPv2, PROSTATEx, TCGA-PRAD, PROMISE12, Article StructMIL | Permissive avec attribution. **Les poids prostatiques AIMI sont la ressource la plus precious du dossier.** |
| **CC BY-NC 4.0** | **PI-CAI (données + annotations)** | **NON COMMERCIAL.** Bloque toute exploitation industrielle et impose de lister l'attribution |
| **CC BY-SA-NC 4.0** | **PANDA (données + soumissions gagnantes)** | **Non commercial + partage à l'identique** : un dérivé des données doit porter la même licence |
| **IJIM exclusive** | Dictionnaire radiomique PM1.0 | **Pas de redistribution du dictionnaire** sans accord. Attention lors de la rédaction |
| **Aucune licence exploitable** | `NabaviLab/Prostate-Cancer`, et 2 dépôts à contrôler | Public ≠ ouvert : lecture seule |

**La synthèse pratique :** côté **code**, la situation est excellente — l'essentiel de la prostate est en MIT ou Apache-2.0, ce qui est mieux que le corpus cérébral. Côté **données**, la situation est contraignante : PI-CAI et PANDA sont toutes deux non commerciales. Un projet étudiant en recherche est parfaitement couvert ; un projet à visée industrielle ne l'est pas.

---

## 11. Tableaux de comparaison

### 11.1 Solutions

| Solution | Tâche | Modèle | Jeu de données | XAI | Performance | Code | Poids | Licence | Difficulté | Recommandation |
|---|---|---|---|---|---|---|---|---|---|---|
| **U-Net de référence PI-CAI** | Détection csPCa (patient + lésion) | 5× U-Net 3D | PI-CAI v2.0 | Grad-CAM + IG (à écrire) | AUROC 0,878 · score 0,718 | Oui | **Dans le dépôt** (LFS) | Apache-2.0 | Facile à moyen | **★ Choix n°1** |
| **AIMI Task788_Prostate** | Seg. glande | nnU-Net v1 | PI-CAI | Aucune | QA radiologue | mince | **Zenodo, CC BY 4.0** | MIT / CC BY 4.0 | Moyen | ★ Indispensable (ROI) |
| **picai_baseline** | Détection csPCa | U-Net / nnU-Net / nnDetection | PI-CAI | Aucune | 0,81 / 0,888 / 0,732 | Oui (142 fichiers) | Non | Apache-2.0 | Facile | ★ Protocole de référence |
| **nnU-Net semi-supervisé** | Détection csPCa | nnU-Net semi-supervisé | PI-CAI | Aucune | score 0,734 · AUROC 0,881 | Oui | Non | Apache-2.0 | Moyen | ★ Meilleure config |
| **GleasonXAI** | Seg. patterns de Gleason | U-Net explicite | 1 015 TMA (54 experts) | **Intrinsèque** | Dice 0,713 ± 0,003 | Oui (80+ fichiers) | **Non** | MIT | Moyen | ★★ Meilleure explicabilité |
| **Modèle PI-RADS (Hamm)** | Détection + justification | CNN + 14 signes | 1 224 patients | **Concepts PI-RADS** | AUROC 0,89/0,87 · expl. 80 % | **Non** | **Non** | Radiology | Difficile | ★★ **Référence clinique** |
| **Radiologie 2024** | Détection csPCa | DL 3D | Cohorte + PROSTATEx | Grad-CAM | = radiologues | Non | Non | Radiology | Difficile | ★ Benchmark |
| **CLAM / ABMIL / TransMIL** | Grading ISUP | MIL à attention | PANDA, TCGA-PRAD | **Attention native** | État de l'art QWK | Oui | Non | MIT | Moyen | ★ Standard histologie |
| **StructMIL / DSPA-U-MIL** | Grading interprétable | MIL à prototypes | PANDA, Camelyon16 | **Prototypes** | +7,4 % kappa ; +6,7 % QWK | Oui | Non | MIT | Moyen | ★ Frontière |
| **ProstAttention-Net** | Seg. par grade | Attention multi-étiquette | IRM clinique | Attention par grade | Med Image Anal 2022 | Non | Non | Elsevier | Difficile | C |
| **U-Net / AH-Net** | Détection + seg. | U-Net, AH-Net | 525 patients | Aucune | sens. 70,9 % · PPV 35,5 % | Non | Non | Springer | Difficile | C — protocole exemplaire |
| **Dictionnaire PM1.0** | Stratification du risque | Classifieurs interprétables | Lésions T2W/DWI/ADC | **Concepts** | exactitude 0,78 ± 0,01 | Non | n/a | **IJIM exclusive** | Moyen | C |
| **MedSAM** | Seg. pilotable | SAM ViT-B | 15 modalités | L'invite | 86/86 vs SAM | Oui | **Oui** | Apache-2.0 | Moyen | ★ Démo interactive |
| **MedSAM2** | Seg./suivi 3D | SAM 2.1 | 3D + vidéo | L'invite | arXiv 2504.03600 | Oui | Oui (restreint) | Apache-2.0 | Moy. à diff. | Avancé |
| **nnU-Net (générique)** | Segmentation | nnU-Net | PI-CAI + AIMI | Aucune | Auto-configuré | Oui | Oui (prostate) | Apache-2.0 | Moy. à diff. | ★ Référence |
| **MIL à annotation guidée** | Détection csPCa | nnU-Net, DA-UNet | 7 756 IRM | Aucune | FROC, 300 externes | pipeline | Non | Apache-2.0 | Moyen | C — idée transférable |
| **TransUNet / UNETR / SegFormer** | Segmentation | Hybrides transformer | Entraînable sur PI-CAI | Attention rollout | Pas de poids prostatiques | Oui | Non prostate | MIT/Apache-2.0 | Moyen | C |
| **ImageNet backbones** | Classification | CNN/ViT/Swin | ImageNet | CAM / rollout | — | Oui | Oui | BSD-3/Apache-2.0 | Facile | ★ Prototype rapide |
| **Gleason2019 / SICAPv2** | Grading pixel | — | 331 TMA / 10 340 patchs | — | — | — | — | CC BY 4.0 | Facile | ★ Histologie |
| **PANDA** | Grading ISUP | MIL / patch classifiers | 10 616 WSI | Attention | Référentiel QWK | — | Non | CC BY-SA-NC 4.0 | Moy. à diff. | ★★ Plus grand jeu |
| **PROSTATEx** | Détection / GGG | — | 346 patients | — | — | — | — | CC BY 3.0 | Facile | **DÉPRÉCIÉ** |
| **PROMISE12 / PCA-Rad** | Vérité histologique | — | ~100 patients appariés | — | — | — | — | CC BY 3.0 | Difficile | ★ Vérité terrain |
| **Radiologie 2024** | Détection csPCa | DL 3D | Cohorte | Grad-CAM | = radiologues | Non | Non | Radiology | Difficile | C |
| **Étude 2019 / PCA-Rad 2022** | Localisation | CNN / U-Net | 307 IRM / 122 patients | Grad-CAM | Recouvrement quantifié | Non | Non | PMC | Moyen | C |
| **pytorch-grad-cam** | XAI | 9 variantes CAM | n/a | — | fournit ARCC | Oui | n/a | MIT | **Facile** | ★ **À utiliser** |
| **Captum** | XAI | 15 méthodes | n/a | — | — | Oui | n/a | BSD-3 | **Facile** | ★ **À utiliser** |
| **SHAP** | XAI | Shapley | n/a | — | — | Oui | n/a | MIT | Facile | ★ Pour le « pourquoi » |
| **TCAV** | XAI | Concepts | n/a | — | ICML 2018 | Oui | n/a | Apache-2.0 | Moy. à diff. | ★ Différenciation |
| **StructMIL / Hamm / PM1.0 / ProstAttention** | Divers | — | — | — | — | **Non** | — | Article | — | **D — référence scientifique** |

### 11.2 Méthodes XAI

| Méthode | Modèles compatibles | Visualisation | Avantages | Limites | Pertinence prostatique |
|---|---|---|---|---|---|
| **Grad-CAM** | CNN (ResNet, EfficientNet, U-Net, AH-Net, DA-UNet) | Carte de chaleur grossière sur l'IRM axiale T2W | 1 fwd+bwd ; **la méthode de référence prostatique** ; vérification terrain existante | Grossière ; **non causale** ; sature avec la perte de Dice | **TRÈS ÉLEVÉE** |
| **Explication par signes PI-RADS** | CNN de détection + classifieurs de signes | Carte **+ compte rendu textuel** | **Validée cliniquement** (80 %, +58 s, confiance 4,1 vs 3,4) | Annotation de signes coûteuse ; pas de code | **TRÈS ÉLEVÉE** |
| **Explication en vocabulaire d'anatomopathologiste** | U-Net de patterns | Carte + vocabulaire | **Intrinsèque** ; labels souples ; Dice 0,713 > 0,691 | Histopathologie ; pas de poids | **TRÈS ÉLEVÉE** |
| **Cartes d'attention MIL** | ABMIL/CLAM/DSMIL/TransMIL | WSI, glandes en évidence | **Natif** ; seul mode viable à l'échelle WSI | Poids d'attention ≠ prototype | **ÉLEVÉE** |
| **Attribution par prototypes** | MIL à pooling prototypique | Prototypes **nommés** | Localisation vérifiée contre les masques | Plus complexe ; pas de dépôt | **ÉLEVÉE** |
| **Integrated Gradients** | Tout modèle différentiable | Attribution signée par voxel | **Complétude** ; complément naturel de la carte de détection | Sensible à la référence ; lent en 3D | **ÉLEVÉE** |
| **SHAP** | Tout modèle via wrapper | Superposition diffuse | Garanties fortes ; déjà utilisé pour le grading | Exponentiel à l'exact ; hors de portée sur WSI | MOY. à ÉLEVÉE |
| **LIME** | Toute boîte noire | Segmentaire | Indépendant du modèle | Superpixels sans correspondance clinique ; lent | FAIBLE à MOYENNE |
| **Occlusion / Feature Ablation** | Tout modèle | Grille de deltas | **L'outil de validation** grâce aux deux vérités terrain | Lent ; fenêtre et remplissage dominent | **TRÈS ÉLEVÉE** |
| **RISE** | Modèle différentiable | Carte lisse | Robuste | Milliers de passes | MOYENNE |
| **Seg-Grad-CAM** | U-Net, nnU-Net, DA-UNet, AH-Net | Superposée au masque | **Aligné sur la segmentation** → accord mesurable | Pas de norme universellement adoptée | **ÉLEVÉE** |
| **Attention rollout / flow** | ViT, Swin, TransUNet, ProstAttention-Net | Grille de patchs | Sans modification ni gradient | Non validée causalement | MOYENNE |
| **TCAV** | Tout réseau | Score signé par concept | Les signes PI-RADS = jeu de concepts prêt | TF ; statistique exigeante | MOYENNE |
| **Dictionnaire PM1.0** | Modèle tabulaire | Tableau de contributions | Variables = **signes cliniques** | Licence exclusive ; exactitude 0,78 | **ÉLEVÉE** |
| **Validation par étude de lecteurs** | Tout modèle clinique | n/a | **Prouve qu'une explication change une décision** | Coûteux en effort humain | **TRÈS ÉLEVÉE** |

### 11.3 Jeux de données

| Jeu | Examens / lames | Patients | Classes | Masques | Accès | Licence | Recommandé |
|---|---|---|---|---|---|---|---|
| **PI-CAI v2.0** | 1 500 examens | 1 476 | 2 (1 075 / 425) | **OUI — 1 295 contours experts + IA pour 1 500** | Zenodo, gratuit | **CC BY-NC 4.0** | **OUI — référence** |
| **PROSTATEx** | 346 sujets | 346 | 2 tâches | Seg. dérivées (98 / 66) | TCIA | CC BY 3.0 | **DÉPRÉCIÉ** |
| **PANDA** | 10 616 WSI | 2 113 | 6 (ISUP 0-5) | **OUI — partiels** | Kaggle / panda.gc | **CC BY-SA-NC 4.0** | **OUI — histologie** |
| **Gleason 2019** | 331 TMA (244 + 87) | 331 | 5 au pixel | **OUI — 6 experts** | grand-challenge / Mendeley | CC BY 4.0 | PEUT-ÊTRE |
| **SICAPv2** | ~10 340 patchs | ~182 | 5 | OUI | Mendeley | CC BY 4.0 | PEUT-ÊTRE |
| **TCGA-PRAD** | 723 lames | 490 | Gleason, **sans bénin** | Non | TCIA | CC BY 3.0 | MAYBE |
| **Prostate-MRI-US-Biopsy** | 842 sujets | 842 | csPCa / non | Partiel | IDC | CC BY 4.0 | MAYBE |
| **PROMISE12 / PCA-Rad** | ~100 appariés | ~100 | n/a | **OUI — histologie macroscopique** | TCIA | CC BY 3.0 | **OUI — vérité terrain** |
| **AIMI Annotations** | 11 collections | — | par tâche | OUI (DICOM-SEG) | Zenodo | CC BY 4.0 | **OUI — évaluation** |
| **QIN-PROSTATE** | 15 sujets | 15 | n/a | Non | TCIA | CC BY 4.0 | NON |
| **DiagSet** | WSI annotés | non documenté | grades de Gleason | OUI | auteurs | non vérifiée | MAYBE |
| **TCGA-PRAD (IRM)** | 164 cas | 164 | n/a | Non | TCIA | CC BY 3.0 | MAYBE |

---

## 12. Top 5 des modèles

Scoring complet dans `TOP_MODELS.md` et sur le site. En bref :

1. **U-Net de référence PI-CAI + jeu PI-CAI** — réutilisation 10/10, poids 10/10, code 10/10, données 10/10, explicabilité 9/10. Le seul artefact qui coche toutes les cases.
2. **GleasonXAI** — explicabilité 10/10, code 10/10, documentation 10/10, mais poids 4/10 (à entraîner) et jeu de données à obtenir.
3. **AIMI Task788_Prostate** — poids 10/10 et la meilleure licence (CC BY 4.0), indispensable pour la région d'intérêt.
4. **picai_baseline** — reproductibilité 10/10, documentation 10/10, installable par pip, mais pas de poids.
5. **DSPA-U-MIL / StructMIL** — explicabilité 10/10, la frontière de la pathologie computationnelle, mais coût de calcul élevé.

---

## 13. Recommandation

Raisonnement complet dans `RECOMMENDATION.md`. Synthèse :

**Choisi : le U-Net de référence PI-CAI (poids déjà publiés) sur PI-CAI, expliqué par Grad-CAM + Integrated Gradients, avec la carte de chaleur confrontée à la fois à la carte de détection du modèle et aux contours experts, et une couche de signes PI-RADS en prolongement de Hamm et al.**

- **Modèle :** U-Net de référence PI-CAI, ensemble de 5 plis, `weights/unet_F0.pt`→`F4.pt` déjà dans le dépôt Apache-2.0.
- **Jeu de données :** PI-CAI Public Training and Development Dataset v2.0 (1 500 examens, 1 295 avec contours experts, scores PI-RADS pour les 1 500 cas).
- **XAI :** Grad-CAM (visualisation) + Integrated Gradients (attribution signée) + occlusion (fidélité) + une couche de signes PI-RADS.
- **Durée :** 1 à 2 jours pour la démonstration ; 2 à 3 semaines pour l'évaluation complète et la couche PI-RADS.
- **GPU :** 4-8 Go suffisent pour l'inférence sur un cas ; 8-12 Go confortable pour l'entraînement des classifieurs de signes.

**Pourquoi :** c'est la seule combinaison du dossier où l'explication est mesurable contre **deux** vérités terrain indépendantes, et où un lexique clinique validé est disponible pour monter d'un niveau.

**Si le temps est très court :** l'option A (U-Net de référence + Grad-CAM, zéro entraînement) donne une démonstration fonctionnelle en un jour.

**Si vous avez un mois :** l'option C (GleasonXAI pour l'histopathologie + la voie IRM) couvre les deux modalités et vous positionne sur l'explicabilité intrinsèque, qui est le front le plus valorisant.

---

## 14. Architecture proposée du projet

```
        ┌──────────────────────────┐
        │  Examen IRM prostatique   │
        │  (T2W axial + ADC + HBV) │
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │ Prétraitement            │  recadrage prostate, rééchantillonnage
        │                          │  128x128x128, calcul ADC, normalisation
        │                          │  (plans de nnUNetPlans.json)
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │ Modèle                   │  U-Net de référence PI-CAI (5 plis)
        │                          │  + AIMI Task788_Prostate pour la ROI
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │ Prédiction               │  (a) carte de détection voxel à voxel
        │                          │  (b) probabilité de csPCa au cas
        │                          │  (c) lésions candidates + scores
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │ IA explicable             │  ① Grad-CAM  → carte visuelle
        │                          │  ② Integrated Gradients → attribution signée
        │                          │  ③ Occlusion → fidélité (chute de probabilité)
        │                          │  ④ Couche PI-RADS → compte rendu clinique
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │ Carte de chaleur /        │  carte Grad-CAM sur la tranche axiale T2W
        │ segmentation / importance │  carte de détection du modèle en contour
        │                           │  + LocAcc (vs carte ET vs contours experts)
        │                           │  + Faith (chute après suppression)
        │                           │  + compte rendu en signes PI-RADS
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │ Interface                │  Gradio / Streamlit : un cas → prédiction,
        │                          │  carte de détection, carte de chaleur,
        │                          │  attribution IG, métriques, compte rendu
        └──────────────────────────┘
```

### Ce que fait chaque bloc

**Prétraitement.** L'entrée PI-CAI est volumique : T2W axial, carte ADC, DWI à haut b-value. Le recadrage et le rééchantillonnage sont définis par les `nnUNetPlans.json` du modèle téléchargé — les reprendre plutôt que les deviner.

**Modèle.** Le U-Net de référence PI-CAI produit la carte de détection. Ce n'est pas l'état de l'art, mais c'est la référence, elle est Apache-2.0, ses poids sont publiés, et sa carte de détection est la meilleure vérité terrain disponible pour valider une carte de chaleur. AIMI Task788_Prostate fournit la glande prostatique, sans quoi la détection n'a pas de ROI.

**Prédiction.** Trois sorties : la carte de détection continue (une **probabilité par voxel**), la probabilité au niveau du cas, et la liste des lésions candidates.

**IA explicable.** Quatre sorties complémentaires :
- **Grad-CAM** répond à *où* — ce qu'un clinicien peut lire d'un coup d'œil.
- **Integrated Gradients** répond à *pourquoi* — avec garantie de complétude, on peut dire « ce voxel a relevé le logit csPCa de 0,8 ».
- **Occlusion** répond à *est-ce que cela compte* — le test de falsification.
- **Couche PI-RADS** répond à *en termes cliniques* — le niveau supérieur, celui que Hamm et al. ont validé.

**Carte de chaleur / segmentation / importance.** Superposition de la carte Grad-CAM sur la tranche axiale T2W à 60 % d'opacité, contour de la carte de détection du modèle par-dessus, et un cartouche de métriques : **LocAcc** contre la carte de détection *et* contre les contours experts, écart de fidélité après suppression, écart de probabilité au niveau du cas. Sur une lésion annotée, vous affichez côte à côte ce que le modèle pointe et ce que l'expert a dessiné — c'est la figure qui rend la différence entre une démonstration et une étude.

**Interface.** Gradio ou Streamlit : on téléverse un cas PI-CAI, on obtient la carte de détection superposée, la carte de chaleur, l'attribution IG, les métriques de fidélité et le compte rendu en signes PI-RADS. Tout sur un seul écran.

---

## 15. Livrables et comment les vérifier

| Fichier | Contenu |
|---|---|
| `RESEARCH_REPORT.md` | Ce document |
| `research-data.json` | Base structurée : 27 modèles, 15 jeux de données, 14 méthodes XAI, 28 articles, 28 dépôts, 3 options, top-5, architecture, registre de vérification |
| `research-data.js` | Les mêmes données en globale JS, pour que le site fonctionne ouvert depuis le disque (un `fetch()` d'un `.json` local est bloqué par le CORS sur `file://`) |
| `index.html` · `style.css` · `script.js` | Le site statique. HTML/CSS/JS purs — aucun framework, aucune étape de build, aucun appel réseau à l'exécution |
| `TOP_MODELS.md` | Le top 5 avec le scoring complet par critère |
| `RECOMMENDATION.md` | Les trois options, le choix final, le plan semaine par semaine |
| `assets/` | Schémas SVG originaux uniquement — voir `assets/README.md` |
| `tools/` | L'outillage de vérification, conservé comme preuve |

### Reproduire la vérification

```bash
python tools/build_data.py          # régénère research-data.json + research-data.js
node tools/check_site.js            # contrôles statiques : champs, ids, ancres, validité des URL
node tools/render_test.js           # rend le site dans jsdom et vérifie qu'il se remplit
python tools/check_all_links.py     # revérifie chaque URL
python tools/probe_raw.py <repo> <branche> <path>...     # sonde un fichier précis
python tools/verify_repos_git.py <owner/repo>...         # clone sans blobs + git ls-tree
```

---

## 16. Sources

Les 45 URL intégrées à ce dossier ont été vérifiées. La liste complète, lisible par machine, avec type, titre et métadonnées, se trouve dans `research-data.json` et est rendue dans la section Sources du site.

**Niveau 1 — sources officielles de jeux de données**
Zenodo (PI-CAI 6624726, AIMI 8290093 et 13244892) · TCIA (PROSTATEx, TCGA-PRAD, PROMISE12, QIN-PROSTATE) · IDC (ProstateX, Prostate-MRI-US-Biopsy) · Kaggle (PANDA) · grand-challenge.org (PI-CAI, Gleason2019) · figshare

**Niveau 2 — articles originaux**
*The Lancet Oncology* 25:879-887 (PI-CAI) · *Radiology* 307(4):e222276 (Hamm) · *Radiology* 312(2):e232635 · *Nature Medicine* 28:154-163 (PANDA) · *Nature Methods* 18:203 (nnU-Net) · *Radiol AI* 5(4) · *Med Image Anal* 77, 110 · *Abdom Radiol* 47(4) · *npj Digit Med* 9:114 · *Clin Genitourin Cancer* 24(5) · *Eur Radiol* · MICCAI / CVPRW / ACL / ICML / KDD / NeurIPS / CVPR

**Niveau 3 — dépôts officiels**
`DIAGNijmegen/picai_unet_gc_algorithm` · `DIAGNijmegen/picai_baseline` · `DIAGNijmegen/picai_labels` · `DIAGNijmegen/picai_nnunet_semi_supervised_gc_algorithm` · `bamf-health/aimi-prostate-mr` · `DBO-DKFZ/GleasonXAI` · `AlexNmSED/DSPA-MIL` · `MIC-DKFZ/nnUNet` · `Project-MONAI/*` · `jacobgil/pytorch-grad-cam` · `pytorch/captum` · `shap/shap` · `tensorflow/tcav` · `bowang-lab/MedSAM` · `bowang-lab/MedSAM2`

**Niveau 4 — documentation officielle**
pi-cai.grand-challenge.org (données, évaluation, étude de lecteurs) · gleason2019.grand-challenge.org · panda.grand-challenge.org · MONAI Model Zoo · API Captum · `pytorch.org/vision` · documentation nnU-Net · API REST Zenodo · pages Kaggle et TCIA

**Niveau 5 — dépôts tiers (signalés individuellement)**
`Kentaroy47/Kaggle-PANDA-1st-Place-Solution` (non vérifié en profondeur) · `NabaviLab/Prostate-Cancer` (aucun README ni LICENSE) · `mahmoodlab/CLAM` (non inspecté en détail)

**Registre de vérification :** 28 dépôts catalogués (21 inspectés en profondeur) · 6 fichiers de poids sondés avec tailles exactes en octets · 13 licences confirmées · 3 dépôts signalés comme sans licence exploitable · 45 liens vérifiés, 0 en échec.

---

*Préparé le 21 octobre 2026. Ce dossier indique volontairement où la littérature prostatique est en avance et où elle reste faible : la détection de lésions a une PPV de 35-48 % et un faux positif par patient, l'explicabilité validée cliniquement ne couvre que l'IRM et non l'histopathologie, et personne n'a encore mesuré la fidélité des explications à l'échelle de PI-CAI. Ces trois vides sont le projet.*
