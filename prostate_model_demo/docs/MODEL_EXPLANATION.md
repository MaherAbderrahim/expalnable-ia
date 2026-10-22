# MODEL_EXPLANATION.md — modèle choisi pour la démonstration (phase 1)

Sous-projet de `prostate_model_demo/` — écrit avant le codage, comme exigé
par le protocole du projet destiné à être montré à la professeure.

---

## 3.1 Le problème

### Quel est le problème médical ?

Le modèle tente de répondre à une question :

> **Étant donné une IRM de la prostate, y a-t-il un cancer de la
> prostate cliniquement significatif (csPCa) caché dans cet examen, et
> où est-il ?**

### Contexte du projet

Ce projet est la première étape d'un projet de recherche sur l'**IA
explicable (XAI) appliquée à la détection du cancer de la prostate sur
IRM**. Avant d'expliquer les prédictions d'un modèle d'IA (phase 2), il
faut d'abord un modèle dont les prédictions fonctionnent réellement
(phase 1) — c'est ce que ce sous-projet démontre.

### Qu'est-ce que la prostate et le cancer de la prostate ?

La prostate est une petite glande (taille d'une noix, ~30-80 mL) située
sous la vessie chez l'homme, autour de l'urètre. Le cancer de la prostate
est un cancer très fréquent chez les hommes âgés. La plupart croissent
lentement et ne nuiraient jamais au patient (« indolents ») ; une minorité
est agressive.

- **csPCa = clinically significant prostate cancer** — le cancer qui
  affecterait réellement le patient (grade de Gleason ≥ 3+4 / ISUP ≥ 2).
  Il nécessite un traitement ou une surveillance.
- Le cancer indolent (ISUP 1, Gleason 6) est commun, et *le détecter fait
  plus de mal que de bien* (chirurgies inutiles). C'est pourquoi le modèle
  cible **spécifiquement le csPCa**, pas « n'importe quel cancer ».

### Pourquoi l'IRM ?

Le cancer de la prostate est le plus souvent invisible à l'échographie ou
au scanner. L'IRM multiparamétrique (bpMRI) montre :
- l'anatomie (T2 pondérée),
- la restriction de la diffusion de l'eau dans les tissus cancéreux
  denses (ADC + DWI haut-b).
C'est pourquoi la **bpMRI est l'examen standard avant les biopsies**
(scoring PI-RADS). Les radiologues lisent des centaines de coupes par
examen, et la qualité de lecture varie (c'est le problème de
variabilité inter-lecteur) — d'où l'intérêt d'une IA qui classe les
lésions de façon constante, et l'intérêt d'une IA dont on peut *vérifier
les raisons* (c'est la phase 2).

### Que doit prédire le modèle ?

Deux sorties liées pour un même examen :
1. **une probabilité niveau patient** que l'examen contienne un csPCa
   (décision « pas de biopsie / biopsie ») ;
2. **une carte de détection niveau lésion** (« detection map ») : une
   carte 3D de suspect où les régions voxels suspectes sont mises en
   évidence (sert à guider les biopsies et à comparer aux rapports et
   annotations).

La vérité terrain est l'histologie : un cas est « positif » si biopsies
ou chirurgie ont trouvé un cancer de **grade ISUP ≥ 2** (Gleason ≥ 3+4).

---

## 3.2 Les données

Voir `DATA_EXPLANATION.md` pour les détails complets (source Zenodo 6624726,
licence CC BY-NC 4.0, annotations picai_labels).

Ce que le modèle consomme : **une IRM = un examen du patient** — une
session d'IRM avec **3 séquences axiales co-enregistrées** :

| Séquence | Ce qu'elle montre | Résolution typique (cas démo) | Rôle |
|---|---|---|---|
| **T2W** (T2 pondérée) | anatomie, zones de la prostate | 384×384 px, 0.5×0.5 mm, 19 coupes ×3 mm | morphologie ; *la carte de détection est définie sur cette grille* |
| **ADC** | coefficient de diffusion apparent (dérivé de la DWI) | 384×384 reformaté, 2.0×2.0 mm, 19 coupes | le tissue cancéreux restreint la diffusion → ADC sombre |
| **HBV** (DWI haut-b) | image de diffusion brute (haute valeur-b) | 384×384 reformaté, 2.0×2.0 mm | le cancer reste brillant à haute valeur-b (exclut le « T2 shine-through ») |

Chaque examen du dataset PI-CAI contient aussi des T2W coronal/sagittal —
**ce modèle ne les utilise pas**.

> Remarque : les grilles ADC/HBV diffèrent physiquement de la grille T2W
> (résolution dans le plan plus faible, p.ex. 2×2 mm vs 0.5×0.5 mm). Le
> preprocessing rééchantillonne les 3 sur une même grille (voir 3.3).

### Dimensions et formalismes

- Format : images lisibles par SimpleITK (`.mha` dans le cas de test
  officiel et le sous-ensemble démo ; `.nii.gz` dans le dataset complet —
  même contenu voxels, SimpleITK gère les deux).
- Taille : varie par cas (p.ex. cas démo : 384×384×19 voxels) ; le dataset
  public PI-CAI contient 1 500 examens de ce type.

### Annotations/labels disponibles (PI-CAI)

Pour chaque cas (`patient_id_study_id`) :

| Élément | Où | Niveau |
|---|---|---|
| `case_csPCa` (YES/NO), `case_ISUP` | `picai_labels/clinical_information/marksheet.csv` | cas |
| `lesion_GS`, `lesion_ISUP`, `lesion_PIRADS` | même fichier | lésion (valeurs séparées par des virgules par lésion) |
| contours de lésions d'experts (masques binaires/multi-classes) | `picai_labels/csPCa_lesion_delineations/human_expert/` | voxel |
| contours de la prostate entière | `picai_labels/anatomical_delineations/` | voxel |

### patient vs examen vs lésion (3 niveaux à ne pas confondre)

- **patient** : une personne, identifiée par `patient_id` (ex. `10008`).
  Peut apparaître plusieurs fois *avec des examens différents* (années
  différentes). → plusieurs **examens**.
- **examen** (aussi *cas*, *study*) : une session d'IRM,
  `patient_id_study_id` (ex. `10008_1000008`). C'est l'unité que le
  modèle traite et l'unité de l'AUROC niveau cas.
- **lésion** : un foyer cancéreux dans un examen. 1 examen peut contenir
  0, 1 ou plusieurs lésions. L'évaluation lésion (FROC, « Sens3/4/5 ») se
  fait sur les lésions.

---

## 3.3 Entrée du modèle

La chaîne exacte (reprise du `process.py` officiel) :

```
3 volumes axiaux (T2W, ADC, HBV), chacun sur sa grille native
   │
   ├─ alignement par centre de gravité        (picai_prep Sample)
   ├─ crop/pad centré vers (20, 256, 256)     (rééchantillonnage+crop)
   ├─ rééchantillonnage vers space (3.0, 0.5, 0.5) mm (z, y, x)
   │
   └─ normalisation z-score par modalité
      (clippée aux percentiles 0.5 / 99.5)
   │
   └─ empilement des modalités en canaux
        ↓
   tenseur torch (1, 3, 20, 256, 256)   ← entrée du modèle
```

Concrètement :
- 3 = les trois séquences IRM (canaux, comme RGB) ;
- 20 = coupes axiales après rééchantillonnage à 3 mm d'épaisseur ;
- 256×256 = pixels de 0.5 mm ;
- 1 = dimension de batch (un examen à la fois).

Donc le modèle voit un **cube 3D de taille fixe** autour de la zone de la
prostate — **un examen entier, pas une coupe**.

---

## 3.4 Sortie du modèle

Deux sorties, toutes deux produites par le script officiel :

1. **Carte de détection** (`cspca_detection_map.mha`) — carte 3D sur la
   grille native du T2W (même forme/espacement que le volume T2W en
   entrée), dont les valeurs sont des scores de csPCa-likelihood dans
   [0, 1] (taches = candidats de lésion ; les détections en anneau
   parasites sont supprimées).

2. **Probabilité niveau patient** (`cspca-case-level-likelihood.json`) —
   nombre réel = `max(carte de détection)`, la probabilité que l'examen
   contienne un csPCa. Seuil d'exploitation : **0.50** (point publié :
   70.9 % de sensibilité, PPV 35.5 %).

Concrètement, pour le cas démo `10032_1000032` :
> Le modèle reçoit l'examen T2W+ADC+HBV et produit :
> - une probabilité de **0.526926 ≈ 52.7 %** — la valeur de référence
>   officielle pour cet examen est 0.5269265174865723 (fournie dans le
>   repo ; nous la reproduisons à 4×10⁻⁶), et
> - une carte de détection (19×384×384 voxels) dont les scores élevés
>   recouvrent le contour de lésion de l'expert (IoU 3D ≈ 0.56 dans
>   notre exécution).
> La vérité terrain de référence dit : cette lésion est ISUP grade 3, donc
> l'examen est csPCa-positif — cohérent avec notre 52.7 % ≥ 0.5.

Note (honnêteté sur les définitions) :
- le modèle ne sort **pas** un masque de segmentation de la lésion (sa
  carte de détection est une carte de scores, pas un masque binaire) ;
- il ne sort **pas** un grade ISUP / un score de Gleason (le grading
  exige des biopsies) — la carte de scores binaire publiée est csPCa vs
  pas-de-csPCa ;
- et il ne sort **pas** un score PI-RADS (c'est le système de scoring du
  radiologue ; les *labels* contiennent les PI-RADS).

## 3.5 Architecture

### En une phrase

Le modèle est un **ensemble (« ensemble ») de cinq U-Nets 3D**
(cross-validation 5-fold), implémenté en PyTorch/MONAI et entraîné par
l'équipe DIAG-Nijmegen sur le dataset public PI-CAI.

### Qu'est-ce qu'un U-Net ?

Un CNN (réseau convolutif) en forme de « U » :
- **encodeur** (bras gauche) : blocs répétés *convolution + sous-
  échantillonnage* qui compriment l'image en motifs de haut niveau
  (chaque étage double les features et réduit la résolution spatiale) ;
- **goulot** (bas) : la représentation la plus comprimée ;
- **décodeur** (bras droit) : *convolutions transposées* qui
  ré-agrandissent, en fusionnant l'information sauvée par les
  **connexions sautées** (« skip connections ») de l'encodeur (pour
  garder le détail spatial fin) — c'est ce qui permet à un U-Net de
  produire une carte *de la même taille que l'entrée*, adapté aux cartes
  de détection, contrairement à un classifieur qui sort juste un nombre.

```
entrée (1,3,20,256,256)
    │
    ▼
┌──────────────────────────── U-Net 3D ────────────────────────────┐
│  conv 32 ──► conv 64 ──► conv 128 ──► conv 256 ──► conv 512      │ encodeur
│    │ skip        │skip        │skip        │skip        │        │ (sous-
│    │             │            │            │            ▼        │ échantillonnage
│    │             │            │            │           goulot    │  par strides)
│    │             │            │            └────► conv 1024 ──► │   │ décodeur
│    │             │            └────────── up+concat ──── 512      │   │ (sur-
│    │             └───────── up+concat ──── 256 ◄──────────────    │   │  échantillonnage
│    └──────── up+concat ──── 128 ◄─────────────────────────────    │   │  + skip)
│      up+concat ── 64 ── ► 64-conv ──► conv(out)=2 canaux ──────► │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
sortie (1, 2, 20, 256, 256) : canal 0 = « pas de csPCa »,
                              canal 1 = « csPCa »
```

Hyperparamètres clés (officiels `default_hyperparam.py`) :
- `channels = [32, 64, 128, 256, 512, 1024]`
- `strides  = [(2,2,2), (1,2,2), (1,2,2), (1,2,2), (2,2,2)]`
- blocs résiduels,
- ~32.5 M de paramètres par fold (vérifié au chargement dans les tests).

### Qu'est-ce que « 3D » ici ?

Les convolutions opèrent sur des volumes de voxels 3D (grille 3×3×3),
pas sur des patchs 2D — la décision d'un voxel utilise son
voisinage dans les trois directions. C'est important car les lésions
sont des structures 3D s'étendant souvent sur plusieurs coupes.

### Le « M » de l'« ensemble 5-fold »

Le modèle complet = 5 copies de cette architecture, chacune entraînée
sur 4/5 du dataset et prédisant sur le cinquième réservé. À l'inférence,
les 5 cartes de sortie sont **moyennées** (après sigmoid + lissage).
L'ensembling réduit la variance de surapprentissage d'un fold unique.
C'est pourquoi la démo charge 5 checkpoints (`unet_F0.pt` …
`unet_F4.pt`).

### Augmentation de données au test (TTA)

Chaque fold voit aussi le volume retourné (gauche-droite), et les deux
prédictions sont moyennées — astuce standard pour réduire le biais
d'orientation.

### Post-traitement (pipeline officiel)

- inversion du crop/rééchantillonnage pour obtenir une carte sur la
  grille native du T2W ;
- `extract_lesion_candidates(..., threshold='dynamic')` du paquet
  Report-Guided-Annotation transforme la carte de scores brute en carte
  de détection propre (blobs = candidats de lésion) ;
- les blobs de valeur < max/5 sont supprimés (artefacts en anneau).

### Source de l'architecture

- Code d'inférence (Apache-2.0) :
  https://github.com/DIAGNijmegen/picai_unet_gc_algorithm
- Architecture + entraînement :
  https://github.com/DIAGNijmegen/picai_baseline
  (`src/picai_baseline/unet/training_setup/neural_networks/unets.py`
  est un `UNet` MONAI ; notre `src/unet_arch.py` reprend exactement ce
  fichier pour que les checkpoints se chargent strictement, avec pour
  seule retouche les deux décorateurs MONAI retirés).
- Model card :
  https://grand-challenge.org/algorithms/the-pi-cai-challenge-baseline-u-net-algorithm/
- Résultats publiés : Saha A, Bosma JS, Twilt JJ, et al. « Artificial
  intelligence and radiologists in prostate cancer detection on MRI
  (PI-CAI) », **Lancet Oncology** 2024;25(7):879–887.
  Sur le jeu de test masqué PI-CAI : AUROC 0.878 (IC95 % 0.855–0.900),
  AP 0.558.

---

## Ce que la phase 1 en fait (et ne fait pas)

- Phase 1 (cette démo) : charger les données → prétraiter
  (réplication officielle) → exécuter l'ensemble → carte de détection +
  probabilité patient → visualiser → vérifier vs vérité terrain.
  **Pas de XAI.**
- Phase 2 (travail futur) : expliquer *pourquoi* le modèle regarde là où
  il regarde : Grad-CAM / Integrated Gradients / occlusion par-dessus
  ces sorties ; comparer ces explications à la carte de détection et aux
  contours d'experts (LocAcc, Faith) — voir la section « Étape suivante »
  du README.
