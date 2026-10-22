# WHAT_I_NEED_TO_UNDERSTAND.md — tout ce que je dois comprendre

*Document écrit par et pour l'étudiant, avant la présentation. Objectif :
être capable d'expliquer le projet en 2-3 minutes, de répondre aux
questions de la professeure, et de savoir où est chaque chiffre.*

---

## 1. Ce que fait le projet d'ensemble (en une phrase)

> On construit d'abord une **preuve de fonctionnement** vendue d'un modèle
> d'IA qui détecte le cancer de la prostate sur IRM (phase 1 : le modèle
> marche, chiffres à l'appui) — avant d'expliquer ses décisions avec des
> méthodes d'IA explicable / XAI (phase 2, à venir).

## 2. Les 10 mots à savoir définir

| Mot | Définition simple |
|---|---|
| **IRM (MRI)** | imagerie 3D par champ magnétique ; ici : 3 séquences de la prostate (T2W + ADC + DWI-HBV) |
| **voxel** | pixel 3D ; un examen = des milliers de voxels (p.ex. 384×384×19) |
| **T2W** | séquence anatomique de référence (prostate en blanc) |
| **ADC / DWI haut-b** | diffusion de l'eau ; les cancers denses apparaissent sombres sur ADC / clairs sur DWI |
| **checkpoint** | fichier de poids entraîné (~135 Mo de nombres) ; ici 5 checkpoints |
| **5-fold ensemble** | 5 copies du réseau entraînées sur des morceaux de données différents ; à l'inférence on moyenne les 5 |
| **TTA** | test-time augmentation : aussi prédire sur l'image retournée et moyenner |
| **csPCa** | cancer **cliniquement significatif** : ISUP ≥ 2 (Gleason ≥ 3+4) confirmé par biopsies — c'EST ÇA positif |
| **PI-RADS** | score radiologique 1-5 de suspicion IRM (fait par les radiologues ; sert de labels comparatifs) |
| **AUROC / AUC** | la probabilité (entre 0.5 hasard et 1.0 parfait) que le modèle classe un positif au-dessus d'un négatif |

*Vus plus loin :* Grad-CAM (cartes de saillance d'une couche CNN),
IoU « Union/Intersection 3D » entre un contour humain et une détection.

## 3. Le modèle : comment le dire en 3 phrases (scénario vocal)

> « C'est le modèle de référence de la compétition publique PI-CAI
> (Nijmegen), publié dans le Lancet Oncology 2024 avec une AUROC 0.878
> sur un jeu de test de 1 000 IRM masquées. C'est un ensemble de cinq
> U-Nets 3D : chaque U-Net lit T2W+ADC+DWI en même temps et produit
> 1) une probabilité de cancer significatif pour l'examen entier
> et 2) une carte de détection 3D posée sur l'IRM, voxel par voxel.
> Je n'ai PAS entraîné ce modèle ; je l'ai téléchargé avec ses poids
> officiels, j'ai répliqué le pipeline d'inférence officiel sans le
> modifier, et je vérifie que ma réplication produit exactement les
> sorties de référence fournies par les auteurs. »

## 4. La preuve de fonctionnement : la chaîne (et où la montrer)

1. **Chargement** (`src/data_loader.py`) : 3 fichiers `.mha` (T2W, ADC,
   HBV) + la vérité terrain dans `marksheet.csv` (colonne `case_csPCa`,
   `case_ISUP`) + le contour d'expert `.nii.gz`.
2. **Prétraitement** (`src/preprocessing.py`) : alignement par centre de
   gravité → crop/pad (20,256,256) → rééchantillonnage 3.0×0.5×0.5 mm →
   z-score par séquence. Sortie : `(1, 3, 20, 256, 256)`.
3. **Inférence** (`src/inference.py`) : chaque fold puis l'ensemble :
   TTA flip horizontal, softmax, moyenne des 5 folds, lissage
   gaussien σ 1.5 → heatmap `(20,256,256)`.
4. **Post-traitement** : re-cropping/rééchantillonnage inverse usages
   `picai_prep` + `report_guided_annotation` → carte de détection sur la
   grille native de T2W, blobs « lésions candidates ».
5. **Décision** : probabilité patient = max(carte) ; seuil 0.50.
6. **Vérification golden** (`run_demo.py`) : sur le cas de test fourni
   par les auteurs (`10032_1000032`), notre probabilité = 0.5269 vs
   référence 0.5269 (3.9e-6 d'écart) → PASSED.
7. **Figures** (`src/visualization.py`) : IRM native, carte de
   détection superposée, contour expert comparé.

## 5. Les chiffres réels à citer (avec où en parler)

Chiffres du dossier (toujours exacts, mesurés, pas inventés) :

| Chiffre | Où il est |
|---|---|
| Probabilité cas 10032 = **52.69 %** (valeur de référence = 0.5269265174865723) | `outputs/predictions/10032_1000032/prediction.json`, docs/EXPERIMENT_RESULTS.md |
| Golden: |diff| = **3.9e-6** (tolérance 1e-3), 1 voxel / 2.8M > 1e-3 | EXPERIMENT_RESULTS.md §1 |
| 7 cas démo : négatifs **20.2 / 4.0 / 2.1 %** et positifs **16.1 / 34.6 / 11.0 / 52.7 %** | EXPERIMENT_RESULTS.md §2 |
| Vitesse : **0.52 s par cas** (ensemble 5-fold + TTA, GPU RTX 5060 Ti) | EXPERIMENT_RESULTS.md §3 |
| IoU 3D moyen carte vs contour expert : **0.19** (7 cas, 0.56 pour le cas golden) | EXPERIMENT_RESULTS.md §2 |
| Performance publiée du modèle : **AUROC 0.878** (IC95 0.855-0.900), AP 0.558, sensibilité 70.9 % avec 1 sens. 35.5 % (Saha et al. Lancet Oncology 2024) | MODEL_EXPLANATION.md §3.5 |
| 1500 IRM publiques ; **425 positifs** | démonstration : 7 cas sélectionnés (3 négatifs, 4 positifs) | DATA_EXPLANATION.md |

Si la question vient : « pourquoi tes positifs font 16-35 %, le modèle
ne marche pas ? » → Non : 1) j'ai seulement 4 positifs (sur 1500), un
échantillon de commodité ; 2) le point de fonctionnement publié à
seuil 0.50 a une sensibilité **70.9 %** : 29 % des positifs sont sous
le seuil MÊME sur le jeu de test complet ; 3) mes 3 négatifs sont tous
sous le seuil, et l'ordre est cohérent (les positifs obtiennent des
scores supérieurs aux négatifs). Une évaluation honnête de la
performance exige le jeu de test complet (ou au moins des dizaines de
cas par classe) — c'est en phase 2 du projet.

## 6. Où se trouve QUOI (carte des fichiers)

```
prostate_model_demo/
├── README.md                     ← porte d'entrée (framework + chiffres)
├── docs/MODEL_EXPLANATION.md     ← AU ROC 0.878 et architecture (avantcodage)
├── docs/DATA_EXPLANATION.md      ← datasets + licence + sélection démo
├── docs/EXPERIMENT_RESULTS.md    ← table des 7 cas + golden + timing
├── docs/PROGRESS_REPORT.md       ← statut étape par étape
├── docs/WHAT_I_NEED_TO_UNDERSTAND.md ← ce fichier
├── src/…                          ← tout le code (6 modules courts)
├── scripts/run_demo.py           ← commande principale
├── scripts/download_or_prepare_data.py ← récupère les 7 cas
├── scripts/streamlit_app.py      ← mini interface (lection de l'UI)
├── tests/                    ← 13 pytest (13 vingt-secondes)
├── data/demo_cases/…             ← 6-7 Mo par cas téléchargé
├── data/picai_labels/…           ← marksheet + contours experts
├── outputs/figures/*.png         ← une image par cas (à projeter)
├── outputs/summary.json          ← résultats machine-lisibles du run
└── notebooks/demo.ipynb          ← démo interactive pas à pas
```

## 7. Commandes à connaître par cœur

```powershell
cd "D:\Esprit_5eme\Expalnable ia\prostate_model_demo"
.\.venv\Scripts\Activate.ps1

# (1) re-télécharger la marksheet + les 7 cas + annotations (idempotent)
python scripts/download_or_prepare_data.py

# (2) LANCER LA DÉMO (golden test + 7 cas + figures + summary.json)
python scripts/run_demo.py

# (3) un seul cas (rapide ~10 s)
python scripts/run_demo.py --case 10032_1000032

# (4) interface visuelle de démonstration
streamlit run scripts/streamlit_app.py      # ctrl+c pour fermer

# (5) tests automatisés (13)
python -m pytest tests
```

Répondre : **« un run complet dure environ 1 minute sur mon GPU »**.

## 8. Questions probables de la professeure (et réponses)

**Q. Tu l'as entraîné toi-même ?**
R. Non. Modèle officiellement publié (Lancet Oncology 2024) + poids
téléchargés + pipeline d'inférence répliqué à l'identique, la preuve
étant le golden test PASSED (3.9e-6 sur la probabilité). L'entraînement
n'entre pas dans le périmètre de la phase 1 ; la phase 2 (XAI) n'en a
pas besoin.

**Q. C'est quoi une detection map ? Une heatmap ?**
R. Voxel par voxel sur la grille IRM : 0 = rien suspect, vers 1 = très
suspect. La valeur max devient la probabilité de l'examen. Le post-
traitement officiel ébranle les petites valeurs et isole les candidats
de lésion (blobs).

**Q. Comment tu sais que tes résultats sont justes (pas inventés) ?**
R. Triple vérification : (a) golden test contre les sorties de référence
fournies par les auteurs eux-mêmes pour UN cas précis → PASSED ;
(b) 13 tests unitaires ; (c) métriques convergentes avec la sensibilité /
spécificité publiées (tendance) et des IoU 3D non-nuls contre les
annotations d'experts.

**Q. Pourquoi ce modèle et pas un autre ?**
R. En résumé : c'est le seul candidat qui réunissait (1) code complet
officiel + (2) poids publics + (3) un cas de test et des sorties de
référence inclus (pour vérifier sans entraîner), (4) 1 500 IRM
publiques labellisées par des experts, (5) une performance publiée de
haut niveau (Lancet Oncology 2024).

**Q. C'est quoi la phase 2 que tu proposes ?**
R. Des méthodes d'explicabilité (Grad-CAM, Integrated Gradients,
occlusion) appliquées à la même architecture, puis une comparaison
quantitative de ces explications contre (a) la carte de détection (où
le « croit » regarder ?), (b) les contours d'experts (où l'humain
regarde ?), (c) les zones que le modèle ignore malgré des lésions
réelles — le pont vers un projet de recherche en XAI médical.

**Q. Combien de temps prend l'inférence ?**
R. ~0.5 s de GPU par cas → 1500 cas ≈ 15 min. Prétraitement ~1 s par
cas (CPU). C'est pratique pour les expériences de la phase 2 aussi.

## 9. Ce que la phase suivante peut ajouter si je dois livrer davantage

- Le dataset complet : télécharger les 5 archives (~27 Go) et évaluer
  l'ensemble complet via picai_baseline → une AUROC/AP réelle sur le
  fold complet (calcul picai_eval officiel).
- Une UI web déployée (FastAPI + front) si on veut montrer davantage.
- La phase 2 XAI : nécessite de réactiver les gradients dans le UNet
  (par-dessus l'ensemble + la TTA) et un protocole d'évaluation
  spécifique.

## 10. Les pièges que j'ai DÉJÀ traversés (ça évitera de paniquer)

| Problème réel | Solution (déjà appliquée) |
|---|---|
| Checkpoints téléchargés en pointeurs LFS de 134 octets | `git lfs fetch --include="weights/*"` puis `git reset --hard` ; vérifié par les tests (127,229,381 octets) |
| `picai_baseline.unets.py` incompatible avec MONAI ≥ 1.6 (décorateurs retirés) | architecture vendue dans `src/unet_arch.py` avec les 2 décorateurs supprimés et une header d'attribution |
| Téléchargement Zenodo complet (27 Go) inutile | téléchargement partiel par HTTP range (remotezip), fichier par fichier |
| Annotation d'expert absente pour certains cas | basculer sur `human_expert/Pooch25/` (labels binaires ISUP≥2) — géré dans `evaluation.py` / `visualization.py` |
| Différences live de float plateformes (1 voxel > 1e-3 sur la golden map) | reconnu, documenté (EXPERIMENT_RESULTS.md §1) : arrondi kernel entre plateforme Linux/torch 2.3 et Windows/torch 2.11 |
| GPU trop récent (Blackwell sm_120) sur les wheels torch vieux | torch 2.11.0+cu128 installé (CUDA év. dispo) |

## 11. Ce que la démo n'est PAS (à dire honnêtement)

- Pas un outil diagnostique (règlement grand-challenge).
- Pas une évaluation clinique (7 cas ≠ test set de 1000).
- Pas du XAI (phase 2). Les heatmaps du modèle sont des probabilités
  du modèle, PAS des explications de « pourquoi ».
- Pas affaire de données secrètes : tout est public (Zenodo + GitHub)
  et sous licence de recherche.
