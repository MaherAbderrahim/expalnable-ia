# DATA_EXPLANATION.md — les données utilisées dans la démonstration

## 1. Le dataset à la source (PI-CAI)

**Nom** : PI-CAI **Public Training and Development Dataset v2.0**
**Source** : enregistrement Zenodo
[6624726](https://zenodo.org/record/6624726)
**Licence** : **CC BY-NC 4.0** (recherche non commerciale uniquement)
**Contenu** : 1 500 examens de prostate bpMRI issus de 1 476 patients,
collectés sur 3 sites néerlandais (RUMC, ZGT, PCNN ; 11 sites sur toute la
compétition). Chaque examen contient les volumes T2W (axial + coronal +
sagittal), ADC et DWI haut-b, plus les données cliniques.

| Caractéristique | Valeur |
|---|---|
| Examens (cas) | 1 500 |
| Patients | 1 476 |
| Examens csPCa-positifs (ISUP ≥ 2) | 425 |
| Examens bénins / indolents | 1 075 |
| Lésions IRM positives annotées | 1 087 (PIRADS 3 : 246, 4 : 438, 5 : 403) |
| Fichiers par cas | `*_t2w.mha`, `*_cor.mha`, `*_sag.mha`, `*_adc.mha`, `*_hbv.mha` |
| Organisation des archives | 5 fichiers zip (`picai_public_images_fold0..4.zip`), ~5.3 Go chacun (~27 Go au total) |

La question clinique encodée dans les labels : un examen est **positif**
s'il contient une lésion de **grade ISUP ≥ 2** (Gleason ≥ 3+4) confirmée
par biopsie/prostatectomie — c'est-à-dire un cancer *cliniquement
significatif* (csPCa). Un examen est **négatif** si les biopsies/suivi
IRM n'ont trouvé que du tissu bénin ou un cancer indolent (ISUP ≤ 1,
PI-RADS ≤ 2 sans histologie positive).

## 2. Annotations (vérité terrain) — picai_labels

**Nom** : [DIAGNijmegen/picai_labels](https://github.com/DIAGNijmegen/picai_labels)
**Licence** : CC BY-NC 4.0

| Ressource | Chemin utilisé dans la démo | Niveau | Signification |
|---|---|---|---|
| `clinical_information/marksheet.csv` | `data/picai_labels/clinical_information/` | cas | âge du patient, PSA, PSAd, volume prostatique, `histopath_type`, `lesion_PIRADS`, `lesion_GS`, **`case_ISUP`, `case_csPCa` (YES/NO) — notre vérité terrain niveau cas |
| `csPCa_lesion_delineations/human_expert/resampled/<case>.nii.gz` | même arborescence dans `data/picai_labels/` | voxel | contour de lésion d'expert sur la **grille T2W** ; labels 0 (ISUP≤1) / 2..5 (grade ISUP de la lésion) |
| `csPCa_lesion_delineations/human_expert/Pooch25/<case>.nii.gz` | même arborescence | voxel | contours d'experts pour les 205 cas ajoutés par [Pooch et al. 2025](https://www.medrxiv.org/content/early/2025/05/13/2025.05.13.25327456) ; **labels binaires** (1 = ISUP ≥ 2) |

Créés par 10 enquêtrices/1 résidente en radiologie sous la supervision de
3 radiologues experts (ITK-SNAP), RUMC + UMCG.
Référence standard : histologie.

## 3. Le cas de test officiel (secours hors-ligne)

**Source** : fourni dans le dépôt du modèle officiel
[DIAGNijmegen/picai_unet_gc_algorithm](https://github.com/DIAGNijmegen/picai_unet_gc_algorithm),
dossier `test/` (fichiers `10032_1000032_{t2w,adc,hbv,cor,sag}.mha`,
`clinical-information-prostate-mri.json`, et les **sorties de
référence** : `cspca-case-level-likelihood.json` = 0.5269265174865723 et
une carte de détection de référence).

- Le cas `10032_1000032` est un vrai examen PI-CAI (patient 10032, étude
  1000032, PSA 19.18, volume prostatique 57 mL, Siemens Skyra, b=1400).
- C'est la même étude que dans le dataset Zenodo (vérifié : nommage des
  fichiers et géométrie identiques).
- Objectif : valider le pipeline *hors-ligne* contre les sorties de
  référence des auteurs eux-mêmes (notre « golden test »).

## 4. Ce que la démo télécharge (partiel, reproductible)

`scripts/download_or_prepare_data.py` :

1. télécharge `marksheet.csv` (97 ko) depuis picai_labels ;
2. liste le contenu des 5 archives Zenodo via des **requêtes HTTP
   range** (remotezip) et localise les cas demandés ;
3. télécharge UNIQUEMENT les 3 modalités axiales (T2W, ADC, HBV) de
   chaque cas sélectionné (fichiers .mha, ~5–20 Mo par cas) ;
4. télécharge l'annotation de lésion d'expert de chaque cas sélectionné
   (`resampled/` d'abord, puis `Pooch25/`).

Résultat : quelques cas réels pour quelques Mo, sans les 27 Go complets.

**Cas sélectionnés** (aussi dans `data/demo_cases.json`) :

| Cas | Pourquoi il est dans la démo | Vérité terrain (marksheet) |
|---|---|---|
| `10032_1000032` | cas de test officiel, référence dorée | csPCa YES, case_ISUP 3, PI-RADS « 5,4 », PSA 19.18 |
| `10005_1000005` | premier positif annoté de la marksheet | csPCa YES, case_ISUP 3, PI-RADS « 4,4 » |
| `10008_1000008` | deuxième positif annoté | csPCa YES, case_ISUP 3, PI-RADS « 5,N/A », GS « 4+3,3+4 » |
| `10012_1000012` | troisième positif annoté | csPCa YES, case_ISUP 2, PI-RADS 5 |
| `10000_1000000` | premier négatif | csPCa NO (PI-RADS 4, biopsie 0+0, ISUP 0) |
| `10001_1000001` | deuxième négatif | csPCa NO (PI-RADS 2, pas d'histologie nécessaire) |
| `10002_1000002` | troisième négatif | csPCa NO (PI-RADS 2) |

C'est un **échantillon de commodité** : les premiers positifs/négatifs
de la marksheet officielle, choisis pour couvrir les deux issues et des
niveaux PI-RADS / grades ISUP variés. Il n'est PAS représentatif et NE
peut PAS servir à affirmer des performances du modèle. Les chiffres de
performance publiés (AUROC 0.878 etc.) proviennent du **jeu de test
masqué de 1 000 scans** de PI-CAI, pas de ce sous-ensemble.

## 5. Autres datasets (NON utilisés dans cette démo, gardés pour phase 2/autres axes)

- **PROSTATEx** (TCIA) : jeu public plus ancien vérifié par biopsies ;
  officiellement *remplacé* par PI-CAI (les 328 cas ProstateX ont été
  intégrés dans PI-CAI) — nous ne l'utilisons pas, en suivant les
  recommandations de picai_labels.
- **PANDA** (histopathologie) : pour l'axe histopathologie du projet de
  recherche, pas pour cette démo IRM.

## 6. Notes éthique/conformité

- Les deux datasets sont en **usage recherche uniquement** (CC BY-NC
  4.0) : aucune utilisation commerciale.
- Aucune tentative de dé-identification ou de redistribution de données
  personnelles ; les données sont utilisées telles que fournies par les
  distributeurs officiels, à des fins de recherche.
- La démo ne revendique aucune validité clinique ; c'est une
  démonstration de reproductibilité et de pipeline.
