# Top 5 des solutions — Sélection

**Complément de** `RESEARCH_REPORT.md` · **Source des données** `research-data.json` (`top_models`)
**Date** 21 octobre 2026

Chaque affirmation « poids disponibles » de cette page a été confrontée à l'arborescence réelle du dépôt (`git ls-tree -r -l HEAD`) ou à une sonde HTTP du fichier brut, avec la taille exacte en octets enregistrée. Chaque licence a été lue dans le fichier `LICENSE` du dépôt ou dans l'enregistrement d'API Zenodo.

---

## Comment le classement a été établi

Dix critères, notés de 1 à 10. La pondération reflète la priorité du cahier des charges : **réutiliser un modèle existant avec des poids pré-entraînés plutôt que d'entraîner depuis zéro**, à condition qu'il soit suffisamment documenté et librement utilisable.

| # | Critère | Ce qu'il mesure |
|---|---|---|
| 1 | **Facilité de réutilisation** | Pouvez-vous obtenir un résultat aujourd'hui, sans projet de recherche ? |
| 2 | **Qualité du modèle** | Précision/Dice publié, et crédibilité de celui-ci |
| 3 | **Disponibilité des poids** | Les fichiers existent-ils, sont-ils récupérables, sous quelle licence ? |
| 4 | **Disponibilité du code** | Y a-t-il du vrai code exécutable — pas seulement un README ? |
| 5 | **Qualité du jeu de données** | Les données sont-elles bonnes, et comportent-elles une vérité terrain ? |
| 6 | **Explicabilité** | À quel point l'artefact soutient la moitié « IA explicable » du projet ? |
| 7 | **Reproductibilité** | Un tiers obtiendrait-il les mêmes chiffres ? |
| 8 | **Coût computationnel** | Combien de GPU et de temps machine ? |
| 9 | **Adéquation étudiante** | Est-ce réalisable, présentable et défendable dans un projet étudiant ? |
| 10 | **Documentation** | Y a-t-il assez d'écrit pour ne pas rester bloqué ? |

---

## ① U-Net de référence PI-CAI + jeu de données PI-CAI

**`github.com/DIAGNijmegen/picai_unet_gc_algorithm`** · Apache-2.0 · 5 étoiles · PyTorch / MONAI

| Réutilis. | Qualité | Poids | Code | Données | XAI | Repro | Calcul | Étudiant | Docs |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 8 | 10 | 10 | 10 | 9 | 10 | 7 | 9 | 9 |

**L'argument en une ligne.** C'est le seul artefact prostatique qui coche **toutes** les cases à la fois : le code d'inférence, le Dockerfile, les cinq points de contrôle entraînés **déjà dans le dépôt**, sous licence permissive, et le jeu de données le plus annoté au monde avec deux vérités terrain indépendantes pour valider l'explication.

**Contenu vérifié** (`git ls-tree -r -l HEAD`, 33 fichiers) :
```
weights/unet_F0.pt … unet_F4.pt   ← 5 plis, 127 229 381 octets chacun (via Git LFS)
Dockerfile            859 o
LICENSE           11 357 o  (Apache-2.0)
process.py         11 253 o
requirements.txt      257 o
build.sh / export.sh / test.sh / setup.cfg
test/  ← exemples .mha (T2, ADC, HBV, coronal, sagittal) + clinical-information-prostate-mri.json
```

**Le détail à ne pas rater.** `weights/unet_F0.pt` renvoie 134 octets : c'est un pointeur Git LFS.
```
version https://git-lfs.github.com/spec/v1
oid sha256:d21d31bb8609ae12272da93a8c8556613cdaa0834c2fe1451f59557f814ce69d
size 127229381
```
Le fichier réel fait **127 229 381 octets**. Avec `git lfs pull`, on l'obtient ; sans cela, l'inférence échoue **silencieusement**. C'est le piège technique n°1 de ce projet.

**Performance publiée** (PI-CAI, phase de test fermé, cohorte cachée de 1 000 scans) : score 0,718 · AUROC 0,878 (IC 95 % 0,855-0,900) · AP 0,558 · Sens3 0,693 · Sens4 0,636 · Sens5 0,472. La variante semi-supervisée atteint 0,734 / 0,881.

**Pourquoi il est en tête.** Aucun autre artefact du dossier ne fournit simultanément des poids entraînés **et** une carte de détection continue. Et cette carte, c'est la matière première de l'explicabilité prostatique : elle est une vérité terrain par construction, ce qui rend la mesure de fidélité possible à l'échelle de PI-CAI.

**Les limites à assumer.** L'entrée est volumique 3D (une tranche isolée n'a pas de sens clinique) ; le pipeline n'embarque aucune IA explicable, la couche XAI est entièrement à écrire ; et les données sont en **CC BY-NC 4.0**, donc usage non commercial.

**Difficulté de réutilisation : facile à moyen.**

---

## ② GleasonXAI — U-Net intrinsèquement explicable

**`github.com/DBO-DKFZ/GleasonXAI`** · MIT (LICENSE 1 066 o) · PyTorch Lightning

| Réutilis. | Qualité | Poids | Code | Données | XAI | Repro | Calcul | Étudiant | Docs |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 9 | 4 | 10 | 7 | 10 | 8 | 8 | 7 | 10 |

**L'argument en une ligne.** C'est le **code de recherche le plus complet du dossier** et la démonstration la plus nette qu'on puisse faire de l'explicabilité intrinsèque : le réseau ne prédit pas un grade, il prédit une *description* en terminologie d'anatomopathologiste, donc l'explication **est** la prédiction.

**Contenu vérifié** (80+ fichiers) : `src/gleasonxai/` (`gleason_data.py` 18 980 o, `lightning_modul.py` 21 798 o, `jdt_losses.py`, `loss_functions.py`, `model_utils.py`, `tree_loss.py`, `augmentations.py`) · `configs/` pour **7 architectures** de segmentation (unet, unetplusplus_efficientnet, densenet, efficientnet, deeplabv3p_efficientnet, fcn, resnet) plus pertes, dataset, optimiseur, trainer, dataloader, augmentations, logger · `scripts/` (`train.py`, `test.py`, `run_gleasonXAI.py`, `calculate_fleiss_kappa.py`, `calculate_dataset_characteristics.py`, `create_metric_visualization.py`, `evaluate_paper_results.ipynb` 896 610 o, `feature_visualization.ipynb` 106 444 o) · `tests/` unitaires et d'intégration · `pyproject.toml`, `uv.lock`, `CHANGELOG.md`.

**Performance publiée** : Dice **0,713 ± 0,003** entraîné sur les **explications**, contre 0,691 ± 0,010 entraîné directement sur les patterns de Gleason. Entraîner sur l'explication est donc *meilleur* — parce que les labels souples encodent l'incertitude inter-annotateur que le grade ne contient pas. Le jeu de 1 015 TMA a été annoté par **54 anatomopathologistes**.

**Pourquoi il est deuxième.** Le poids est le critère faible (4/10) : aucun point de contrôle n'est publié, il faut entraîner, et le jeu de données des auteurs doit être obtenu. Le critère données est également faible (7/10) pour la même raison.

**Les limites à assumer.** Histopathologie et non IRM ; pas de poids ; la reproduction exige d'obtenir le jeu des auteurs. Mais le patron technique — configs complètes, tests unitaires et d'intégration, notebooks d'évaluation, calcul du kappa de Fleiss — est exactement ce qu'il faut copier pour un projet reproductible.

**Difficulté de réutilisation : moyen.**

---

## ③ AIMI Task788_Prostate — nnU-Net v1, CC BY 4.0

**`github.com/bamf-health/aimi-prostate-mr`** · MIT (code) / **CC BY 4.0 (poids)** · PyTorch + nnU-Net v1

| Réutilis. | Qualité | Poids | Code | Données | XAI | Repro | Calcul | Étudiant | Docs |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 8 | 10 | 7 | 10 | 4 | 9 | 6 | 8 | 7 |

**L'argument en une ligne.** Les poids prostatiques les plus permissivement licenciés trouvés : **CC BY 4.0**, entraînés sur PI-CAI, donc dans le même domaine que le projet.

**Contenu vérifié** (API Zenodo, enregistrement 8290093) : modèle nnU-Net **v1**, entraîné sur les données PI-CAI, fichier `Task788_Prostate.zip` = **1 675 047 442 octets (1,67 Go)**, licence **CC BY 4.0**, dépôt de code `bamf-health/aimi-prostate-mr`. Le dépôt GitHub est mince : README (5 586 o) + LICENSE (1 068 o, MIT).

**Pourquoi il est dans la sélection.** En pratique, il est **indispensable** : sans segmentation de la prostate, la détection de lésion n'a pas de région d'intérêt. Et le CC BY 4.0 sur des poids médicaux est une rareté — dans le dossier cérébral, seul AIMI l'offrait aussi.

**Les limites à assumer.** nnU-Net v1, donc une version ancienne ; la segmentation de la glande seulement, pas des lésions ; 1,67 Go à télécharger ; et lower XAI (4/10) puisque le modèle ne contient aucune explicabilité.

**Difficulté de réutilisation : moyen.**

---

## ④ picai_baseline — la boîte à outils officielle

**`github.com/DIAGNijmegen/picai_baseline`** · Apache-2.0 · 142 fichiers · PyTorch / MONAI / nnDetection

| Réutilis. | Qualité | Poids | Code | Données | XAI | Repro | Calcul | Étudiant | Docs |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 9 | 7 | 10 | 10 | 5 | 10 | 7 | 9 | 10 |

**L'argument en une ligne.** La façon la plus correcte de reproduire les chiffres publiés — parce qu'elle fournit **les découpages 5 plis prêts à l'emploi**, qui sont l'élément le plus difficile à reproduire soi-même, et une installation par `pip`.

**Contenu vérifié** (racine, 142 fichiers) : `.gitattributes`, `.gitignore`, `LICENSE` (Apache-2.0), `README.md`, **`nndetection_baseline.md`**, **`nnunet_baseline.md`**, **`unet_baseline.md`**, `pyproject.toml`, `requirements.txt`, `requirements_dev.txt`, `setup.cfg`, `setup.py`.

**Les trois références fournies** partagent la même préparation de données : U-Net, nnU-Net et nnDetection. La configuration nnU-Net retenue est **0,81 (M) / 0,888 (C) / 0,732** selon le post-traitement appliqué — un écart de 0,08 selon le choix, qu'il faut signaler.

**Pourquoi il est dans la sélection.** Sans lui, vous n'avez pas de référence de découpage, donc pas de comparaison possible. C'est le socle méthodologique, pas un modèle.

**Les limites à assumer.** Aucun poids ; 142 fichiers de code à comprendre ; l'entraînement est en 3D sur GPU.

**Difficulté de réutilisation : facile.**

---

## ⑤ DSPA-U-MIL / StructMIL — la frontière de l'explicabilité en pathologie

**`github.com/AlexNmSED/DSPA-MIL`** · MIT (44 fichiers) · et [npj Digital Medicine 9:114](https://www.nature.com/articles/s41746-025-02289-4)

| Réutilis. | Qualité | Poids | Code | Données | XAI | Repro | Calcul | Étudiant | Docs |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 9 | 6 | 8 | 9 | 10 | 8 | 5 | 6 | 8 |

**L'argument en une ligne.** Ici, l'explicabilité n'est pas une méthode ajoutée : c'est le **mécanisme de décision**. Et la tendance actuelle remplace l'attention opaque — dont on sait qu'elle ne désigne pas un prototype représentatif — par des **prototypes histologiques nommés**.

**Performance publiée** :
- **StructMIL** : Camelyon16, +3,2 % d'AUC inter-centre (AUC 0,967) ; PANDA, **+7,4 % de gain de Cohen's kappa en robustesse de grading inter-scanner**.
- **DSPA-U-MIL** : jusqu'à **+6,7 % de QWK** sur la base solide CLAM, sur 5 jeux (10 809 WSI + 1 133 TMA), avec le plus grand gain sur TCGA-PRAD.
- **Grading prototype adaptatif** : les prototypes à attention > 0,5 localisent **systématiquement** des structures glandulaires bien définies, en accord clair avec les masques de vérité terrain.

**Le point méthodologique décisif** : DSPA-U-MIL valide l'explication en montrant que **les distributions de poids de gating du modèle élève sont alignées sur les annotations de patterns des anatomopathologistes**. C'est la bonne façon de tester une explicabilité, et c'est un modèle à imiter.

**Les limites à assumer.** Coût de calcul élevé (des dizaines de milliers de patchs par lame, stockage et IO dominants) ; pas de poids publiés ; et le domaine de la pathologie computationnelle est très concurrentiel sur l'interprétabilité.

**Difficulté de réutilisation : moyen.**

---

## La sélection en un coup d'œil

| | Solution | Sans entraînement ? | Poids dans le dépôt ? | Licence du code | Licence des données | Explicabilité incluse ? | Vérité terrain pour la noter ? | Difficulté |
|---|---|---|---|---|---|---|---|---|
| ① | U-Net de référence PI-CAI | **Oui** | **Oui** (5 plis) | Apache-2.0 | CC BY-NC 4.0 | Non (à écrire) | **Deux** (carte du modèle + contours experts) | Facile à moyen |
| ② | GleasonXAI | Non (à entraîner) | Non | MIT | à obtenir | **Oui — intrinsèque** | Oui (masques Gleason) | Moyen |
| ③ | AIMI Task788_Prostate | **Oui** | Non (Zenodo) | MIT | CC BY 4.0 | Non | Oui (segmentation de la glande) | Moyen |
| ④ | picai_baseline | Non | Non | Apache-2.0 | CC BY-NC 4.0 | Non | — (fournit les découpages) | Facile |
| ⑤ | DSPA-U-MIL / StructMIL | Non | Non | MIT | CC BY-SA-NC 4.0 | **Oui — natives** | Oui (masques PANDA) | Moyen |

---

## Ce que la sélection dit dans son ensemble

1. **Cinq solutions sur cinq n'exigent aucun entraînement** pour produire un résultat (① ③ ④ sont des poids publiés ; ② ⑤ demandent un entraînement mais sont des modèles documentés et faisables).
2. **Le point d'entrée « zéro friction » est ①.** Code + poids + Docker + licence Apache-2.0, dans un dépôt de 33 fichiers. C'est l'équivalent prostatique direct de ce qu'on cherchait.
3. **Aucune des cinq ne fournit d'explicabilité intégrée** — sauf ② et ⑤, qui sont des architectures *intrinsèquement* explicables, donc le problème est résolu par construction plutôt que par collage.
4. **La contrainte structurante n'est pas le code, ce sont les données.** PI-CAI et PANDA sont toutes deux en **non commercial**. Le code prostatique est en outre bien mieux licencié que le code cérébral (MIT/Apache quasi partout) — c'est une amélioration nette par rapport au dossier précédent.
5. **La prostate a un avantage que le cerveau n'a pas** : un lexique clinique standardisé (PI-RADS, patterns de Gleason) qui permet une explication au niveau des concepts, validée cliniquement par Hamm et al. (Radiology 2023). Vous n'êtes pas obligé de l'inventer.

---

## Également examinés, et pourquoi ils sont écartés

| Candidat | Pourquoi il n'est pas dans le top 5 |
|---|---|
| **Modèle PI-RADS de Hamm et al.** (*Radiology* 2023) | **C'est la référence clinique** — 80 % d'explications confirmées par des experts, confiance des non-experts 3,4 → 4,1, −58 s de lecture. Mais **ni code ni poids n'ont été localisés**, donc non exécutable. À citer, à mesurer, à dépasser. |
| **PI-CAI Baseline nnU-Net semi-supervisé** | Apache-2.0, 33 fichiers, trois entraîneurs nnU-Net personnalisés. Le meilleur score (0,734). Mais pas de poids, et le modèle supervisé est plus simple à démarrer. |
| **MedSAM / MedSAM2** | Apache-2.0 sur le code *et* les poids (MedSAM), pilotable, avec GUI et notebook. Écarté du top 5 seulement parce qu'il ne fait pas de détection de csPCa : c'est un **outil de production de masque**, très utile en complément pour valider une carte de chaleur en quelques minutes. **Fort candidat pour l'étape d'annotation.** |
| **PROSTATEx** | CC BY 3.0, 346 sujets, et c'est le jeu de la validation externe de Hamm et al. Mais la TCIA le déclare **explicitement déprécié et remplacé par PI-CAI**, dont il contient les 328 cas de ProstateX. L'utiliser en parallèle créerait un chevauchement de patients. |
| **PANDA** | Le plus grand jeu de grading prostatique public, avec des masques d'annotation partiels — donc une vérité terrain rare pour valider les cartes d'attention MIL. Mais **CC BY-SA-NC** (non commercial + partage à l'identique), et le test final est détenu par Kaggle. |
| **Dictionnaire radiomique PM1.0** | La forme la plus lisible d'explicabilité : chaque variable du modèle est un signe PI-RADS. Écarté parce que l'article est sous **licence exclusive** — vous ne pouvez pas redistribuer le dictionnaire, et il n'y a pas de code. Exactitude 0,78, en dessous des modèles profonds. |
| **ProstAttention-Net** | L'idée est excellente (attention multi-étiquette par grade d'agressivité : la granularité de la prédiction *est* l'explication). Mais ni code ni poids ni jeu de données public. |
| **U-Net / AH-Net** (*Abdom Radiol* 2022) | Son **protocole de données** est la référence de rigueur (vérité par biopsie IRM-TRUS, deux institutions, découpage strict) et ses chiffres sont honnêtes : sensibilité 70,9 %, **PPV 35,5 %**, ~1,4 faux positif par patient. Mais rien n'est publié en code. À citer pour le rappel de la réalité clinique. |
| **`NabaviLab/Prostate-Cancer`** | Cité par arXiv:2212.12844 comme fournisseur de code, mais **ni README ni LICENSE à la racine** lors de l'inspection. **[!] Ne construisez pas votre projet dessus.** |
| **`Kentaroy47/Kaggle-PANDA-1st-Place-Solution`** | README (7 594 o) et LICENSE (19 346 o) présents, mais le clone blobless a expiré : **l'arborescence n'a pas pu être vérifiée**. À contrôler avant de s'y fier. |
