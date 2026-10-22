# Recommandation finale

**Projet :** IA explicable pour la détection / l'analyse du cancer de la prostate en imagerie
**Focus :** IRM prostatique biparamétrique (csPCa, ISUP ≥ 2) et histopathologie H&E (grading de Gleason / ISUP)
**Date** 21 octobre 2026
**Complément de** `RESEARCH_REPORT.md` (preuves) et `TOP_MODELS.md` (la sélection)

---

## Le choix, en un encadré

```
┌────────────────────────────────────────────────────────────────────────┐
│  MODÈLE    U-Net de référence PI-CAI (supervisé)                        │
│            ensemble de 5 U-Net 3D ; poids unet_F0.pt → unet_F4.pt      │
│            DÉJÀ dans le dépôt DIAGNijmegen/picai_unet_gc_algorithm    │
│            + AIMI Task788_Prostate (nnU-Net v1) pour la ROI prostatique│
│                                                                        │
│  DONNÉES   PI-CAI Public Training and Development Dataset v2.0          │
│            1 500 IRM biparamétriques · 1 476 patients · 3 centres NL    │
│            dont 1 295 avec CONTOURS DE LÉSION PAR EXPERT HUMAIN         │
│            + scores PI-RADS pour les 1 500 cas (Pooch et al., 2025)     │
│            Zenodo 6624726 · CC BY-NC 4.0 (non commercial)                 │
│                                                                        │
│  XAI       ① Grad-CAM           (pytorch-grad-cam) → la carte visuelle │
│            ② Integrated Gradients (Captum)      → l'attribution signée │
│            ③ Occlusion          (Captum)        → la fidélité          │
│            ④ Couche PI-RADS                    → le compte rendu      │
│                                                                        │
│  SORTIE    Carte de détection superposée + carte de chaleur + attribution│
│            + LocAcc (vs carte du modèle ET vs contours experts)         │
│            + Faith (chute de probabilité après suppression)              │
│            + compte rendu en signes PI-RADS                            │
└────────────────────────────────────────────────────────────────────────┘
```

**Temps d'entraînement :** **aucun** pour la démonstration — les poids sont publiés. 1 à 2 jours pour l'évaluation complète ; 2 à 3 semaines avec la couche PI-RADS.
**GPU :** 4-8 Go suffisent pour l'inférence sur un cas ; 8-12 Go confortable pour entraîner les classifieurs de signes.

---

## Pourquoi celui-ci et pas un autre

Le cahier des charges disait : *« Je préfère largement réutiliser un modèle existant avec des poids pré-entraînés si celui-ci est suffisamment documenté et librement utilisable. »*

Sur la prostate, cet axe est **beaucoup mieux servi** que sur le cerveau, et il faut le dire clairement :

- **Les poids existent, ils sont publiés, et ils sont sous licence permissive.** `weights/unet_F0.pt` à `unet_F4.pt` sont dans le dépôt Apache-2.0, avec un Dockerfile et un script d'inférence. C'est l'équivalent prostatique direct de ce que l'on cherchait — et c'est plus simple à obtenir.
- **La documentation est complète** : README, `process.py` (11 253 o), scripts `build.sh` / `export.sh` / `test.sh`, exemples de test, image Docker.
- **Les données sont Excellence** : 1 500 examens de 1 476 patients, trois centres, 1 295 contours d'expert, et des scores PI-RADS pour l'ensemble du jeu.

Mais ce qui rend ce choix **décisif** tient en une phrase :

> **Le modèle produit une CARTE DE DÉTECTION continue. C'est une vérité terrain par construction, et 1 295 cas portent en plus des contours d'expert.**

Vous avez donc **deux références indépendantes** pour la même image, ce qui permet de *mesurer* une carte de chaleur au lieu de la simply afficher. Concrètement :

- **LocAcc contre la carte de détection** → le modèle est-il cohérent avec lui-même ?
- **LocAcc contre les contours experts** → le modèle et son explication désignent-ils la bonne lésion ?
- **Faith** → supprimer la zone la plus saillante fait chuter la probabilité de csPCa de combien ?

Et il y a une troisième raison, décisive pour un projet de recherche : **un lexique clinique standardisé existe déjà**. Les 14 signes PI-RADS ne sont pas des concepts à inventer, ce sont ceux que les radiologues utilisent déjà. Hamm et al. (*Radiology* 2023) ont démontré qu'un modèle peut écrire sa justification dans ce lexique — et l'ont **validée par des experts** (80 % d'explications jugées correctes sur 1 352 cas) **et par une étude de lecteurs** (confiance des non-experts 4,1 vs 3,4 sur 5, p = 0,007 ; temps de lecture réduit de 58 s, p = 0,009).

Vous n'êtes donc pas dans une terra incognita. Vous êtes sur un terrain où quelqu'un a posé la reference — **et où personne n'a encore mesuré la fidélité des explications à l'échelle de PI-CAI.**

### Pourquoi EfficientNet / U-Net de référence et pas un réseau plus exotic

On pourrait viser plus haut : SwinUNETR, les 5 finalistes de PI-CAI, un détecteur de lésions moderne. Mais :

- **Le U-Net de référence est la RÉFÉRENCE.** Lesfinalistes du défi sont publiés sur grand-challenge.org **sur demande d'accès**, ce qui n'est pas directement réutilisable. Le U-Net de référence est dans un dépôt Apache-2.0, avec ses poids.
- **C'est le modèle dont la carte de détection est la plus documentée**, donc la plus facile à utiliser comme référence de votre explication.
- **Sa sortie est volumique, donc la vérité terrain l'est aussi** — ce qui rend la mesure possible.
- **C'est un CNN 3D simple**, donc Grad-CAM, LayerCAM, Score-CAM et Seg-Grad-CAM s'appliquent sans contorsion. Si vous visez plus tard un transformateur, il vous faudra attention rollout ou Eigen-CAM — une bonne raison de garder le réseau de base ennuyeux.

### Pourquoi quatre méthodes d'explicabilité et non une

Elles couvrent les angles morts l'une de l'autre, et c'est le point que la littérature prostatique fait le plus clairement :

- **Grad-CAM** répond à *où*. C'est ce qu'un clinicien peut lire d'un coup d'œil. C'est aussi la méthode de référence du domaine — donc votre ligne de base.
- **Integrated Gradients** répond à *pourquoi*. Avec la **garantie de complétude**, on peut affirmer « ce voxel a relevé le logit csPCa de 0,8 » et l'énoncé est arithmétiquement vrai. Grad-CAM ne peut pas le formuler.
- **Occlusion** répond à *est-ce que cela compte*. Supprimer la zone mise en évidence et mesurer la chute de probabilité : c'est le test de falsification, et c'est ici que la vérité terrain prostatique le rend possible.
- **Couche PI-RADS** répond à *en termes cliniques*. C'est le niveau supérieur, et c'est la contribution la plus valorisante : elle fait passer le projet de « je montre une carte de chaleur » à « je produis un compte rendu ».

**Une nuance conceptuelle à ne pas rater**, formulée par l'étude de 2022 sur la segmentation prostatique corrélée à l'histologie macroscopique :

> « *Les cartes Grad-CAM décrivent la localisation de l'attention du CNN, alors que la segmentation CNN ne considère que l'activation de classe résultante, qui peut provenir de n'importe où dans le champ réceptif.* »

Donc : **une carte de chaleur et un masque ne disent pas la même chose**, même quand ils se recouvrent. C'est la distinction que votre projet doit rendre visible, et c'est ce qui distingue un travail rigoureux d'une galerie d'images.

### Les compromis, dits honnêtement

| Compromis | Détail | Contournement |
|---|---|---|
| **Non commercial** | PI-CAI est en **CC BY-NC 4.0** | Acceptable pour un projet de recherche ; bloque toute exploitation industrielle. **À vérifier avec votre encadrant avant de commencer.** |
| **Git LFS** | Les poids sont des pointeurs de 134 octets ; le fichier réel fait 127 229 381 octets | `git lfs pull` après le clone. **Sinon l'inférence échoue silencieusement.** Vérifiez avant de conclure à un « modèle sans poids ». |
| **Entrée volumique** | Un examen complet, pas une tranche | Le pipeline est déjà en 3D ; un slice isolé n'a pas de sens clinique. Pour la démo, extraire une tranche axiale T2W **pour l'affichage** mais calculer l'explication sur le volume. |
| **Pas d'IA explicable intégrée** | Le pipeline PI-CAI n'en contient aucune | C'est votre travail : ~30-60 lignes avec pytorch-grad-cam pour Grad-CAM, plus l'adaptation Seg-Grad-CAM et l'occlusion. |
| **Le modèle n'est pas l'état de l'art** | Le U-Net de référence date de 2019 | C'est la **référence**, pas le champion. Et les finalistes ne sont pas librement réutilisables. |
| **La couche PI-RADS est ambitieuse** | Les scores PI-RADS sont par **cas**, pas par lésion | Commencez par l_scores au niveau du cas, puis raffinez. L'étude de Hamm a produit 1 352 explications sur 3 260 lésions : c'est faisable mais long. |
| **Sensibilité 70-90 %, PPV 35-50 %** | La détection de lésions prostatiques génère ~1,4 à 1,9 faux positif par patient | **Assumez-le dans le rapport.** Une IA explicable qui ne montre pas ses faux positifs n'est pas une IA explicable. |
| **L'AUROC dépend de la population** | ~0,88 en suspicion (PI-CAI) contre 0,83 en dépistage (G2) | Rapportez toujours la population. C'est l'argument le plus solide face à un lecteur critique. |

---

## Les trois voies

### Option A — La plus simple

| | |
|---|---|
| **Modèle** | **U-Net de référence PI-CAI** (poids déjà dans le dépôt) |
| **Jeu de données** | PI-CAI Public Training and Development Dataset v2.0 |
| **XAI** | Grad-CAM (pytorch-grad-cam) + Integrated Gradients (Captum) |
| **Pourquoi** | **Zéro entraînement, zéro réimplémentation.** Clonez le dépôt (avec `git lfs pull`), lancez l'inférence sur un cas, superposez la carte de chaleur sur la carte de détection. Vous avez en quelques heures un pipeline d'IA explicable prostatique fonctionnel, entièrement sous licence permissive côté code. |
| **Durée** | **1 à 2 jours** pour une démonstration fonctionnelle (dont ~27 Go de données et 635 Mo de poids) |
| **GPU** | Recommandé (4-8 Go), pas indispensable pour un premier test |
| **Avantages** | Poids entraînés et publiés, sous Apache-2.0 ; code d'inférence et Dockerfile fournis ; **la carte de détection du modèle sert de vérité terrain pour votre carte de chaleur** ; installation triviale (`pip install grad-cam captum`) ; même jeu pour entraînement, validation et évaluation |
| **Inconvénients** | L'inférence est 3D, il faut le volume complet ; pas de couche d'explicabilité préexistante ; données non commerciales ; le modèle de référence n'est pas l'état de l'art |

**Utilisez-la si** le délai est court ou si vous voulez d'abord valider que le pipeline tourne.

---

### Option B — Le meilleur compromis  ← **RECOMMANDÉ**

| | |
|---|---|
| **Modèle** | U-Net de référence PI-CAI (5 plis) pour la détection, **plus** une couche d'explication par concepts PI-RADS inspirée de Hamm et al. |
| **Jeu de données** | PI-CAI (1 500 cas + 1 295 contours experts + scores PI-RADS pour les 1 500 cas), avec PROSTATEx en validation externe malgré sa dépréciation |
| **XAI** | Grad-CAM (carte visuelle) + Integrated Gradients (attribution signée) + fidélité par occlusion + une couche PI-RADS qui traduit la sortie en signes |
| **Pourquoi** | Vous ajoutez la partie qui manque à la quasi-totalité des travaux prostatiques : une explication au **niveau des concepts cliniques**, validée contre les scores PI-RADS désormais publiés pour les 1 500 cas. C'est le prolongement direct de Hamm et al. et cela vous positionne sur l'axe le plus valorisant du domaine : passer de la carte de chaleur à l'affirmation de style compte rendu. |
| **Durée** | **2 à 3 semaines** pour une démonstration soignée, incluant l'évaluation et l'ablation CAM |
| **GPU** | Recommandé — 8-12 Go pour l'inférence 3D et l'entraînement des classifieurs de signes |
| **Avantages** | Vous surfez l'axe de recherche le plus actif du domaine ; les scores PI-RADS publiés rendent la couche de concepts faisable **sans annotation supplémentaire** ; validation possible en comparant votre explication PI-RADS aux scores de praticiens ; la comparaison avec Hamm et al. devient directe et chiffrée ; deux vérités terrain pour l'évaluation |
| **Inconvénients** | La couche de classification de signes est plus complexe qu'un simple CAM ; les scores PI-RADS sont par cas, pas par lésion — donc la granularité lésion est plus difficile ; toujours non commercial ; deux semaines de plus que l'option A |

**Utilisez-la si** vous avez une quinzaine de jours et voulez un résultat, pas une démonstration.

---

### Option C — La plus ambitieuse

| | |
|---|---|
| **Modèle** | **GleasonXAI** (U-Net intrinsèquement explicable) pour l'histopathologie, avec extension à l'IRM via le pipeline PI-CAI |
| **Jeu de données** | Gleason 2019 (244 TMA d'entraînement, 6 experts, annotation pixel) + PANDA (10 616 WSI, 2 113 patients) pour la voie histologie ; PI-CAI pour la voie IRM |
| **XAI** | Explication **intrinsèque** par vocabulaire d'anatomopathologiste (aucune méthode post-hoc) + attention MIL et attribution par prototypes pour la version WSI |
| **Pourquoi** | Vous attaquez le problème de l'explicabilité à sa racine : le réseau ne prédit pas un grade, il prédit une **description**. L'explication EST la prédiction. C'est la démonstration la plus nette qu'on puisse faire, et elle est en MIT avec un code de recherche complet, des tests et des notebooks d'évaluation. Combinée à la voie IRM, elle donne un projet qui couvre **les deux modalités** du cancer prostatique. |
| **Durée** | **3 à 4 semaines** (le jeu de 1 015 TMA annotés par 54 experts doit être obtenu auprès des auteurs) |
| **GPU** | Modéré — U-Net 2D sur patchs ; pas de 3D lourd, mais le pipeline WSI est exigeant en IO |
| **Avantages** | Explicabilité intrinsèque : la question « est-ce que la carte est fidèle ? » ne se pose même pas ; traite explicitement la variabilité inter-annotateur par des labels souples, qui est le vrai problème de la pathologie ; **de meilleurs résultats qu'un modèle entraîné sur le grade** (Dice 0,713 contre 0,691) ; code de recherche le plus complet du dossier (7 configs de modèle, tests unitaires et d'intégration, notebooks d'évaluation, calcul du kappa de Fleiss) ; deux modalités couvertes |
| **Inconvénients** | Pas de poids publiés : il faut entraîner, et le jeu des auteurs est à obtenir ; la voie histopathologie seule dans un premier temps — l'IRM reste à gérer en parallèle ; coût de stockage et d'IO élevé sur les WSI ; domaine très concurrentiel sur l'interprétabilité |

**Utilisez-la si** vous avez un mois et que l'ambition est un article de conférence.

---

## Plan semaine par semaine pour l'option recommandée

### Semaine 1 — fondations et vérité terrain
- **Jour 1.** `git clone https://github.com/DIAGNijmegen/picai_unet_gc_algorithm` puis **`git lfs pull`** — et **vérifiez** que `weights/unet_F0.pt` fait bien 127 229 381 octets et non 134. Si c'est 134, votre installation de git-lfs est cassée : réparez-la maintenant, pas dans trois jours. Téléchargez PI-CAI (5 plis, ~27 Go) depuis Zenodo et les annotations depuis `DIAGNijmegen/picai_labels`.
- **Jour 2.** Lisez intégralement le README de `picai_labels` : c'est la documentation d'annotations la plus précise qui soit (quels cas ont un contour expert, quels cas ont des contours IA, répartition PI-RADS et ISUP, résolution des annotations). **C'est votre table des matières** : elle détermine ce que vous pouvez réellement faire.
- **Jour 3.** Faites tourner l'inférence sur 10 cas avec le `process.py` fourni. Comparez visuellement la carte de détection à la vérité terrain disponible. Notez le temps d'inférence et la consommation mémoire : ce sont vos chiffres de rapport.
- **Jour 4.** Calculez la **performance de référence** sur un sous-ensemble : AUROC patient, AP, et la FROC lésion. Vous devez pouvoir reproduire ~0,88 d'AUROC. Si vous êtes loin, le problème est dans le prétraitement, pas dans le modèle. **Gardez ce chiffre comme référence interne, jamais comme résultat de votre projet.**
- **Jour 5.** Ajoutez Grad-CAM. Produisez la galerie de cartes de chaleur sur 20 cas. **Regardez-les de manière critique.** Si une carte tombe sur le bord de l'image, sur la vessie ou sur la zone périphérique saine, c'est un résultat sur le modèle, pas un bug à masquer. Notez chaque cas où cela arrive : c'est votre premier contenu scientifique.

### Semaine 2 — la mesure (c'est la contribution)
- **Jour 6.** Implémentez les **trois métriques** : LocAcc contre la **carte de détection du modèle** (auto-cohérence) et contre les **contours experts** (justesse clinique) ; Faith (chute de probabilité de csPCa après suppression des k voxels les plus saillants). Ce sont les métriques de `arXiv:2601.12826` transposées, et le protocole de l'étude de 2022 sur l'histologie macroscopique.
- **Jour 7.** Ajoutez Integrated Gradients. Testez **trois références** (noire, moyenne, floutée) et rapportez à quel point l'attribution change : c'est la limite que tout rapport IG honnête mentionne. Vérifiez la complétude (la somme des attributions doit correspondre à l'écart de score) — c'est votre preuve que l'attribution est Plus qu'une image.
- **Jour 8.** Lancer l'**ablation CAM** : Grad-CAM vs Grad-CAM++ vs LayerCAM vs Score-CAM, sur le même modèle, les mêmes cas, notés avec les mêmes métriques. Score-CAM est le plus intéressant ici : il n'utilise aucun gradient, donc il se comporte différemment là où la perte de Dice sature.
- **Jour 9.** Ajoutez un **témoin négatif** : appliquez toute la chaîne d'explicabilité à des cas **sans csPCa** (les 1 075 cas bénins/indolents). Une bonne explication ne doit montrer aucune localisation cohérente sur un cerveau sain. Si elle en montre une, le modèle utilise un raccourci — et c'est un résultat.
- **Jour 10.** Commencez la **couche PI-RADS** : entraînez un classifieur de signes sur les scores PI-RADS désormais publiés (Pooch et al. 2025), en visant d'abord la classification **au niveau du cas**, puis par lésion. Comparez la précision de vos signes prédits aux scores publiés : c'est votre premier test chiffré contre une référence clinique.

### Semaine 3 — validation et livraison
- **Jour 11.** **Validation externe** : faites tourner le modèle sur PROSTATEx (les cas non inclus dans PI-CAI — vérifiez lesquels) ou, mieux, documentez l'écart entre population de suspicion et population de dépistage avec l'étude G2 (AUROC 0,83). C'est l'argument le plus solide face à un lecteur critique, et il est déjà publié.
- **Jour 12.** Rédigez le rapport : méthode, protocole, les trois métriques, l'ablation CAM, l'expérience témoin, la couche PI-RADS, la validation externe, et **les limites** (non commercial, PPV 35-50 %, ~1,4 faux positif par patient, dépendance à la population).
- **Jour 13.** Construisez l'interface : Gradio ou Streamlit. Un cas → carte de détection superposée, carte de chaleur, attribution IG, les trois métriques, et le compte rendu PI-RADS. Un seul écran, sans navigation.
- **Jour 14.** Tampon. Rejouez tout depuis un clone propre (**avec git-lfs**), écrivez le README avec l'attribution complète (Apache-2.0 pour le code, **CC BY-NC 4.0 pour les données PI-CAI**, et les citations obligatoires : le protocole PI-CAI, le protocole d'étude de lecteurs, Pooch et al. 2025, Saha et al. 2024).

---

## Ce à quoi doit ressembler un bon résultat à la fin de ce projet

Le résultat que vous voulez n'est **pas** « notre modèle atteint 88 % d'AUROC ». Ce chiffre est déjà publié, il est reproductible, et il ne dit rien de l'explicabilité — c'est le score de la *prédiction*, pas de l'*explication*.

Le résultat que vous voulez est un énoncé de cette forme :

> *« Sur le jeu de test PI-CAI, les explications Grad-CAM du modèle de référence recouvrent les contours experts de lésion avec une LocAcc moyenne de X (contre Y pour les cas sans csPCa), et la carte de détection du modèle lui-même avec une LocAcc de Z — ce qui montre que l'explication et la prédiction désignent bien la même région. La suppression des 10 % de voxels les plus saillants fait chuter la probabilité de csPCa de W points. La même mesure appliquée à Grad-CAM++ / LayerCAM / Score-CAM montre que [méthode] produit les explications les plus fidèles pour un coût de [N] passes directes. Enfin, la couche PI-RADS produit un compte rendu textuel dont la concordance avec les scores PI-RADS publiés est de V, contre U pour une explanation par heatmap seule. »*

C'est mesurable, falsifiable, reproductible à partir d'artefacts publics, et **personne ne l'a fait à cette échelle**.

---

## Si vous ne deviez retenir qu'une chose

Prenez **le U-Net de référence PI-CAI** (option A). Les poids sont dans le dépôt, sous Apache-2.0, avec un Dockerfile et un script d'inférence. Clonez, `git lfs pull`, lancez, et vous avez en un jour un pipeline d'IA explicable prostatique fonctionnel — avec, en prime, la vérité terrain la plus exploitable de tout le dossier.

Et gardez une chose en tête en permanence : **vérifiez que vos poids font bien 127 229 381 octets, pas 134.** C'est le seul piège de ce projet, et il échoue sans bruit.

---

## Sources de la recommandation

| Affirmation | Source |
|---|---|
| Poids `unet_F0.pt`→`F4.pt` dans le dépôt, Apache-2.0, tailles réelles via LFS | `git ls-tree -r -l HEAD` + sonde HTTP sur `raw.githubusercontent.com` |
| Fichier `Task788_Prostate.zip` = 1 675 047 442 o, CC BY 4.0, nnU-Net v1, entraîné sur PI-CAI | API REST Zenodo, enregistrement 8290093 |
| PI-CAI : 1 500 examens / 1 476 patients / 3 centres, 1 295 contours experts, 1 087 lésions positives, 776 lésions ISUP, scores PI-RADS | `DIAGNijmegen/picai_labels` (README, 18 002 o) + enregistrement Zenodo 6624726 (v2.0, CC BY-NC 4.0) |
| PROSTATEx déprécié, remplacé par PI-CAI | Page collection TCIA PROSTATEx |
| Performance du U-Net de référence : score 0,718, AUROC 0,878, AP 0,558, Sens3/4/5 | Classements PI-CAI, phase de test fermé (grand-challenge.org) |
| PI-CAI : non-infériorité des IA vs radiologues ; étude de lecteurs 50+ radiologues sur 400 cas | Saha et al., *Lancet Oncol* 25(7):879-887, PMID 38876123 |
| Hamm : AUROC 0,89/0,87, explications 80 %, signes à 98 %, confiance 4,1 vs 3,4, −58 s | *Radiology* 307(4):e222276, PMID 37039688 |
| « Don't Trust, Verify » | *Radiology* 307(4) éditorial, PMC10323286 |
| « carte de chaleur ≠ masque » et corrélation à l'histologie macroscopique | PMC8976981 |
| Première vérification quantitative Grad-CAM vs PI-RADS vs anatomopathologiste | PMC6920905 (2019) |
| Grad-CAM sur la lésion dans 35/38 et 56/58 vrais positifs ; performance = radiologues | *Radiology* 312(2):e232635 |
| AUROC 0,83 en population de DÉPISTAGE | *Eur Radiol*, PMC13086701, PMID 41359160 |
| Sensibilité 70,9 %, PPV 35,5 %, 1,41 faux positif/patient | *Abdom Radiol* 47(4):1425-1434, PMID 35099572 |
| Dice 0,713 (explications) > 0,691 (grade) ; 1 015 TMA, 54 experts | arXiv:2410.15012 / PMC12508442 |
| MIL à attention ; +7,4 % de kappa inter-scanner ; +6,7 % de QWK ; alignement vérifié | npj Digit Med 9:114, PMID 41540138 ; Med Image Anal 110:104005, PMID 41762944 |
| PANDA : 10 616 WSI, 2 113 patients, 2 centres, masks partiels, CC BY-SA-NC 4.0 | PMC8799467 ; règles de la compétition Kaggle |
| 1 087 lésions positives (PI-RADS 3 : 246, 4 : 438, 5 : 403) et 776 lésions ISUP | `DIAGNijmegen/picai_labels` |
| licences des données (CC BY-NC 4.0 PI-CAI, CC BY-SA-NC 4.0 PANDA) | Enregistrements Zenodo 6624726 et règles Kaggle PANDA |
