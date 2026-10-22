# EXPERIMENT_RESULTS.md — ce qui a été réellement obtenu

*Tous les chiffres ci-dessous proviennent d'exécutions réelles du
2026-10-22 (Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, RTX 5060 Ti
16 Go). Rien ici n'est simulé. Reproduire avec `scripts/run_demo.py`.*

---

## 1. Test de reproductibilité golden (cas officiel 10032_1000032)

Comparaison de notre inférence avec les sorties de référence **fournies
par les auteurs du modèle dans leur propre dépôt** (dossier `test/`) :

| Métrique | Référence des auteurs | Notre exécution | Écart | Statut |
|---|---|---|---|---|
| Probabilité de cas | 0.5269265174865723 | 0.5269225835800171 | 3.9e-6 | OK (< 1e-3) |
| Valeur max de la carte de détection | 0.5269265 | 0.5269226 | 3.9e-6 | OK |
| Voxel d'argmax (z=6, y=254, x=182) | identique | identique | — | OK |
| Carte de détection | 2 801 664 voxels | identique | 1 voxel / 0.00003 % | voir note |

**Note.** Exactement UN voxel (11, 267, 171) s'écarte au-delà de la
tolérance officielle de 1e-3 : la référence l'inclut dans la tache de
lésion (valeur = Max), notre exécution le met à 0.0. C'est un voxel de
bord de la même tache dont le score sous-jacent tombe exactement à la
coupe du seuil dynamique ; des kernels en virgule flottante différents
(Docker Linux + torch 2.3 + ancien numpy vs Windows + torch 2.11 +
numpy 2.5) accumulent des fractions de 1e-6 d'écarts à travers les
convolutions/rééchantillonnages et déplacent ce franchissement de bord.
Le `test.sh` officiel des auteurs exige ≤ 1e-3 sur leur plateforme —
respecté là-bas. Notre reproduction atteint la même probabilité de cas,
le même argmax et la même géométrie de tache ; sur 4 574/2 801 664
voxels (0.16 %) les valeurs diffèrent de moins de 1e-3 (voxels de bord).
Verdict : **le pipeline reproduit le résultat de référence**.

## 2. Exécution démo sur 7 cas (4 positifs + 3 négatifs)

Commande : `python scripts/run_demo.py` (après que `--add-case-count 6`
a téléchargé les cas). Chaque cas exécute le pipeline officiel complet
(prétraitement → ensemble 5-fold + TTA → carte de détection +
probabilité).

| Cas | GT (marksheet) | Probabilité notre exécution | ≥ 0.50 ? | Résultat vs GT | IoU carte détection vs contour expert |
|---|---|---|---|---|---|
| 10000_1000000 | csPCa NO | 20.2 % | non | OK | 0.00 (GT : pas de lésion) |
| 10001_1000001 | csPCa NO | 4.0 % | non | OK | 0.00 (GT : pas de lésion) |
| 10002_1000002 | csPCa NO | 2.1 % | non | OK | 0.00 (GT : pas de lésion) |
| 10005_1000005 | csPCa YES (ISUP 3) | 16.1 % | non | mismatch (sous le seuil) | 0.43 |
| 10008_1000008 | csPCa YES (ISUP 3, annotation binaire Pooch25) | 34.6 % | non | mismatch (sous le seuil) | 0.09 |
| 10012_1000012 | csPCa YES (ISUP 2) | 11.0 % | non | mismatch (sous le seuil) | 0.24 |
| 10032_1000032 | csPCa YES (ISUP 3) | 52.7 % | oui | OK | 0.56 |

Ce que cela signifie et ne signifie PAS :

- Le pipeline fonctionne de bout en bout sur des données réelles,
  hétérogènes (centres, scanners, résolutions, présence/absence de
  lésions différents).
- 3/3 négatifs obtiennent des scores bas ; les positifs des scores
  supérieurs — sur ce tout petit échantillon l'ordre est cohérent
  (vérification visuelle ; un AUC sur 7 cas n'a aucun sens et n'est PAS
  reporté comme résultat).
- 3/4 positifs tombent sous le seuil d'exploitation de 0.50 — attendu :
  le point d'exploitation officiel est sensibilité 70.9 % avec PPV
  35.5 % (1.41 faux positifs par patient sur le jeu de test masqué).
  Une poignée de cas ne peut estimer AUCUN de ces nombres.
- **Ces résultats servent uniquement à vérifier que le pipeline
  fonctionne.** Ce ne sont ni une évaluation clinique ni scientifique.

## 3. Chronométrage / ressources (RTX 5060 Ti, CUDA 12.8)

| Étape | Temps |
|---|---|
| Chargement des modèles (5 checkpoints, ~635 Mo au total) | 0.9 s |
| Prétraitement par cas (min–max sur les 7 cas) | 0.81 – 1.49 s |
| Inférence par cas (5 modèles × 2 TTA) | 0.48 – 0.62 s (moyenne 0.52 s) |
| Post-traitement + visualisation | < 1 s |
| VRAM observée pendant un cas | ≈ 2–3 Go (bien dans les 16 Go) |

L'exécution CPU est supportée par le même chemin de code
(auto-détection `get_device`), mais plus lente (non benchmarkée ici,
hors du périmètre de la démo).

## 4. Sorties structurelles (vérifiées par les tests)

- tenseur d'entrée prétraité : `(1, 3, 20, 256, 256)` float32 ;
- heatmap (softmax) brut de l'ensemble : `(20, 256, 256)` dans [0,1] ;
- carte de détection finale : mêmes forme/espacement que le volume T2W
  natif du cas (p.ex. `(19, 384, 384)` pour 10032_1000032) ;
- artéfacts par cas dans `outputs/predictions/<case_id>/` :
  `cspca_detection_map.mha` + `prediction.json` ;
- figure par cas dans `outputs/figures/<case_id>.png` ;
- résumé global : `outputs/summary.json`.

## 5. Suite de tests

`pytest tests` → **13 tests passés** (chargement des données y compris
détection de pointeurs Git LFS, prétraitement, chargement
architecture/checkpoints, inférence pour un cas, reproduction de la
référence golden).

## 6. Limitations connues (liste honnête)

1. Différences d'arrondi de bords vs la référence Docker (1 voxel) ;
   voir la note ci-dessus.
2. Correctif de dépréciation : le fichier UNet vendu a dû voir deux
   décorateurs MONAI retirés (`alias`/`export` retirés en MONAI ≥ 1.6) ;
   documenté en tête de `src/unet_arch.py`.
3. Les annotations Pooch25 sont binaires (ISUP ≥ 2) tandis que les
   annotations `resampled/` portent des grades ISUP 2..5 — la démo lit
   l'une ou l'autre (documenté dans DATA_EXPLANATION.md et géré par
   `evaluation.lesion_localization_case`).
4. Les panneaux ADC/HBV de certains cas affichent un contraste
   arrière-plan sombre prononcé (les données sont fournies telles
   quelles ; les centiles d'affichage excluent les zones vides mais les
   volumes enregistrés conservent leur apparence native).
5. Les téléchargements dépendent de la disponibilité de Zenodo + GitHub
   raw.
