# XAI — comparaison des trois familles (valeurs réellement mesurées)

> Généré par `scripts/xai_report.py` à partir de `experiments/results/xai/xai_quality.csv` (run real pipeline, 6 cas de validation, poids officiels PI-CAI 5 plis, sortie expliquée = score csPCa fusionné). Aucune valeur n'est simulée ; les moyennes par famille agrègent les cas audités.

| Méthode | Famille | Loc. IoU top-5% (vs contours experts) | Enrichissement | Insertion AUC (fidelité↑) | Deletion AUC (fidelité↓) | Randomisation r (sensibilité modèle ↓) | Stabilité (jitter ±1 voxel) |
|---|---|---|---|---|---|---|---|
| gradcam | CAM (class activation mapping) | 0.0020 | 1.00 | 0.558 | 0.400 | 0.022 | 0.428 |
| gradcampp | CAM (variant Grad-CAM++) | 0.0129 | 10.50 | 0.517 | 0.276 | 0.022 | 0.899 |
| integrated_gradients | gradient-path (integrated gradients) | 0.0028 | 1.70 | 0.626 | 0.253 | 0.622 | 0.991 |
| occlusion | perturbation (occlusion sensitivity) | 0.0149 | 5.98 | 0.636 | 0.310 | 0.217 | 0.952 |

Protocols & reading guide (NOT comparable as a single ranking):
- insertion AUC: higher = the model regains score by re-inserting the
  top-attributed voxels on a blurred baseline (1 = best possible in this
  normalized-curve protocol).
- deletion AUC: LOWER = removing the top-attributed voxels destroys the
  score faster (better faithfulness).
- randomization r: LOWER = the map reacts to weight corruption
  (Adebayo-style control, top-block scheme for CAM/IG-depth-top,
  single-fold coarse scan for occlusion).
- stability: 1 = invariant to a 1-voxel y-shift (0 < r <= 1).

## Supplementary control — IG randomization depth='all'

| case | IG r (fold-0 fully randomized) |
|---|---|
| 10388_1000394 | 0.381 |
| 10650_1000666 | 0.418 |
| 10810_1000826 | 0.377 |
| 10925_1000942 | 0.411 |
| 11175_1001198 | 0.353 |
| 11341_1001364 | 0.395 |

## Cross-case variability (input-dependence in practice)

| method | mean pairwise Pearson across cases | undefined pairs |
|---|---|---|
| gradcam | 0.007 | 9 |
| integrated_gradients | 0.678 | 0 |
| occlusion | 0.143 | 0 |
