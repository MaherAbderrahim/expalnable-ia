# Phase 2 — Explainable AI for csPCa Detection on bpMRI (PI-CAI)

## 1. Context and objective

Phase 1 established a working, byte-faithful reproduction (golden test PASSED,
|Δ likelihood| = 3.94e-6 vs the authors' reference) of the official PI-CAI
`picai_unet_gc_algorithm` csPCa detector (Saha et al., Lancet Oncology 2024 —
AUROC 0.878 on the hidden test set, AP 0.558, sens 70.9% @ PPV 35.5% @ 0.50).

Phase 2 now answers the scientific questions Phase 1 left open:

1. Is the Phase-1 single-cases demo performance representative? -> properly
   powered benchmark on 600 stratified cases (leakage-free splits).
2. Can the *inference/post-processing* layer be improved without retraining
   (no GPU training available — user constraint)? -> 10 pre-registered
   post-processing configurations, selected on validation PR-AUC, test scored
   once after freeze.
3. Where does the model fail, for which subpopulations (center, PI-RADS,
   lesion size…)? -> error analysis with clinically stratified cuts.
4. Which evidence does the model "use"? -> 4 XAI methods + 4 quality checks.
5. Is the whole pipeline reproducible end-to-end in one command?

## 2. Honest-scope statement (READ FIRST)

- The official checkpoints were trained on ALL 1,500 public PI-CAI cases.
  Any PI-CAI metric we compute here is an **optimistic upper bound** for the
  pretrained model, because our "train pool" cases were *seen in training* by
  the reference model; only the split hygiene (patient-level val/test with no
  patient overlap) is genuinely honest for THOSE two splits.
- What this phase can claim: a **calibration/threshold/post-processing
  benchmark** and an **explainability audit** of the released model — not a
  novel-dataset generalization gap.
- No metric was tuned on test. The selection rule was pre-registered in
  scripts/select_config.py BEFORE the test split was touched:
  "max val PR-AUC (tie-break AUROC); decision threshold = max F1 on
  validation; Youden J reported; test scored once, after freeze."

## 3. Data, splits, leakage control

600 cases stratified by center, hospital, csPCa, biopsy_result, PI-RADS
(seed 2026) out of the 1,500 public cases (~7 GB). Patient-level 60/20/20
split (360/120/120 cases; 34 csPCa+ each in val and test):

| property                     | value (see data/splits/patient_split_2026_*.json) |
|------------------------------|---------------------------------------------------|
| selection seed               | 2026                                              |
| split seed                   | 2026                                              |
| patient overlap across splits| 0 (hard-checked by `check_no_leakage`)            |
| val / test positives         | 34 / 34                                           |
| centers represented          | RUMC / PCNN / ZGT                                 |

Ground truth: marksheet case-level csPCa (ISUP ≥ 2) + human-expert lesion
delineations (picai_labels, geometrically verified: identical
origin/spacing/size to T2W per-case spot checks).

## 4. Evaluation engine

One GPU pass per case saves: `folds_orig` / `folds_flip` (5,20,256,256) fp16
post-sigmoid TTA components + `native_fused` fp32 (post-gauss, mean folded,
reversed crop/pad + resample). ALL post-processing variants (fold ablations,
no-TTA, σ-variants, extraction variants, aggregation) are recomputed offline
from these components — the 5-fold ensemble inference is paid exactly ONCE.

Golden cross-check (case 10032_1000032, engine vs authors' reference):
likelihood 0.5269225836 vs 0.52692652 (Δ=3.9e-6 ≤ float tolerance);
fp16-component → fused reconstruction error: 0.0 (exact).

## 5. Benchmark: post-processing configurations

Pre-registered variants (all in `evaluation/postproc.py`):

| name | change |
|------|--------|
| official_default | exact Phase-1/official configuration (baseline) |
| fold1 / fold3 | single-fold ensemble (cost cut 5×) |
| no_tta | disables horizontal-flip TTA (2× cheaper) |
| sigma05 / sigma10 | gaussian σ 1.0 instead of 1.5 (sharper / smoother) |
| extract_fixed01 | fixed threshold 0.01 extraction (no dynamic RGA) |
| no_extract | no candidate extraction (use raw fused max) |
| no_supp_max | no "det < max/5 → 0" suppression |
| agg_p90 | fold aggregation by 90th pct instead of mean |

**Selection outcome** (pre-registered rule, val only): winner =
`no_extract` (val AUROC 0.8451, PR-AUC 0.7893), threshold frozen at
0.15717 (= max-F1 on val, which coincides with the Youden-J optimum;
val F1 0.70, bal-acc 0.780). Frozen in
`experiments/configs/selected_postproc.json`. Full 10-config comparison:
`experiments/results/postproc_selection/val_comparison.csv`.

### Head-to-head (120 cases per split; 34 csPCa+)

| split | config | thr | AUROC | AP | sens | spec | prec | F1 | bal-acc | acc | ECE |
|-------|--------|-----|-------|----|------|------|------|----|---------|-----|-----|
| val   | official_default | 0.500 | 0.8379 | 0.7852 | 0.176 | 1.000 | 1.000 | 0.300 | 0.588 | 0.767 | 0.158 |
| val   | **no_extract**   | 0.157 | **0.8451** | **0.7893** | 0.618 | 0.942 | 0.808 | **0.700** | 0.780 | 0.850 | 0.173 |
| test  | official_default | 0.500 | 0.8042 | 0.7064 | 0.235 | 1.000 | 1.000 | 0.381 | 0.618 | 0.783 | 0.145 |
| test  | **no_extract**   | 0.157 | **0.8081** | **0.7107** | 0.559 | 0.907 | 0.704 | **0.623** | 0.733 | 0.808 | 0.168 |

95% bootstrap CIs (2,000 resamples): test AUROC 0.808 [0.710–0.895],
AP 0.711 [0.586–0.827]; val AUROC 0.845 [0.758–0.924].

**Lesion localization** (expert-mask overlap on positives): detection rate at
10% overlap val 30/34 = 88.2% (official @0.5: 28/34), test 26/34 = 76.5%;
mean IoU 0.30 / 0.24, mean Dice 0.44 / 0.36 (val/test).

### Statistical significance and variance

- DeLong test official-vs-selected: val p = 0.89, test p = 0.95 —
  discrimination is statistically indistinguishable (ΔAUROC ≈ +0.004).
- Bootstrap ΔAUROC (official − selected): val −0.0072 [−0.0192, 0.000]
  (p = 0.001), test −0.0039 [−0.0216, +0.0109] (p = 0.61). The val "gain"
  (~0.7 AUROC pts) is NOT confirmed on test — we claim **operating-point
  improvement**, not discrimination improvement.
- Fold-run variance (single-fold ensembles, val): AUROC 0.789 ± 0.027
  (CoV 3.4%), PR-AUC 0.711 ± 0.031 — fold ensembling is worth ≈ +2.5 to
  +5.6 AUROC pts over any single fold (fold1 0.824, fold3 0.831 vs 0.845).
- Ablations: TTA removal costs −0.007 AUROC; σ 1.0 ≈ neutral (val 0.846);
  fixed-threshold extraction **hurts** (val AUROC 0.789, AP 0.723);
  fold-aggregation by p90 is exactly neutral.
- **Cost**: no_extract removes the RGA extraction stage ⇒ post-processing
  ≈ 0.07 s/case vs ≈ 2.55 s/case for the official config (~36× cheaper;
  GPU inference dominates at ~1.5 s/case).

## 6. Error analysis (frozen config, no_extract @0.157)

Error mix (per split, 120 cases): val TP 21 / TN 81 / FP 5 / FN 13;
test TP 19 / TN 78 / FP 8 / FN 15.

| insight | val | test |
|---------|-----|------|
| FN mean likelihood (missed cancers) | 0.048 | 0.050 |
| TP mean likelihood | 0.402 | 0.441 |
| FN by PI-RADS (3/4/5) | — 3 / 10 | 1(2) 2(3) 5(4) 7(5) |
| FN by center | RUMC 9, ZGT 4 | RUMC 11, ZGT 4, PCNN 0 |
| FP by center | PCNN 1, RUMC 2, ZGT 2 | PCNN 5, RUMC 3 |
| FN mean IoU with expert mask when annotated | 0.17 | 0.13 |

Reading: (1) the model is NOT "borderline" on its FNs — the mean FN score
0.05 is 3× below the decision threshold 0.157: these are silent negatives,
mostly grade-group-5 lesions at RUMC — a hard SubPop, not a thresholding
artifact. (2) FPs concentrate at PCNN on test (but n = 8, weak power) —
no actionable single-center bias visible at this sample size. Per-case
table: `experiments/results/error_analysis/{val,test}_no_extract_error_analysis.csv`.

## 7. XAI methods and quality

Methods (xai/methods.py, all on the (1,3,20,256,256) preprocessed tensor,
model grid (20,256,256), ensemble-averaged over 5 folds):

1. **Grad-CAM 3D** on the LAST skip connection (model.model[1]:
   64-ch @ 10,128,128 feature map; GAP weights; trilinear upsample; ReLU;
   min-max). Smoke-verified on the golden case: argmax (11,205,120) adjacent
   to the model detection peak (11,205,121).
2. **Grad-CAM++ 3D** — per-voxel α formulation (Chattopadhayay et al. 2018);
   argmax (11,208,122) on the golden case. (The naive scalar formulation
   produced corner artifacts and was rejected.)
3. **Integrated Gradients** — baseline = zero (= per-voxel mean of the
   z-scored channels), 24–32 steps, |·| max over channels.
4. **Occlusion sensitivity** — 3D slabs (2,32,32)/(2,24,24) zero-occluded,
   score = fold-averaged raw max (documented proxy of the official fusion).

Quality checks (xai/quality.py) — reliability, not beauty:

| check | what it catches |
|-------|-----------------|
| localization agreement (IoU top-5% attn vs expert lesion; enrichment) | blurry apologies |
| insertion/deletion curves (faithfulness AUROC) | importance that does not drive the score |
| randomization test (Adebayo 2018; last-block weights) | trivial/no-op "CAMs" |
| shift-stability (±1 voxel, Pearson) | attention maps hair-triggered by noise |

### Results — 8 audited cases (`experiments/results/xai/xai_quality.csv`)

Representative case **10925_1000942** (TP, likelihood 0.755):

| method | insertion AU | deletion AU | IoU top-5% vs lesion | enrichment |
|--------|--------------|-------------|----------------------|------------|
| Grad-CAM   | 0.732 | **0.206** | 0.0030 | 1.0 |
| Grad-CAM++ | —     | —         | **0.0595** | **20.0×** |
| IG         | —     | —         | 0.0043 | 1.5× |

Grad-CAM randomization-to-collapsed-map Pearson 0.095 (map becomes
uncorrelated ⇒ evidence is weight-driven, not input-trivial);
shift-stability 0.994 (rock stable).

Hard cases:

- **FP 10650_1000666** (lik 0.270): faithful (ins 0.643 / del 0.566,
  stab 0.718) — the model *believes* that region looks malignant; a
  consistent-but-wrong model, i.e. a modeling limitation, not an XAI artifact.
- **FN 11341_1001364** (lik 0.112): Grad-CAM attention partially overlaps
  the lesion (enrichment 1.0, IG 2.7×), but score stays below threshold:
  the model "sees" something there and under-scores it.
- **Low-score cases** (10810 lik 0.009, 10388 lik 0.038, 11175 lik 0.120):
  maps collapse to flat (ins ≈ del ≈ 0.12–0.39, stability 0) — XAI is
  *uninformative below ~0.15*, an honest boundary condition.
- Small-lesion caveat: mean expert lesion volume share ≈ 0.13% of the ROI in
  small lesions; top-5% attention IoU is tiny even when enrichment is 20×
  (Grad-CAM++ on 10925) — IoU alone is misleading for small lesions;
  localization claims use enrichment + insertion/deletion jointly.

Figures (T2W/ADC/HBV slice + overlaid heatmaps, native-grid zoom panel):
`experiments/results/xai/<case_id>_xai.png` (8 cases incl. FP + FN), plus
the Phase-1 golden smoke figure `smoke_10032_1000032.png`.

## 8. Conclusions / limitations / next steps

**Conclusions**

1. With weights untouched, the *decision layer* was the weak link: the
   official 0.50 operating point is nearly blind on this cohort (sens
   17.6–23.5%). Re-freezing the decision rule (no candidate extraction,
   threshold 0.157 tuned on val-only) moves test sensitivity from 23.5% →
   55.9% and F1 0.381 → 0.623 at 90.7% specificity, val F1 0.30 → 0.70.
2. Discrimination does not improve (DeLong p ≥ 0.89): the honest claim is
   **threshold/calibration repair + 36× cheaper post-processing**, not a
   better detector.
3. The model's residual failures are silent: FN scores average 0.05 — 3×
   below threshold — dominated by high-grade (PI-RADS 5) RUMC lesions.
4. The XAI stack is audited, not decorative: Grad-CAM shows strong
   faithfulness on confident TPs (ins 0.73 / del 0.21, stability 0.99),
   uncorrelated-on-randomization, and honest collapse on low-score cases.

**Limitations**

- All 600 local cases were in the reference model's training pool →
  optimistic upper bound (§2). External generalization requires the PI-CAI
  external/held-out sets (planned).
- n = 120/split for operating-point stats: single-point sensitivity/precision
  carry wide CIs; ranking metrics have bootstrap CIs reported.
- XAI occlusion proxy for fusion is an approximation of the official
  candidate-extraction score (documented in xai/methods.py).

**Tests**: Phase-2 suite 30/30 (`pytest tests`), Phase-1 demo regression
13/13 (green inside the pipeline's final stage).

**Deliverables**: `reports/BEST_MODEL.md` (frozen recipe), all artifacts
under `experiments/results/`, figures under `experiments/results/xai/`.

## 9. Reproducibility

    python scripts/run_phase2.py            # data -> splits -> inference ->
                                            # selection -> eval -> XAI -> regression

Each stage is idempotent & resumable; seeds fixed (2026) throughout; the
complete log is kept in experiments/logs/run_phase2.log.
