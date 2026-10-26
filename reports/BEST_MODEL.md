# BEST_MODEL — Phase-2 frozen deliverable

## The frozen model (what "best" means here)

**Weights: unchanged — the official PI-CAI 5-fold UNet ensemble** (`unet_F0..F4.pt`,
32.5 M params/fold). Phase 2 does not retrain the network (user constraint: no
GPU training); the improvement lives entirely in the **post-processing /
decision layer**.

**Frozen configuration (chosen on VALIDATION only, test scored once):**

| element | official (Phase 1) | frozen (Phase 2) |
|---|---|---|
| ensemble | 5-fold mean | 5-fold mean (unchanged) |
| TTA (h-flip) | yes (official) | yes |
| gaussian σ | 1.5 | 1.5 |
| candidate extraction | RGA `dynamic` + ring suppression (det<max/5→0) | **none (`no_extract`)** — raw fused max is used |
| decision threshold | 0.50 | **0.1572** (max-F1 on validation) |

Selection rule (pre-registered before test was touched): max validation
PR-AUC (tie-break AUROC) → threshold = max F1 on validation. Written in
experiments/configs/selected_postproc.json.

## Why this wins — the honest decomposition

Discrimination (ranking) does NOT change: DeLong p = 0.95, bootstrap ΔAUROC
−0.004 [−0.022, +0.011] — the ensemble never "sees" differently.

What changes is: (1) the operating point the raw histogram actually supports
(off-the-shelf 0.50 was nearly blind: 17–24% sensitivity), and (2) the
expensive RGA extraction stage (≈2.5 s/case, ~50× post-processing cost) whose
max/5 suppression *discarded* true candidates: extracting candidates
contributed nothing once the raw fused map is used directly.

## Test split results (1st and only test run, 2026-10-23)

| metric | official @0.50 | frozen (no_extract @0.157) |
|---|---|---|
| AUROC | 0.8042 | 0.8081 (CI 0.710–0.895) |
| AP/PR-AUC | 0.7064 (CI 0.580–0.825) | 0.7107 (CI 0.586–0.827) |
| sensitivity | 0.235 | 0.559 |
| specificity | 1.000 | 0.907 |
| precision | 1.000 | 0.704 |
| F1 | 0.381 | 0.623 |
| balanced accuracy | 0.618 | 0.733 |
| accuracy | 0.783 | 0.808 |
| ECE (10 bins) | 0.145 | 0.168 |
| error mix (120 cases) | TP 8, TN 93, FP 0, FN 26 | **TP 19, TN 78, FP 8, FN 15** |
| postproc cost/case | ≈2.55 s (RGA) | **≈0.07 s** |

Validation split: F1 0.30 → 0.70, balanced accuracy 0.588 → 0.780 (the
threshold 0.1572 was optimized THERE — it generalizes to test: F1 0.623).

## Variance / stability evidence

- single-fold ensembles (val): AUROC 0.789 ± 0.027 (cov 3.4%) — the fold
  ensembling accounts for ~2.5 AUROC points over fold1 (0.824 → 0.845 val).
- TTA matters little (val ΔAUROC −0.0003); σ ablation ±0.01; fixed-threshold
  extraction HURTS badly (AP 0.789 → 0.723) — the dynamic RGA step was the
  official pipeline's "feature", but the raw fused max scores the same.
- Phase-1 PGU golden test still passes (13/13 demo tests; engine likelihood
  Δ = 3.9e-6 vs authors' reference).

## What this "best model" is NOT

- NO local training was done; all 600 local cases were in the reference
  model's training pool → our numbers are an OPTIMISTIC upper bound of
  field performance; the published hidden-set numbers (AUROC 0.878, sens
  70.9 % @ PPV 35.5 %) remain the honest external reference.
- Radical operating-point shifts trade specificity for sensitivity.

## Reproduce

    python scripts/run_phase2.py            # full pipeline, resumable
    INFO: experiments/results/final_evaluation/eval_final_summary.json
