"""Shared paths and constants for the Phase-2 XAI pipeline (repo root)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEMO_DIR = REPO_ROOT / "prostate_model_demo"
WEIGHTS_DIR = DEMO_DIR / "third_party" / "picai_unet_gc_algorithm" / "weights"

DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "picai" / "images"
MASKS_DIR = DATA_DIR / "picai" / "label_masks"
SPLITS_DIR = DATA_DIR / "splits"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MARKSHEET_PATH = DEMO_DIR / "data" / "picai_labels" / "clinical_information" / "marksheet.csv"

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EXPERIMENTS_CONFIGS = EXPERIMENTS_DIR / "configs"
EXPERIMENTS_RESULTS = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"

REPORTS_DIR = REPO_ROOT / "reports"

# Reproduce the official preprocessing / inference specification (Phase 1).
IMG_SPEC = {
    "image_shape": [20, 256, 256],
    "spacing": [3.0, 0.5, 0.5],
    "num_channels": 3,
    "num_classes": 2,
}
UNET_STRIDES = [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
UNET_CHANNELS = [32, 64, 128, 256, 512, 1024]
N_FOLDS = 5

# Default random seed for all Phase-2 experiments (overridable per config).
DEFAULT_SEED = 2026
