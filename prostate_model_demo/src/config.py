"""Paths and constants for the PI-CAI baseline U-Net model demo.

All scientific constants (input resolution, spacing, model hyperparameters)
are taken verbatim from the official inference script:
    https://github.com/DIAGNijmegen/picai_unet_gc_algorithm/blob/main/process.py
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

THIRD_PARTY_DIR = PROJECT_ROOT / "third_party" / "picai_unet_gc_algorithm"
WEIGHTS_DIR = THIRD_PARTY_DIR / "weights"          # unet_F0.pt ... unet_F4.pt (LFS)

DATA_DIR = PROJECT_ROOT / "data"
OFFICIAL_TEST_CASE_DIR = THIRD_PARTY_DIR / "test"  # case 10032_1000032 (mha, bundled)
DEMO_CASES_DIR = DATA_DIR / "demo_cases"           # cases downloaded from Zenodo
LABELS_DIR = DATA_DIR / "picai_labels"             # ground truth (lesion contours + marksheet)
MARKSHEET_PATH = LABELS_DIR / "clinical_information" / "marksheet.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Official inference setup (verbatim from process.py)
# ---------------------------------------------------------------------------
# Model input specification:
#   image_shape = (slices, rows, columns) after preprocessing
#   spacing     = (1 / 1 / 1) mm spacing applied as (z, y, x)
IMG_SPEC = {
    "image_shape": [20, 256, 256],   # 3D matrix size after crop/resample/pad
    "spacing": [3.0, 0.5, 0.5],      # mm, (slice, row, column)
    "num_channels": 3,               # T2W + ADC + HBV
    "num_classes": 2,                # channel 0: background, channel 1: csPCa
}

# Model hyperparameters (verbatim from picai_baseline default_hyperparam.py)
UNET_HYPERPARAMS = {
    "model_strides": [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
    "model_features": [32, 64, 128, 256, 512, 1024],
}

# Number of ensemble members (5-fold cross-validation ensemble)
N_FOLDS = 5

# Case-level decision threshold used by PI-CAI evaluation (Saha et al.,
# Lancet Oncology 2024: threshold 0.5 corresponds to sensitivity 70.9 %
# and PPV 35.5 % on the PI-CAI test set).
CASE_LEVEL_THRESHOLD = 0.5

# Modality file suffixes: PI-CAI Zenodo files are .mha, labels are .nii.gz
MODALITY_SUFFIXES = {
    "t2w": ("_t2w.mha", "_t2w.nii.gz"),
    "adc": ("_adc.mha", "_adc.nii.gz"),
    "hbv": ("_hbv.mha", "_hbv.nii.gz"),
}
