"""Loading of PI-CAI cases (images + ground truth) for the model demo.

Two source formats are supported:

1. The official repository test case (``third_party/picai_unet_gc_algorithm/test``):
   one case (``10032_1000032``) split into four SimpleITK ``.mha`` files,
   one per orientation/modality as required by the grand-challenge.org
   interfaces:

       test/images/transverse-t2-prostate-mri/*.mha
       test/images/transverse-adc-prostate-mri/*.mha
       test/images/transverse-hbv-prostate-mri/*.mha
       test/images/coronal-t2-prostate-mri/*.mha     (present, NOT used by the model)
       test/images/sagittal-t2-prostate-mri/*.mha    (present, NOT used by the model)

2. The PI-CAI Public Training and Development Dataset v2.0 (Zenodo record
   6624726), unzipped under ``data/demo_cases/<case_id>/``:

       <case_id>/<case_id>_t2w.nii.gz
       <case_id>/<case_id>_adc.nii.gz
       <case_id>/<case_id>_hbv.nii.gz
       <case_id>/<case_id>_cor.nii.gz     (present, NOT used by the model)
       <case_id>/<case_id>_sag.nii.gz     (present, NOT used by the model)

Ground truth for the demo:
- case-level clinical outcome (csPCa ISUP >= 2, PI-RADS, ISUP grade) from
  ``picai_labels/clinical_information/marksheet.csv``;
- expert lesion delineations from
  ``picai_labels/csPCa_lesion_delineations/human_expert/resampled/<case_id>.nii.gz``
  (resampled to the T2W grid, i.e. directly comparable with the detection map).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .config import OFFICIAL_TEST_CASE_DIR, MODALITY_SUFFIXES, MARKSHEET_PATH, DEMO_CASES_DIR


# The three axial modalities consumed by the model, in input order (T2W, ADC, HBV).
MODEL_MODALITIES = ["t2w", "adc", "hbv"]


@dataclass
class PicaIcase:
    """One MRI examination (one patient, one study) with its ground truth."""

    case_id: str                     # e.g. "10032_1000032"
    images: dict = field(default_factory=dict)   # modality -> sitk.Image
    clinical: dict = field(default_factory=dict) # whatever clinical info is at hand
    lesion_annotation: sitk.Image | None = None  # expert delineation (T2W grid)
    source: str = ""

    @property
    def t2w_shape(self):
        return self.images["t2w"].GetSize()  # (x, y, z) in SimpleITK convention


def _read_single(dir_path: Path, pattern: str) -> Path:
    matches = sorted(dir_path.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly 1 file matching {pattern} in {dir_path}, got {matches}")
    return matches[0]


def load_official_test_case() -> PicaIcase:
    """Load case 10032_1000032 shipped inside the official algorithm repo.

    This case is a real PI-CAI examination, shipped by the authors so users
    can validate their deployment. It comes with the reference detection map
    and the reference case-level likelihood (0.5269265...), which we use as
    our golden output comparison.
    """
    case = PicaIcase(case_id="10032_1000032", source="official_repo_test")
    case.images = {
        "t2w": sitk.ReadImage(str(_read_single(OFFICIAL_TEST_CASE_DIR / "images" / "transverse-t2-prostate-mri", "*.mha"))),
        "adc": sitk.ReadImage(str(_read_single(OFFICIAL_TEST_CASE_DIR / "images" / "transverse-adc-prostate-mri", "*.mha"))),
        "hbv": sitk.ReadImage(str(_read_single(OFFICIAL_TEST_CASE_DIR / "images" / "transverse-hbv-prostate-mri", "*.mha"))),
    }
    with open(OFFICIAL_TEST_CASE_DIR / "clinical-information-prostate-mri.json", encoding="utf-8") as f:
        case.clinical = json.load(f)
    # Reference outputs, for comparison with our inference:
    case.clinical["reference_case_level_likelihood"] = float(
        json.loads((OFFICIAL_TEST_CASE_DIR / "cspca-case-level-likelihood.json").read_text())
    )
    ref_map_path = OFFICIAL_TEST_CASE_DIR / "cspca-detection-map" / "cspca_detection_map.mha"
    if ref_map_path.exists():
        case.clinical["reference_detection_map"] = sitk.ReadImage(str(ref_map_path))
    return case


def load_case_from_dir(case_dir: Path, case_id: str | None = None) -> PicaIcase:
    """Load one PI-CAI examination from its folder (.mha from Zenodo or .nii.gz).

    If a patient folder holds several studies, ``case_id`` selects the right
    files (naming: <case_id>_<modality>.mha).
    """
    case_id = case_id or case_dir.name
    case = PicaIcase(case_id=case_id, source=str(case_dir))
    for mod in MODEL_MODALITIES:
        suffixes = MODALITY_SUFFIXES[mod]
        matches = []
        for suffix in suffixes:
            matches += sorted(case_dir.glob(f"{case_id}{suffix}"))
        if not matches:
            raise FileNotFoundError(
                f"missing {mod} ({[s for s in suffixes]}) for {case_id} in {case_dir}"
            )
        case.images[mod] = sitk.ReadImage(str(matches[0]))
    return case


def load_case(case_id_or_dir: str | Path) -> PicaIcase:
    """Dispatch: official test case by id, or a case id/directory under data/demo_cases.

    Cases are stored as data/demo_cases/<patient_id>/<case_id>_<modality>.mha
    (possibly several studies under one patient folder).
    """
    if str(case_id_or_dir) == "10032_1000032":
        return load_official_test_case()
    p = Path(case_id_or_dir)
    if p.is_dir():
        return load_case_from_dir(p)
    case_id = str(case_id_or_dir)
    patient_dir = DEMO_CASES_DIR / case_id.split("_")[0]
    if (patient_dir / f"{case_id}_t2w.mha").exists():
        return load_case_from_dir(patient_dir, case_id=case_id)
    if (patient_dir / f"{case_id}_t2w.nii.gz").exists():
        return load_case_from_dir(patient_dir, case_id=case_id)
    raise FileNotFoundError(f"no files for {case_id} found under {DEMO_CASES_DIR}")


# ---------------------------------------------------------------------------
# Clinical ground truth (case-level label) from the official marksheet
# ---------------------------------------------------------------------------


def list_available_cases(demo_dir: Path | None = None) -> list[str]:
    """Discover downloaded cases across data/demo_cases/<patient>/*.mha."""
    demo_dir = demo_dir or DEMO_CASES_DIR
    case_ids = []
    for patient_dir in sorted(demo_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        for f in sorted(patient_dir.glob("*_t2w.mha")):
            case_ids.append(f.name[: -len("_t2w.mha")])
    return case_ids


def load_marksheet() -> "object":
    """Return the official clinical marksheet as a pandas DataFrame."""
    import pandas as pd
    if not MARKSHEET_PATH.exists():
        raise FileNotFoundError(
            f"marksheet not found at {MARKSHEET_PATH}. Run scripts/download_or_prepare_data.py first."
        )
    return pd.read_csv(MARKSHEET_PATH)


def get_case_ground_truth(mark: "object", case_id: str) -> dict:
    """Case-level ground truth for one case, from the OFFICIAL marksheet columns.

    The marksheet of picai_labels already encodes the official outcome:
      case_ISUP   = max ISUP grade among lesions (0 if none)
      case_csPCa  = 'YES' if at least one lesion has ISUP >= 2, else 'NO'
    (protocol: histologically-confirmed ISUP>=2 positives and histologically
    or MRI-confirmed negatives; Saha et al., Lancet Oncology 2024).
    """
    row = mark.loc[mark["patient_id"].astype(str) == str(case_id).split("_")[0]]
    # several studies can share a patient id: disambiguate on study id
    if len(row) > 1:
        study_part = str(case_id).split("_")[-1]
        sel = row.loc[row["study_id"].astype(str) == study_part]
        if len(sel) == 1:
            row = sel
    if len(row) == 0:
        raise KeyError(f"case {case_id} not found in marksheet")
    r = row.iloc[0]

    isup_variant = r.get("case_ISUP", None)
    pirads = r.get("lesion_PIRADS", "")

    case_cspca_raw = str(r.get("case_csPCa", "")).strip().upper()

    return {
        "patient_id": str(r["patient_id"]),
        "histopath_type": r.get("histopath_type", ""),
        "raw_lesion_GS": r.get("lesion_GS", ""),
        "raw_lesion_ISUP": r.get("lesion_ISUP", ""),
        "max_ISUP": int(isup_variant) if _is_int_like(isup_variant) else None,
        "lesion_PIRADS": pirads if not _is_float_nan(pirads) else "",
        "psa": r.get("psa", None),
        "prostate_volume": r.get("prostate_volume", None),
        "psad": r.get("psad", None),
        "center": r.get("center", ""),
        "case_level_csPCa": bool(case_cspca_raw == "YES"),
    }


def _is_int_like(v) -> bool:
    try:
        int(v)
        return True
    except (TypeError, ValueError):
        return False


def _is_float_nan(v) -> bool:
    import math
    return isinstance(v, float) and math.isnan(v)


# ---------------------------------------------------------------------------
# Expert lesion annotations (pixel-level ground truth)
# ---------------------------------------------------------------------------

def load_lesion_annotation(case_id: str, resampled_dir: Path | None = None) -> sitk.Image | None:
    """Load the expert csPCa delineation, if available.

    Two official locations in picai_labels are checked:
      csPCa_lesion_delineations/human_expert/resampled/<case_id>.nii.gz
          multi-class: 0 = ISUP<=1, 2..5 = ISUP grade (T2W grid)
      csPCa_lesion_delineations/human_expert/Pooch25/<case_id>.nii.gz
          binary: 1 = ISUP>=2 (the 205 positives annotated by Pooch et al. 2025)
    """
    if resampled_dir is None:
        from .config import LABELS_DIR

        resampled_dir = (
            LABELS_DIR
            / "csPCa_lesion_delineations"
            / "human_expert"
        )
    for sub in ("resampled", "Pooch25"):
        p = Path(resampled_dir) / sub / f"{case_id}.nii.gz"
        if p.exists():
            return sitk.ReadImage(str(p))
    return None
