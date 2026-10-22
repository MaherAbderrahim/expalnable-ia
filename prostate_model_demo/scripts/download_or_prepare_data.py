"""Download the demo data: PI-CAI cases, clinical marksheet, expert annotations.

What it downloads (and where):
  data/picai_labels/clinical_information/marksheet.csv
      - official clinical/case-level ground truth (PI-RADS, Gleason/ISUP, PSA...)
      - from https://github.com/DIAGNijmegen/picai_labels
  data/picai_labels/csPCa_lesion_delineations/human_expert/resampled/<case_id>.nii.gz
      - expert lesion contours (on the T2W grid) for the selected demo cases
  data/demo_cases/<patient_id>/<case_id>_<modality>.mha
      - the axial T2W / ADC / HBV scans for the selected demo cases, extracted
        from the PI-CAI Zenodo record 6624726 archives with HTTP RANGE requests
        (no need to download the full ~5.3 GB fold archives).

Licenses: images CC BY-NC 4.0 (Zenodo 6624726); labels CC BY-NC 4.0
(picai_labels). Non-commercial research only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, LABELS_DIR, DEMO_CASES_DIR  # noqa: E402

ZENODO_RECORD = 6624726
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
PICAI_LABELS_BASE = "https://raw.githubusercontent.com/DIAGNijmegen/picai_labels/main"
N_FOLDS_ZENODO = 5

# Default demo case selection (see docs/DATA_EXPLANATION.md)
DEFAULT_CASE_IDS = [
    "10032_1000032",  # the study also shipped with the official repo (offline copy available)
]


def _print(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# marksheet + expert annotations
# ---------------------------------------------------------------------------

def download_marksheet() -> Path:
    import requests
    url = f"{PICAI_LABELS_BASE}/clinical_information/marksheet.csv"
    out = LABELS_DIR / "clinical_information" / "marksheet.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        _print(f"[skip] marksheet already present: {out}")
        return out
    _print("[get ] marksheet from picai_labels ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    _print(f"[ok  ] saved {out}")
    return out


def download_expert_annotation(case_id: str) -> Path | None:
    """Fetch the expert lesion delineation for one case.

    Two official locations exist in picai_labels:
      - csPCa_lesion_delineations/human_expert/resampled/<case_id>.nii.gz
        (multi-class: 0 = ISUP<=1, 2..5 = ISUP grade of the lesion)
      - csPCa_lesion_delineations/human_expert/Pooch25/<case_id>.nii.gz
        (binary: 1 = ISUP>=2; the 205 positives newly annotated by experts,
         Pooch et al., 2025)
    Both are provided on the T2W grid when in resampled layout.
    """
    import requests
    candidates = [
        ("resampled (ISUP 2..5 labels)",
         f"{PICAI_LABELS_BASE}/csPCa_lesion_delineations/human_expert/resampled/{case_id}.nii.gz"),
        ("Pooch25 (binary ISUP>=2 labels)",
         f"{PICAI_LABELS_BASE}/csPCa_lesion_delineations/human_expert/Pooch25/{case_id}.nii.gz"),
    ]
    for label, url in candidates:
        out = LABELS_DIR / url.split("/picai_labels/main/")[-1]
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            _print(f"[skip] expert annotation already present: {out.relative_to(LABELS_DIR)}")
            return out
    for label, url in candidates:
        out = LABELS_DIR / url.split("/picai_labels/main/")[-1]
        r = requests.get(url, timeout=180)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        out.write_bytes(r.content)
        _print(f"[ok  ] saved {out.relative_to(LABELS_DIR)} ({len(r.content) / 1e3:.0f} kB)"
               f" [{label}]")
        return out
    _print(f"[warn] no expert annotation file for {case_id} (checked both locations)")
    return None


# ---------------------------------------------------------------------------
# images: partial download from the 5 Zenodo fold archives
# ---------------------------------------------------------------------------

_MODALITIES = ("t2w", "adc", "hbv")  # coronal/sagittal T2W are NOT used by the model


def locate_cases(case_ids: list[str]) -> dict[str, str | None]:
    """Scan the central directories of the 5 fold archives; return case -> ref.

    ref format: 'fold{N}::picai_public_images_fold{N}.zip::<zip path>'
    Range requests make each listing cheap (a few hundred kB per archive).
    """
    from remotezip import RemoteZip

    wanted = {cid: None for cid in case_ids}
    case_to_patient = {cid: cid.split("_")[0] for cid in case_ids}

    for fold in range(N_FOLDS_ZENODO):
        name_part = f"picai_public_images_fold{fold}.zip"
        url = f"{ZENODO_URL}/files/{name_part}/content"
        with RemoteZip(url) as rz:
            for name in rz.namelist():
                zippath = Path(name)
                patient = zippath.parts[0]
                for m in _MODALITIES:
                    if zippath.name.endswith(f"_{m}.mha"):
                        cid = zippath.name[: -len(f"_{m}.mha")]
                        break
                else:
                    continue
                if cid in wanted and wanted[cid] is None and case_to_patient[cid] == patient:
                    wanted[cid] = f"fold{fold}::{name_part}::{name}"
    remaining = [c for c, r in wanted.items() if r is None]
    _print(f"[loc ] located {len(case_ids) - len(remaining)}/{len(case_ids)} case(s)"
           + (f"; not found: {remaining}" if remaining else ""))
    return wanted


def download_case_images(case_id: str, ref: str) -> Path:
    """Fetch one case (T2W + ADC + HBV only) via HTTP range requests."""
    from remotezip import RemoteZip

    patient_id = case_id.split("_")[0]
    case_dir = DEMO_CASES_DIR / patient_id
    case_dir.mkdir(parents=True, exist_ok=True)

    _, foldzip, zippath = ref.split("::", 2)
    prefix = str(Path(zippath).parent)
    url = f"{ZENODO_URL}/files/{foldzip}/content"

    with RemoteZip(url) as rz:
        for mod in _MODALITIES:
            entry = f"{prefix}/{case_id}_{mod}.mha"
            out = case_dir / f"{case_id}_{mod}.mha"
            if out.exists():
                _print(f"[skip] {out.name} already present")
                continue
            _print(f"[get ] {foldzip}/{entry} ...")
            data = rz.read(entry)
            out.write_bytes(data)
            _print(f"[ok  ] {out} ({len(data) / 1e6:.1f} MB)")
    return case_dir


# ---------------------------------------------------------------------------
# case selection helper
# ---------------------------------------------------------------------------

def pick_cases(n_extra: int) -> list[str]:
    """Pick alternating positive/negative annotated cases from the marksheet."""
    import csv
    mpath = LABELS_DIR / "clinical_information" / "marksheet.csv"
    if not mpath.exists():
        raise FileNotFoundError(mpath)
    with open(mpath, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def is_positive(row) -> bool:
        gs = (row.get("lesion_GS") or "").strip().lower()
        if gs in ("", "nan", "n/a"):
            return False
        for part in gs.split(","):
            try:
                a, b = part.split("+")
            except ValueError:
                continue
            try:
                if int(a) + int(b) >= 7:
                    return True
            except ValueError:
                continue
        return False

    positives = [r for r in rows if is_positive(r)]
    negatives = [r for r in rows if not is_positive(r)]
    half_pos = n_extra // 2
    picks = positives[:half_pos] + negatives[: n_extra - half_pos]
    return [f"{r['patient_id']}_{r['study_id']}" for r in picks]


def main() -> None:
    ap = argparse.ArgumentParser(description="Download/prepare the demo data subset.")
    ap.add_argument(
        "--cases", nargs="*", default=None,
        help="case ids as <patient>_<study>, e.g. 10032_1000032 (default: saved selection).",
    )
    ap.add_argument(
        "--add-case-count", type=int, default=0,
        help="pick this many more annotated cases (alternating positive/negative) from the marksheet.",
    )
    args = ap.parse_args()

    selection_file = DATA_DIR / "demo_cases.json"
    case_ids = args.cases or (
        json.loads(selection_file.read_text())["case_ids"] if selection_file.exists() else DEFAULT_CASE_IDS
    )
    if args.add_case_count > 0:
        downloaded_marksheet = download_marksheet()
        case_ids = case_ids + pick_cases(args.add_case_count)
    else:
        download_marksheet()

    # persist the selection
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    selection_file.write_text(json.dumps({"case_ids": case_ids}, indent=2))
    _print(f"[sel ] demo cases: {case_ids}")

    _print("=" * 60)
    _print("PI-CAI DEMO DATA DOWNLOADER")
    _print("=" * 60)

    located = locate_cases(case_ids)

    ok = 0
    for cid, ref in located.items():
        if ref is None:
            continue
        try:
            download_case_images(cid, ref)
            download_expert_annotation(cid)
            ok += 1
        except Exception as e:  # keep going for the remaining cases
            _print(f"[fail] {cid}: {type(e).__name__}: {e}")

    _print("-" * 60)
    _print(f"done: {ok}/{len(located)} case(s) available under {DEMO_CASES_DIR}")


if __name__ == "__main__":
    main()
