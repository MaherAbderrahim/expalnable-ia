"""Phase 2 — data preparation: stratified PI-CAI case selection + robust download.

What it does
------------
1. Selects a stratified subset of PI-CAI public cases from the official marksheet
   (patient-level: at most one study per patient; stratification by center and
   csPCa status, with max-lesion PI-RADS buckets for positives).
2. Writes a reproducible selection manifest (seed, criteria, per-case metadata).
3. Downloads the axial T2W / ADC / HBV images (.mha) from the Zenodo record
   6624726 fold archives using HTTP range requests (no full-archive download),
   with resume-safety: existing valid files are skipped, retries with backoff.
4. Downloads the expert lesion delineations (picai_labels, resampled grid first,
   Pooch25 fallback) for every selected case.

Licenses: images CC BY-NC 4.0 (Zenodo 6624726); labels CC BY-NC 4.0.
Non-commercial research only.

Usage
-----
  python scripts/prepare_data.py --cases 600 --seed 2026 --parallel 4
  python scripts/prepare_data.py --select-only          # manifest without download
  python scripts/prepare_data.py --skip-images --skip-annotations
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.paths import DATA_DIR, LOGS_DIR, MARKSHEET_PATH  # noqa: E402

ZENODO_RECORD = 6624726
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
N_FOLDS_ZENODO = 5
MODALITIES = ("t2w", "adc", "hbv")  # coronal/sagittal T2W are NOT used by the model

IMAGES_DIR = DATA_DIR / "picai" / "images"
MASKS_DIR = DATA_DIR / "picai" / "label_masks"
SELECTION_DIR = DATA_DIR / "splits"
LOG_PATH = LOGS_DIR / "prepare_data.log"

_log_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Official marksheet + stratified selection
# ---------------------------------------------------------------------------

def load_marksheet():
    import pandas as pd

    if not MARKSHEET_PATH.exists():
        raise FileNotFoundError(
            f"marksheet not found at {MARKSHEET_PATH}; the Phase-1 demo already bundles it."
        )
    df = pd.read_csv(MARKSHEET_PATH)
    df["case_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)
    return df


def max_pirads(pirads_value) -> int:
    """max PI-RADS grade of the case from the comma-separated per-lesion list."""
    if isinstance(pirads_value, float):  # NaN
        return 0
    vals = re.findall(r"\d+", str(pirads_value))
    return max((int(v) for v in vals), default=0)


def select_cases(df, n_target: int, seed: int) -> list[str]:
    """Patient-level stratified selection of case_ids.

    - one case per patient (23 patients have >=2 studies: keep the first listed)
    - positives kept at their natural prevalence (~28.3 %), stratified by
      (center, max PI-RADS bucket) using largest-remainder allocation
    - negatives stratified by (center, max PI-RADS bucket)
    - deterministic given (seed, n_target)
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    df = df.copy()

    # patient-level: keep one study per patient (first listed study)
    first_study = df.groupby("patient_id", as_index=False).first()
    df = df[df["case_id"].isin(first_study["case_id"])].copy()

    df["positive"] = (df["case_csPCa"].astype(str) == "YES").astype(int)
    df["max_pirads"] = df["lesion_PIRADS"].map(max_pirads)
    df["pirads_bucket"] = df["max_pirads"].clip(upper=5).astype(str)

    n_pos = round(n_target * df["positive"].mean())
    n_pos = min(n_pos, (df["positive"] == 1).sum())
    n_neg = n_target - n_pos

    def stratified_pick(sub: "object", k: int) -> list[str]:
        """proportional allocation by (center, pirads_bucket), largest remainder."""
        if k <= 0 or len(sub) == 0:
            return []
        groups = sub.groupby(["center", "pirads_bucket"]).size().reset_index(name="n")
        groups = groups.sort_values("n", ascending=False).reset_index(drop=True)
        # start with the floor and distribute the remaining seats to the
        # groups with the largest fractional remainders among available cases
        remaining_sub = sub.copy()
        picked: list[str] = []
        remain = k
        raw = (groups["n"] / groups["n"].sum() * remain)
        floors = np.floor(raw).astype(int)
        frac = raw - floors
        # allocate floor seats (bounded by group size)
        for row_idx, want in enumerate(floors):
            row = groups.iloc[row_idx]
            sel_df = remaining_sub[
                (remaining_sub["center"] == row["center"])
                & (remaining_sub["pirads_bucket"] == row["pirads_bucket"])
            ]
            take = min(int(want), len(sel_df))
            if take > 0:
                idx = rng.permutation(sel_df.index)[:take]
                picked += list(sel_df.loc[idx, "case_id"])
                remaining_sub = remaining_sub.drop(index=idx)
                remain -= take
        # distribute leftovers by largest fractional remainder
        order = frac.sort_values(ascending=False).index
        for g_idx in order:
            if remain <= 0:
                break
            row = groups.loc[g_idx]
            sel_df = remaining_sub[
                (remaining_sub["center"] == row["center"])
                & (remaining_sub["pirads_bucket"] == row["pirads_bucket"])
            ]
            if len(sel_df) > 0:
                idx = [sel_df.index[0]]
                picked += [sel_df.loc[idx[0], "case_id"]]
                remaining_sub = remaining_sub.drop(index=idx)
                remain -= 1
        return picked

    pos = df[df["positive"] == 1]
    neg = df[df["positive"] == 0]
    pos_ids = stratified_pick(pos, min(n_pos, len(pos)))
    neg_ids = stratified_pick(neg, min(n_neg, len(neg)))

    selected = pos_ids + neg_ids
    rng.shuffle(selected)
    return sorted(selected)


def build_manifest(df, case_ids: list[str], seed: int, n_target: int) -> None:
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    sub = df[df["case_id"].isin(case_ids)][
        ["case_id", "patient_id", "study_id", "center", "case_csPCa", "case_ISUP",
         "lesion_PIRADS"]
    ].copy()
    sub["max_pirads"] = sub["lesion_PIRADS"].map(max_pirads)
    sub = sub.sort_values("case_id")

    manifest = {
        "seed": int(seed),
        "n_target": int(n_target),
        "n_selected": len(case_ids),
        "n_positive": int((sub["case_csPCa"] == "YES").sum()),
        "criteria": (
            "patient-level (one study per patient); stratified by center and "
            "csPCa status; positives keep natural prevalence; max-lesion "
            "PI-RADS parsed from the per-lesion list"
        ),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "case_ids": sorted(case_ids),
    }
    (SELECTION_DIR / f"case_selection_seed{seed}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    sub.to_csv(SELECTION_DIR / f"case_selection_seed{seed}.csv", index=False)
    log(f"[sel ] manifest saved: {len(case_ids)} cases "
        f"({manifest['n_positive']} csPCa+) -> {SELECTION_DIR}")


# ---------------------------------------------------------------------------
# Image download (range requests, resume-safe, parallel)
# ---------------------------------------------------------------------------

def _zenodo_file_url(zip_name: str) -> str:
    return f"{ZENODO_URL}/files/{zip_name}/content"


def locate_cases(case_ids: list[str]) -> dict[str, str]:
    """Scan the 5 Zenodo fold archives once; map case_id -> zip file entry.

    Returns {case_id: '<zip_name>::<zip path prefix>'} or case missing.
    """
    from remotezip import RemoteZip

    wanted = {c: None for c in case_ids}
    patient_of = {c: c.split("_")[0] for c in case_ids}
    for fold in range(N_FOLDS_ZENODO):
        if not any(v is None for v in wanted.values()):
            break
        name = f"picai_public_images_fold{fold}.zip"
        url = _zenodo_file_url(name)
        log(f"[loc ] scanning {name} ...")
        attempts = 0
        while True:
            try:
                with RemoteZip(url) as rz:
                    for name_ in rz.namelist():
                        parts = Path(name_).parts
                        if len(parts) < 2:
                            continue
                        patient, fname = parts[0], parts[-1]
                        m = re.search(r"_(t2w|adc|hbv)\.mha$", fname)
                        if not m:
                            continue
                        cid = fname[: -(len(m.group(0)))]
                        if cid in wanted and wanted[cid] is None and patient_of[cid] == patient:
                            wanted[cid] = f"{name}::{str(Path(name_).parent)}"
                break
            except Exception as e:  # noqa: BLE001
                attempts += 1
                if attempts >= 3:
                    raise
                log(f"[warn] {name} scan failed ({e}); retry in 20 s")
                time.sleep(20)
    found = sum(1 for v in wanted.values() if v)
    log(f"[loc ] located {found}/{len(case_ids)} cases across 5 fold archives")
    return {c: v for c, v in wanted.items() if v}


def _valid_mha(path: Path, expected_size: int) -> bool:
    """cheap resume-safety check: size + 'ObjectType' magic in the header."""
    try:
        if path.stat().st_size != expected_size:
            return False
        with open(path, "rb") as f:
            head = f.read(256)
        return b"ObjectType" in head or b"IT" in head
    except OSError:
        return False


def download_case_images(case_id: str, ref: str, parallel: int = 1) -> list[str]:
    from remotezip import RemoteZip
    from urllib.error import HTTPError

    zip_name, prefix = ref.split("::", 1)
    url = _zenodo_file_url(zip_name)
    patient_dir = IMAGES_DIR / case_id.split("_")[0]
    patient_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []
    attempts = 0
    while True:
        ok = True
        try:
            with RemoteZip(url) as rz:  # central directory room ~a few 100 kB
                entries = {}
                for mod in MODALITIES:
                    zip_path = f"{prefix}/{case_id}_{mod}.mha"
                    info = rz.getinfo(zip_path)
                    out = patient_dir / f"{case_id}_{mod}.mha"
                    if out.exists() and _valid_mha(out, info.file_size):
                        out_paths.append(str(out))
                        continue
                    entries[zip_path] = (out, info.file_size)
                for zip_path, (out, _sz) in entries.items():
                    data = rz.read(zip_path)
                    # integrity: zip.read validates CRC32 inherently
                    out.write_bytes(data)
                    out_paths.append(str(out))
            break
        except (HTTPError, OSError, Exception) as e:  # noqa: BLE001
            attempts += 1
            if attempts >= 4:
                log(f"[fail] {case_id}: {type(e).__name__}: {e}")
                raise
            backoff = 15 * attempts
            log(f"[warn] {case_id} attempt {attempts} failed ({e}); backoff {backoff}s")
            time.sleep(backoff)
    return out_paths


def download_images(case_to_ref: dict[str, str], parallel: int) -> None:
    done = [0]
    fails = []
    t0 = time.time()

    def progress():
        with _log_lock:
            done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(case_to_ref):
            rate = done[0] / max(time.time() - t0, 1e-9)
            log(f"[dl  ] images {done[0]}/{len(case_to_ref)} cases "
                f"({rate:.1f} cases/s)")

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futures = {
            ex.submit(download_case_images, cid, ref): cid for cid, ref in case_to_ref.items()
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                fut.result()
                progress()
            except Exception as e:  # noqa: BLE001
                fails.append((cid, f"{type(e).__name__}: {e}"))
                log(f"[fail] {cid}: {e}")
    if fails:
        log(f"[dl  ] {len(fails)} failure(s): {[cid for cid, _ in fails]}")
        for cid, err in fails:
            log(f"[fail-detail] {cid}: {err}")


# ---------------------------------------------------------------------------
# Expert annotations (picai_labels)
# ---------------------------------------------------------------------------

PICAI_LABELS_BASE = "https://raw.githubusercontent.com/DIAGNijmegen/picai_labels/main"


def download_annotation(case_id: str, attempts: int = 3) -> str | None:
    import requests

    candidates = [
        ("resampled", f"{PICAI_LABELS_BASE}/csPCa_lesion_delineations/human_expert/resampled/{case_id}.nii.gz"),
        ("Pooch25", f"{PICAI_LABELS_BASE}/csPCa_lesion_delineations/human_expert/Pooch25/{case_id}.nii.gz"),
    ]
    for label, url in candidates:
        out = MASKS_DIR / f"{case_id}.nii.gz"
        if out.exists() and out.stat().st_size > 0:
            return f"{label}::cached"
        for i in range(attempts):
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 404:
                    break  # try next candidate
                r.raise_for_status()
                MASKS_DIR.mkdir(parents=True, exist_ok=True)
                out.write_bytes(r.content)
                return f"{label}::downloaded"
            except Exception:  # noqa: BLE001
                if i == attempts - 1:
                    break
                time.sleep(5 * (i + 1))
    return None


def download_annotations(case_ids: list[str], parallel: int = 8) -> None:
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for cid, res in zip(case_ids, ex.map(download_annotation, case_ids)):
            results[cid] = res
    got = sum(1 for v in results.values() if v is not None)
    log(f"[masks] annotations ready: {got}/{len(case_ids)} in {time.time()-t0:.0f}s")
    missing = [c for c, v in results.items() if v is None]
    if missing:
        log(f"[masks] missing annotations: {missing}")
    (SELECTION_DIR / "mask_presence.json").write_text(
        json.dumps({k: v for k, v in sorted(results.items())}, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 data preparation (stratified PI-CAI subset).")
    ap.add_argument("--cases", type=int, default=600, help="number of cases to select (default 600)")
    ap.add_argument("--seed", type=int, default=2026, help="selection seed (default 2026)")
    ap.add_argument("--parallel", type=int, default=4, help="parallel image downloads (default 4)")
    ap.add_argument("--select-only", action="store_true", help="only write the selection manifest")
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-annotations", action="store_true")
    ap.add_argument("--reuse-manifest", action="store_true",
                    help="reuse an existing selection manifest instead of re-selecting")
    args = ap.parse_args()

    try:
        t0 = time.time()
        seed_file = SELECTION_DIR / f"case_selection_seed{args.seed}.json"
        if args.reuse_manifest and seed_file.exists():
            selected = json.loads(seed_file.read_text(encoding="utf-8"))["case_ids"]
            log(f"[sel ] reusing existing manifest: {seed_file.name} ({len(selected)} cases)")
        else:
            df = load_marksheet()
            selected = select_cases(df, args.cases, args.seed)
            build_manifest(df, selected, args.seed, args.cases)
        if args.select_only:
            return

        if not args.skip_images:
            refs = locate_cases(selected)
            missing = sorted(set(selected) - set(refs))
            if missing:
                log(f"[warn] {len(missing)} case(s) not found in fold archives: {missing[:8]} ...")
            download_images(refs, parallel=args.parallel)

        if not args.skip_annotations:
            download_annotations(selected, parallel=8)

        log(f"[done] prepare_data finished in {(time.time()-t0)/60:.1f} min")
    except Exception:
        log("[FATAL] " + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
