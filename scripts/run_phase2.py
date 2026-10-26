"""One-command Phase-2 pipeline (idempotent, resumable):

    prepare_data -> build_splits -> run_inference (val+test)
        -> select_config -> eval_final -> explain -> Phase-1 regression test

Usage:
    python scripts/run_phase2.py                 # run everything missing
    python scripts/run_phase2.py --skip-download # data already on disk
    python scripts/run_phase2.py --from eval     # resume at a given stage

Stages are skipped automatically when their outputs already exist; pass
--force <stage> to re-run a stage. Everything is logged to
experiments/logs/run_phase2.log in addition to stdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.paths import (  # noqa: E402
    IMAGES_DIR, PREDICTIONS_DIR, SPLITS_DIR, LOGS_DIR,
)

STAGES = ["download", "splits", "inference", "select", "eval", "explain", "regression"]

DEMO_VENV_PY = REPO_ROOT / "prostate_model_demo" / ".venv" / "Scripts" / "python.exe"
PYTHON = sys.executable if not DEMO_VENV_PY.exists() else str(DEMO_VENV_PY)


def _log(msg: str, log_fh) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


def _run(cmd: list[str], stage: str, log_fh, workdir: Path = REPO_ROOT) -> None:
    import os as _os
    import subprocess as _sp

    _log(f"[{stage}] $ {' '.join(str(c) for c in cmd)}", log_fh)
    env = {**_os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = _sp.Popen([str(c) for c in cmd], cwd=str(workdir), env=env,
                     stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,
                     encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        log_fh.write(line if line.endswith("\n") else line + "\n")
        log_fh.flush()
    proc.wait()
    if proc.returncode != 0:
        _log(f"[{stage}] FAILED (exit {proc.returncode})", log_fh)
        raise SystemExit(1)
    _log(f"[{stage}] OK", log_fh)


def stage_done(stage: str) -> bool:
    split_json = sorted(SPLITS_DIR.glob("patient_split_*.json"))
    if stage == "download":
        manifest = SPLITS_DIR / "case_selection_seed2026.json"
        if not manifest.exists():
            return False
        sel = json.loads(manifest.read_text(encoding="utf-8"))
        ids = sel.get("case_ids", [])
        have = sum(1 for c in ids
                   if (IMAGES_DIR / c.split("_")[0] / f"{c}_t2w.mha").exists())
        # annotations phase finished -> mask_presence.json exists (negative
        # cases legitimately have NO mask file, so count() is not a proxy)
        annotations_done = (SPLITS_DIR / "mask_presence.json").exists()
        return bool(ids) and have == len(ids) and annotations_done
    if stage == "splits":
        return bool(split_json)
    if stage == "inference":
        return (any(PREDICTIONS_DIR.glob("val/*.npz"))
                and any(PREDICTIONS_DIR.glob("test/*.npz")))
    if stage == "select":
        return (Path("experiments/configs/selected_postproc.json")).exists()
    if stage == "eval":
        return (Path("experiments/results/final_evaluation/eval_final_summary.json")).exists()
    if stage == "explain":
        return (Path("experiments/results/xai/xai_quality.csv")).exists()
    if stage == "regression":
        return False  # always re-run
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start_at", choices=STAGES, default=None,
                    help="force resuming from this stage (later done-stages skip)")
    ap.add_argument("--force", nargs="*", choices=STAGES, default=[],
                    help="re-run these stages even if their outputs exist")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--explain-cases", type=int, default=12)
    args = ap.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "run_phase2.log"
    mode = "a" if log_path.exists() else "w"
    with open(log_path, mode, encoding="utf-8") as log_fh:
        _log(f"=== run_phase2 start (python={PYTHON}) ===", log_fh)

        order = STAGES
        if args.start_at:
            order = STAGES[STAGES.index(args.start_at):]

        for stage in order:
            if stage == "download":
                if args.skip_download:
                    _log("[download] skipped by flag", log_fh)
                    continue
                if stage_done("download") and "download" not in args.force:
                    _log("[download] data complete -> skip", log_fh)
                    continue
                _run([PYTHON, "scripts/prepare_data.py", "--reuse-manifest"],
                     "download", log_fh)
                continue
            if stage == "splits":
                if stage_done("splits") and "splits" not in args.force:
                    _log("[splits] split file exists -> skip", log_fh)
                    continue
                _run([PYTHON, "scripts/build_splits.py"], "splits", log_fh)
                continue
            if stage == "inference":
                if stage_done("inference") and "inference" not in args.force:
                    _log("[inference] predictions exist -> skip", log_fh)
                    continue
                _run([PYTHON, "scripts/run_inference.py", "--split", "val",
                      "--split", "test"], "inference", log_fh)
                continue
            if stage == "select":
                if stage_done("select") and "select" not in args.force:
                    _log("[select] selected_postproc.json exists -> skip", log_fh)
                    continue
                _run([PYTHON, "scripts/select_config.py", "--workers", "4"],
                     "select", log_fh)
                continue
            if stage == "eval":
                if stage_done("eval") and "eval" not in args.force:
                    _log("[eval] metrics_final.json exists -> skip", log_fh)
                    continue
                _run([PYTHON, "scripts/eval_final.py", "--workers", "4"],
                     "eval", log_fh)
                continue
            if stage == "explain":
                if stage_done("explain") and "explain" not in args.force:
                    _log("[explain] xai_quality.csv exists -> skip", log_fh)
                    continue
                _run([PYTHON, "scripts/explain.py",
                      "--n-cases", str(args.explain_cases)],
                     "explain", log_fh)
                continue
            if stage == "regression":
                _run([PYTHON, "-m", "pytest", "tests", "-q"],
                     "phase1-regression", log_fh,
                     workdir=REPO_ROOT / "prostate_model_demo")
                continue
        _log("=== run_phase2 done ===", log_fh)


if __name__ == "__main__":
    main()
