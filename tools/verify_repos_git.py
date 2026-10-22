"""Verify repos using git's partial clone (no GitHub API rate limit).

`git clone --filter=blob:none --no-checkout` downloads only the commit/tree
objects, so we get the exact, authoritative file list + blob sizes without
pulling file contents and without touching the REST API quota.

Usage: python tools/verify_repos_git.py owner/repo [...]
Output: JSON to $VERIFY_OUT (default tools/verified_repos_git.json)
"""

import json
import os
import subprocess
import sys
import tempfile

WEIGHT_EXT = (".pth", ".pt", ".ckpt", ".h5", ".hdf5", ".keras", ".safetensors",
              ".onnx", ".model", ".pkl", ".bin", ".tar.gz", ".zip")
CODE_HINTS = ("train", "inference", "predict", "gradcam", "grad_cam", "xai",
              "explain", "saliency", "salience", "attention", "cam.py")
XAI_HINTS = ("gradcam", "grad_cam", "gradcamplusplus", "grad-cam", "shap", "lime",
             "integrated_gradients", "captum", "scorecam", "layercam",
             "eigencam", "attention_map", "xai", "explainab", "interpret")
DATA_HINTS = ("dataset", "data_loader", "dataloader", "dataloaders", "manifest")


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=180)
    return p.returncode, p.stdout, p.stderr


def inspect(repo):
    out = {"repo": repo}
    with tempfile.TemporaryDirectory() as tmp:
        rc, so, se = run(["git", "clone", "--filter=blob:none", "--no-checkout",
                           "--depth", "1", f"https://github.com/{repo}.git", "r"], tmp)
        if rc != 0:
            out["status"] = "clone_failed"
            out["error"] = (se or so).strip()[:300]
            return out
        rc, so, se = run(["git", "ls-tree", "-r", "-l", "HEAD"], os.path.join(tmp, "r"))
        if rc != 0:
            out["status"] = "ls-tree_failed"
            return out
        entries = []
        for line in so.splitlines():
            # <mode> <type> <object> <size>\t<path>
            if "\t" not in line:
                continue
            meta, path = line.split("\t", 1)
            parts = meta.split()
            size = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            entries.append((path, size))
        rc2, so2, _ = run(["git", "log", "-1", "--format=%H|%ad|%an", "--date=short"],
                          os.path.join(tmp, "r"))
        rc3, so3, _ = run(["git", "ls-tree", "-l", "HEAD"], os.path.join(tmp, "r"))

    out["status"] = "ok"
    out["last_commit"] = so2.strip()
    out["file_count"] = len(entries)
    paths = [p for p, _ in entries]

    weights = []
    for p, s in entries:
        lp = p.lower()
        if lp.endswith(WEIGHT_EXT) and s > 50_000:
            weights.append({"path": p, "bytes": s})
    out["weight_files"] = weights
    out["weight_count"] = len(weights)

    out["xai_files"] = [p for p in paths if any(h in p.lower() for h in XAI_HINTS)][:30]
    out["code_files"] = [p for p in paths if any(h in p.lower() for h in CODE_HINTS)][:30]
    out["data_files"] = [p for p in paths if any(h in p.lower() for h in DATA_HINTS)][:15]
    out["notebooks"] = [p for p in paths if p.lower().endswith(".ipynb")][:20]
    out["has_docker"] = any(p.lower().startswith("dockerfile") for p in paths)
    out["has_license"] = any(p.lower() in ("license", "license.txt", "license.md", "copying") for p in paths)
    out["has_requirements"] = any(p.lower().endswith(("requirements.txt", "environment.yml",
                                                     "pyproject.toml", "setup.py", "poetry.lock")) for p in paths)
    out["has_readme"] = any(p.lower().startswith("readme") for p in paths)
    out["total_bytes"] = sum(s for _, s in entries)
    out["root_files"] = sorted({p.split("/")[0] for p in paths if "/" not in p})
    return out


def main():
    repos = [r.strip() for r in sys.argv[1:] if r.strip()]
    results = []
    for i, r in enumerate(repos, 1):
        res = inspect(r)
        results.append(res)
        print(f"[{i}/{len(repos)}] {r} -> {res.get('status')} files={res.get('file_count')} "
              f"weights={res.get('weight_count')} xai={len(res.get('xai_files', []))} "
              f"nb={len(res.get('notebooks', []))}", file=sys.stderr)
    outfile = os.environ.get("VERIFY_OUT", "tools/verified_repos_git.json")
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(outfile)


if __name__ == "__main__":
    main()
