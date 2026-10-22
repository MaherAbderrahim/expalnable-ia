"""Verify GitHub repositories for XAI/tumor-detection projects.

Checks (via GitHub REST API, no auth):
  - repo exists, is not archived, stars, license SPDX id
  - presence of real weight/checkpoint files (not just README claims)
  - presence of training / inference / XAI code
  - presence of notebooks, dataset links, docker, API/UI
  - license of the default branch

Outputs a JSON blob to stdout so the research report can cite verified facts.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.github.com"
UA = {"User-Agent": "xai-tumor-research", "Accept": "application/vnd.github+json"}

WEIGHT_EXT = (".pth", ".pt", ".ckpt", ".h5", ".keras", ".safetensors", ".bin", ".onnx", ".pkl.tar")
WEIGHT_DIR_HINTS = ("weights", "checkpoints", "checkpoint", "models/best", "pretrained", "ckpt")
CODE_HINTS = ("train", "inference", "predict", "gradcam", "grad_cam", "xai", "explain", "saliency")
XAI_HINTS = ("gradcam", "grad_cam", "gradcamplusplus", "grad-cam", "shap", "lime", "integrated_gradients",
             "captum", "scorecam", "layercam", "attention_map", "attentionmap", "xai", "explainability")
DATA_HINTS = ("dataset", "data_loader", "dataloader", "dataloaders")
DOC_HINTS = ("readme", "docs/", "documentation/", "license", "requirements.txt", "environment.yml")


def get(url):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"__status__": e.code}
    except Exception as e:  # noqa: BLE001
        return {"__error__": str(e)}


def list_tree(repo, branch, limit=4000):
    """Recursive git tree listing (truncated but enough for root-level detection)."""
    data = get(f"{API}/repos/{repo}/git/trees/{branch}?recursive=1")
    if "__status__" in data or "__error__" in data:
        return []
    return [t.get("path", "") for t in data.get("tree", [])][:limit]


def inspect(repo):
    meta = get(f"{API}/repos/{repo}")
    if meta.get("__status__"):
        return {"repo": repo, "status": f"HTTP {meta['__status__']}"}
    if meta.get("__error__"):
        return {"repo": repo, "status": "error", "detail": meta["__error__"]}

    branch = meta.get("default_branch", "main")
    lic = meta.get("license") or {}
    out = {
        "repo": repo,
        "url": meta.get("html_url"),
        "stars": meta.get("stargazers_count"),
        "forks": meta.get("forks_count"),
        "archived": meta.get("archived"),
        "default_branch": branch,
        "license_spdx": lic.get("spdx_id"),
        "license_name": lic.get("name"),
        "created": meta.get("created_at", "")[:10],
        "pushed": meta.get("pushed_at", "")[:10],
        "description": meta.get("description"),
        "topics": meta.get("topics", []),
    }
    tree = list_tree(repo, branch)
    out["file_count"] = len(tree)

    # weights: real files only
    weights = [p for p in tree if p.lower().endswith(WEIGHT_EXT)]
    # ignore tiny generic files under .git-ish dirs
    weights = [w for w in weights if not w.lower().endswith(("config.json", "vocab.json"))]
    out["weight_files"] = weights[:40]
    out["weight_count"] = len(weights)

    # weight-like directories
    out["weight_dirs"] = sorted({p.split("/")[0] + "/" + p.split("/")[1]
                                 for p in tree
                                 if len(p.split("/")) > 1
                                 and any(h in p.lower() for h in WEIGHT_DIR_HINTS)})[:20]

    low = [p.lower() for p in tree]
    out["has_notebooks"] = any(p.endswith(".ipynb") for p in low)
    out["notebook_count"] = sum(1 for p in low if p.endswith(".ipynb"))
    out["train_files"] = [p for p in tree if any(h in p.lower() for h in ("train", "fit", "optim"))][:20]
    out["infer_files"] = [p for p in tree if any(h in p.lower() for h in ("infer", "predict", "test", "demo", "app"))][:20]
    out["xai_files"] = [p for p in tree if any(h in p.lower() for h in XAI_HINTS)][:25]
    out["data_files"] = [p for p in tree if any(h in p.lower() for h in DATA_HINTS)][:15]
    out["has_docker"] = any("dockerfile" in p.lower() for p in low)
    out["has_api"] = any(p.lower().endswith(("app.py", "api.py", "main.py", "server.py", "streamlit_app.py")) for p in low)
    out["has_docs"] = any(p.startswith(("docs/", "documentation/")) for p in tree)
    out["readme_len"] = 0
    for p in tree:
        if p.lower() == "readme.md":
            rm = get(f"{API}/repos/{repo}/readme")
            out["readme_len"] = rm.get("size", 0)
            break
    return out


def main():
    repos = [r.strip() for r in sys.argv[1:] if r.strip()]
    results = []
    for i, r in enumerate(repos, 1):
        res = inspect(r)
        results.append(res)
        print(f"[{i}/{len(repos)}] {r} -> {res.get('status', 'ok')} "
              f"stars={res.get('stars')} lic={res.get('license_spdx')} "
              f"weights={res.get('weight_count')} ipynb={res.get('notebook_count')} "
              f"xai={len(res.get('xai_files', []))}", file=sys.stderr)
        time.sleep(0.35)
    outfile = os.environ.get("VERIFY_OUT", "verified_repos.json")
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(outfile)


if __name__ == "__main__":
    main()
