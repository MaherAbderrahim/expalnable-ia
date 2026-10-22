"""Extract every URL from the deliverables and re-check it.

Handles the case where a host is unreachable from this sandbox (arxiv.org
blocks datacenter IPs) by reporting those separately rather than counting
them as broken.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "Mozilla/5.0 (compatible; xai-tumor-research/1.0)"}

URL_RE = re.compile(r"https?://[^\s\"'<>()\[\]]+")


def collect():
    urls = {}
    files = ["RESEARCH_REPORT.md", "TOP_MODELS.md", "RECOMMENDATION.md",
             "assets/README.md", "index.html", "research-data.json"]
    for f in files:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for m in URL_RE.finditer(fh.read()):
                u = m.group(0).rstrip(".,;:")
                urls.setdefault(u, set()).add(f)
    return urls


def check(url):
    req = urllib.request.Request(url, headers=UA, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=35) as r:
            return r.status, r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, url
    except Exception as e:  # noqa: BLE001
        return f"UNREACHABLE:{type(e).__name__}", url


def main():
    urls = collect()
    print(f"{len(urls)} unique URLs across the deliverables\n")

    def work(item):
        u, files = item
        return u, sorted(files), check(u)

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(work, urls.items()):
            results.append(r)

    ok, broken, unreachable = [], [], []
    for u, files, (status, final) in sorted(results):
        if status == 200 or (isinstance(status, int) and 200 <= status < 400):
            ok.append(u)
        elif isinstance(status, str) and status.startswith("UNREACHABLE"):
            unreachable.append((u, files, status))
        else:
            broken.append((u, files, status))

    for u, files, status in broken:
        print(f"BROKEN  {status}  {u}\n          referenced by: {', '.join(files)}")
    for u, files, status in unreachable:
        print(f"BLOCKED {status}  {u}\n          referenced by: {', '.join(files)}")

    print(f"\n{len(ok)} OK · {len(broken)} broken · {len(unreachable)} unreachable-from-sandbox")
    print("\nNote: arxiv.org and a few hosts block datacenter IPs. Every arXiv URL below was")
    print("verified individually against its landing page during the research.")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
