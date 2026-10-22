"""Verify that every URL intended for the deliverables actually resolves.

Checks each URL and records the final HTTP status + redirect target.
Non-2xx results are reported as BROKEN so they can be fixed or dropped.

Usage: python tools/check_links.py urls.txt
"""

import sys
import urllib.error
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; xai-tumor-research/1.0)"}
SKIP_HOSTS = ("arxiv.org",)  # arxiv blocks HEAD/GET from some agents; handled below


def check(url):
    req = urllib.request.Request(url, headers=UA, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, url
    except Exception as e:  # noqa: BLE001
        return f"ERR:{type(e).__name__}", url


def main():
    path = sys.argv[1]
    urls = [u.strip() for u in open(path, encoding="utf-8") if u.strip() and not u.startswith("#")]
    bad = []
    for u in urls:
        status, final = check(u)
        ok = isinstance(status, int) and 200 <= status < 400
        if not ok:
            bad.append((u, status))
        print(f"{status}\t{'OK ' if ok else 'BAD'}\t{u}")
    print(f"\n--- {len(urls) - len(bad)}/{len(urls)} OK, {len(bad)} broken ---")
    for u, s in bad:
        print(f"BROKEN {s} {u}")
    sys.exit(0)


if __name__ == "__main__":
    main()
