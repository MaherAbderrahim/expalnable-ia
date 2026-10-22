"""Probe GitHub raw URLs to confirm/deny the presence of specific files.

Usage: python tools/probe_raw.py <repo> <branch> <path> [<path> ...]
Exit 0 if at least one path exists.
Prints one line per path: STATUS<TAB>BYTES<TAB>PATH
"""

import sys
import urllib.error
import urllib.request

UA = {"User-Agent": "xai-tumor-research"}


def head(url):
    req = urllib.request.Request(url, headers=UA, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            # Read a little so the connection closes cleanly; report content-length.
            n = 0
            while n < 4096:
                chunk = r.read(4096)
                if not chunk:
                    break
                n += len(chunk)
            return r.status, r.headers.get("Content-Length", "?")
    except urllib.error.HTTPError as e:
        return e.code, 0
    except Exception as e:  # noqa: BLE001
        return f"ERR:{type(e).__name__}", 0


def main():
    repo, branch, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    found = 0
    for p in paths:
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{p}"
        status, size = head(url)
        ok = status == 200
        found += 1 if ok else 0
        print(f"{status}\t{size}\t{p}")
    sys.exit(0 if found else 1)


if __name__ == "__main__":
    main()
