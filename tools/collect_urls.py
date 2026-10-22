"""Collecte toutes les URL des livrables pour une verification finale.

Gere les cas suivants :
  - lien markdown `[texte](url)`   -> le `)` final ne fait pas partie de l'URL
  - DOI avec parentheses, ex. `10.1016/S1470-2045(24)00220-1` -> parentheses
    legitimes a conserver
  - accents graves markdown `` `url` `` -> a retirer
"""

import os
import re

FILES = ["RESEARCH_REPORT.md", "TOP_MODELS.md", "RECOMMENDATION.md",
         "index.html", "research-data.json", "assets/README.md"]
PATTERN = re.compile(r"https?://[^\s\"'`<>\[\]]+")


def tidy(u):
    u = u.rstrip(".,;:`")
    # Retirer un `)` final non equilibre (syntaxe markdown `](url)`),
    # mais conserver les parentheses equilibrees des DOI.
    while u.endswith(")") and u.count(")") > u.count("("):
        u = u[:-1]
    return u


def main():
    urls = set()
    for f in FILES:
        if not os.path.exists(f):
            continue
        with open(f, encoding="utf-8-sig") as fh:
            for m in PATTERN.finditer(fh.read()):
                u = tidy(m.group(0))
                if not u or "{" in u or "}" in u:
                    continue
                urls.add(u)
    out = os.path.join("tools", "final_urls.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(urls)))
    print("URL collectees :", len(urls))


if __name__ == "__main__":
    main()
