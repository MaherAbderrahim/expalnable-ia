"""Corrige les deux URL reellement cassees du dossier prostate.

1. L'URL TCIA de la collection « IRM + histologie macroscopique » etait
   erronee (prostate-mri-ablation n'existe pas). La collection correcte est
   PROSTATE-MRI (26 sujets, IRM + histologie de prostatectomie, CC BY 3.0),
   et PROSTATE-DIAGNOSIS (92 sujets, IRM + SEG).
2. L'URL thelancet.com du papier PI-CAI ne repond pas ; le DOI
   resolu. On le remplace partout.
"""

import glob
import os

REPLACEMENTS = [
    # 1. Collections TCIA
    ("https://www.cancerimagingarchive.net/collection/prostate-mri-ablation/",
     "https://www.cancerimagingarchive.net/collection/prostate-mri/"),
    # 2. Article PI-CAI : URL thelancet -> DOI (verifie 200)
    ("https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(24)00220-1/fulltext",
     "https://doi.org/10.1016/S1470-2045(24)00220-1"),
]


def main():
    files = (["RESEARCH_REPORT.md", "TOP_MODELS.md", "RECOMMENDATION.md",
              "index.html", "research-data.json", "assets/README.md"]
             + glob.glob("tools/chunks/*.json"))
    total = 0
    for path in files:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8-sig") as f:
            src = f.read()
        out = src
        for old, new in REPLACEMENTS:
            out = out.replace(old, new)
        if out != src:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(out)
            total += 1
            print("corrige :", os.path.basename(path))
    print(f"{total} fichier(s) corrige(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
