"""Nettoie les artefacts de caracteres parasites introduits en ecrivant les
fichiers de donnees (CJK, mots colles, ponctuation etranger).

Idempotent : peut etre relance autant de fois que necessaire.
"""

import glob
import json
import os
import re
import sys

# 1. Fragments CJK -> suppression (avec remplacement contextuel si besoin)
CJK_FIXES = [
    ("PAR CONCEPTS,，基于 le lexique", "PAR CONCEPTS, based on le lexique"),
    ("refléter le取了 patch", "refléter le patch"),
    ("de corrélation空间 entre IRM", "de corrélation spatiale entre IRM"),
    ("cohortes de凭借 biopsie", "cohortes avec biopsie"),
    ("les涵 1 500", "les 1 500"),
    ("structures histologiques意义 plus fiablement",
     "structures histologiques de façon plus fiable"),
    ("plus损失 functions", "plus les fonctions de perte"),
    ("de l'模型 élève", "de l modèle élève"),
]

# 2. Mots colles / ponctuation etrangere
WORD_FIXES = [
    ("lattention n'est pasCellule-un-graphe", "l'attention n'est pas une carte de chaleur"),
    ("le plus Error.", "le plus lisible."),
    ("trèsfastidious", "très fastidieux"),
    ("la.validité", "la validité"),
    ("lattention", "l'attention"),
    ("n'est pasCellule", "n'est pas"),
    ("  de l modèle", "  du modèle"),
    ("le modèle élève", "le modèle élève"),
    ("Vousursed l'axe", "Vous surfez l'axe"),
    ("attention n'est pasCellule-un-graphe", "attention n'est pas une carte"),
    ("seems", "semble"),
    ("，", ", "),
    ("；", "; "),
    ("（", " ("),
    ("）", ") "),
]

CJK = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+")


def clean_text(s):
    for a, b in CJK_FIXES:
        s = s.replace(a, b)
    for a, b in WORD_FIXES:
        s = s.replace(a, b)
    # Tout CJK résiduel : suppression pure.
    s = CJK.sub("", s)
    # Espaces doubles parasites.
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r" +([,;:.!?])", r"\1", s)
    s = re.sub(r"([,;:]) +", r"\1 ", s)
    return s


def walk(o):
    if isinstance(o, str):
        return clean_text(o)
    if isinstance(o, list):
        return [walk(v) for v in o]
    if isinstance(o, dict):
        return {k: walk(v) for k, v in o.items()}
    return o


def main():
    total = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "tools", "chunks", "*.json"))):
        with open(path, encoding="utf-8-sig") as f:
            raw = f.read()
        if not CJK.search(raw) and "lattention" not in raw:
            continue
        data = json.loads(raw)
        cleaned = walk(data)
        with open(path, "w", encoding="utf-8", newline="") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
            f.write("\n")
        total += 1
        print("cleaned", os.path.basename(path))

    print(f"{total} fichier(s) nettoyé(s)")
    return 0


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
