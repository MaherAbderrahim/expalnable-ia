"""Met a jour script.js pour la version prostate du dossier.

1. Libelles des statistiques (IRM/histopathologie).
2. Ajout du bloc « Perspective prostate » dans les cartes XAI.
3. Libelles de la colonne de comparaison adaptes.
"""

import re

PATH = "script.js"

REPLACEMENTS = [
    # --- statistiques ----------------------------------------------------
    ("""      { n: c.models, l: "Modèles catalogués", s: "CNN, transformeurs, famille U-Net, hybrides" },
      { n: c.datasets, l: "Jeux de données", s: "IRM, CT, mammographie, TEP" },
      { n: c.xai_methods, l: "Méthodes XAI", s: "Gradient, perturbation, attention, concepts" },
      { n: c.papers, l: "Articles", s: "Sources primaires aux liens vérifiés" },
      { n: c.repositories, l: "Dépôts", s: "Chacun inspecté fichier par fichier" },
      { n: c.recommendations, l: "Voies d'implémentation", s: "Simple / compromis / ambitieuse" }""",
     """      { n: c.models, l: "Modèles catalogués", s: "U-Net, nnU-Net, MIL, prototypes, attention" },
      { n: c.datasets, l: "Jeux de données", s: "IRM prostatique, histopathologie H&E" },
      { n: c.xai_methods, l: "Méthodes XAI", s: "Gradient, perturbation, concepts, prototypes" },
      { n: c.papers, l: "Articles", s: "Sources primaires aux liens vérifiés" },
      { n: c.repositories, l: "Dépôts", s: "Chacun inspecté fichier par fichier" },
      { n: c.recommendations, l: "Voies d'implémentation", s: "Simple / compromis / ambitieuse" }"""),

    # --- carte XAI : bloc prostate ---------------------------------------
    ("""      c.appendChild(el("h4", null, "Principe"));
      c.appendChild(el("p", null, x.how));""",
     """      c.appendChild(el("h4", null, "Principe"));
      c.appendChild(el("p", null, x.how));

      if (x.prospect_prostate) {
        c.appendChild(el("h4", null, "Persistance prostatique"));
        c.appendChild(el("p", "rel", x.prospect_prostate));
      }"""),

    # --- carte modèle : ligne XAI plus explicite --------------------------
    ("""      row("Licence", m.license);
      row("XAI", m.xai);
      c.appendChild(dl);""",
     """      row("Licence", m.license);
      row("XAI", m.xai);
      c.appendChild(dl);"""),

    # --- comparaison : libelles de poids --------------------------------
    ("""        code: rep ? (rep.category === "D" ? "README seul" : "oui") : "dépôt",
        weights: ws === "yes" ? "dans le dépôt" : ws === "external" ? "externes" : "aucun",""",
     """        code: rep ? (rep.category === "D" ? "vide / non vérifié" : rep.category === "C" ? "code partiel" : "oui") : "dépôt",
        weights: ws === "yes" ? "dans le dépôt" : ws === "external" ? "externes" : "aucun","""),

    ("""      var wc = r.weights === "dans le dépôt" ? "tag-a" : r.weights === "externes" ? "tag-b" : "tag-d";""",
     """      var wc = r.weights === "dans le dépôt" ? "tag-a" : r.weights === "externes" ? "tag-b" : "tag-d";"""),
]


def main():
    with open(PATH, encoding="utf-8") as f:
        src = f.read()

    applied, skipped = 0, []
    for old, new in REPLACEMENTS:
        if old in src:
            src = src.replace(old, new, 1)
            applied += 1
        elif old != new:
            skipped.append(old.splitlines()[0][:70])

    with open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(src)

    print(f"{applied} remplacement(s) appliqué(s)")
    for s in skipped:
        print("  INTROUVABLE :", s)
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
