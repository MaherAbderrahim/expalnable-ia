"""Applique deux corrections a script.js pour la version prostate.

1. Le lien « Code » des cartes XAI doit passer par firstUrl(), car certains
   champs `code` contiennent une URL suivie de texte (ex. « Article : URL -
   code non localise »).
2. Le type de source des depots doit etre lu depuis le champ `type`, afin
   que les entrees sans depot GitHub soient etiquetees « Article » plutot
   que « GitHub ».
"""

PATH = "script.js"

OLD1 = '''      var code = el("div");
      code.appendChild(el("span", null, "Code : "));
      code.appendChild(linkBtn((x.code || "").split(" | ")[0], "ouvrir", true));'''

NEW1 = '''      var code = el("div");
      code.appendChild(el("span", null, "Code : "));
      code.appendChild(linkBtn(firstUrl((x.code || "").split(" | ")[0]), "ouvrir", true));'''

OLD2 = '''    D.repositories.forEach(function (r) { add("GitHub", r.name + " — " + r.github, r.github, r.license + " · " + (r.stars ? r.stars + " étoiles · " : "") + "Cat " + r.category); });'''

NEW2 = '''    D.repositories.forEach(function (r) { add(r.type || "GitHub", r.name + " — " + r.github, r.github, r.license + " · " + (r.stars ? r.stars + " étoiles · " : "") + "Cat " + r.category); });'''


def main():
    with open(PATH, encoding="utf-8") as f:
        src = f.read()

    applied, skipped = 0, []
    for old, new in ((OLD1, NEW1), (OLD2, NEW2)):
        if old in src:
            src = src.replace(old, new, 1)
            applied += 1
        else:
            skipped.append(old.splitlines()[0][:70])

    with open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(src)

    print(f"{applied} remplacement(s) applique(s)")
    for s in skipped:
        print("  INTROUVABLE :", s)
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
