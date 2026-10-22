"""Ajoute les entrees de depot manquantes (travaux sans depot) et le champ
`type` utilise par le site pour libeller correctement le lien.

Les entrees sans depot GitHub pointent vers l'article ou la page projet
verifiee : c'est le lien primaire reel, et le site l'affiche sous le bon
type plutot que sous un libelle "GitHub" trompeur.
"""

import collections
import json
import os

PATH = os.path.join("tools", "chunks", "05-repos.json")

# repo_id -> (nom, type, url, licence, note)
PAPER_ENTRIES = {
    "structmil": (
        "StructMIL (npj Digital Medicine 2026) - pas de dépôt de code localisé",
        "Article",
        "https://www.nature.com/articles/s41746-025-02289-4",
        "CC BY 4.0 (article open access)",
        "MIL à prototypes pour la généralisation inter-centre. Aucun dépôt de code n'a été localisé dans cette recherche : à lire comme une référence scientifique, pas comme un artefact réutilisable.",
    ),
    "hammm": (
        "Modèle XAI à signes PI-RADS (Hamm et al., Radiology 2023) - pas de dépôt de code localisé",
        "Article",
        "https://pubs.rsna.org/doi/10.1148/radiol.222276",
        "Radiology - à vérifier pour la redistribution",
        "LA référence clinique de l'IA explicable prostatique : explication textuelle en signes PI-RADS, validée par des experts (80 %) et par une étude de lecteurs. Ni code ni poids identifiés.",
    ),
    "pm10": (
        "Dictionnaire radiomique PM1.0 - pas de dépôt de code localisé",
        "Article",
        "https://pubmed.ncbi.nlm.nih.gov/40608191/",
        "Licence exclusive IJIM - redistribution non autorisée",
        "Relie chaque caractéristique radiomique à un signe PI-RADS. Attention : la licence de l'article est exclusive, ce qui interdit de redistribuer le dictionnaire sans accord.",
    ),
    "prostatattention": (
        "ProstAttention-Net (Med Image Anal 2022) - pas de dépôt de code localisé",
        "Article",
        "https://pubmed.ncbi.nlm.nih.gov/35085952/",
        "Elsevier - à vérifier",
        "Segmentation multi-étiquette par grade d'agressivité : l'attention par grade EST l'explication. Concept excellent, artefacts absents.",
    ),
    "ahnet": (
        "U-Net / AH-Net (Abdom Radiol 2022) - pas de dépôt de code localisé",
        "Article",
        "https://pubmed.ncbi.nlm.nih.gov/35099572/",
        "Springer - à vérifier",
        "Protocole de données exemplaire (vérité par biopsie ciblée, deux institutions) mais 35-48 % de PPV seulement : le rappel Reality de la détection prostatique.",
    ),
    "promisecorr": (
        "Corrélation IRM ↔ histologie macroscopique (PCA-Rad) - pas de dépôt de code localisé",
        "Article",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8976981/",
        "PMC - accès libre",
        "Fournit la vérité histologique par lésion, c'est-à-dire la VÉRITÉ TERRAIN contre laquelle une carte de chaleur prostatique peut être mesurée. C'est la source de la distinction conceptuelle « carte de chaleur ≠ masque ».",
    ),
    "gradcam-prostate-mri": (
        "Études Grad-CAM sur IRM prostatique (4 travaux) - pas de dépôt de code localisé",
        "Article",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC6920905/",
        "PMC - accès libre",
        "La ligne de base de l'IA explicable prostatique : Grad-CAM, et pour l'étude de 2019 la PREMIÈRE vérification quantitative du recouvrement avec les cibles PI-RADS des radiologues ET les localisations de l'anatomopathologiste.",
    ),
}


def main():
    with open(PATH, encoding="utf-8-sig") as f:
        repos = json.load(f, object_pairs_hook=collections.OrderedDict)

    existing = {r["id"] for r in repos}
    added = 0
    for rid, (name, typ, url, lic, note) in PAPER_ENTRIES.items():
        if rid in existing:
            continue
        entry = collections.OrderedDict()
        entry["id"] = rid
        entry["github"] = url
        entry["name"] = name
        entry["category"] = "D"
        entry["type"] = typ
        entry["stars"] = None
        entry["license"] = lic
        entry["framework"] = "n/a"
        entry["model"] = name.split(" - ")[0]
        entry["dataset"] = "Prostate (IRM ou histopathologie)"
        entry["xai"] = "Voir l'article"
        entry["weights"] = "NON"
        entry["verified_contents"] = note
        entry["difficulty"] = "Difficile"
        entry["note"] = note
        repos.append(entry)
        added += 1

    # Ajouter le champ `type` aux entrées existantes qui sont des dépôts GitHub.
    for r in repos:
        r.setdefault("type", "GitHub")

    with open(PATH, "w", encoding="utf-8", newline="") as f:
        json.dump(repos, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"{added} entrée(s) ajoutée(s), {len(repos)} dépôts au total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
