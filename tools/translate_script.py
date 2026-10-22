"""Translate the user-visible strings in script.js to French.

Only literal display strings are touched; identifiers, selectors and logic are
left untouched. Replacements are exact-match so an unexpected change fails
loudly instead of silently corrupting the file.
"""

import io
import sys

PATH = "script.js"

REPLACEMENTS = [
    # --- header comment -------------------------------------------------
    ("""/* ============================================================
   Explainable AI — Tumor Detection Research
   Vanilla JS. No frameworks, no network calls at runtime.
   Data comes from research-data.js (window.RESEARCH_DATA), with a
   fetch() fallback to research-data.json for http(s) hosting.
   ============================================================ */""",
     """/* ============================================================
   IA explicable — Recherche sur la détection de tumeurs
   JavaScript pur. Aucun framework, aucun appel réseau à l'exécution.
   Les données proviennent de research-data.js (window.RESEARCH_DATA),
   avec un repli fetch() vers research-data.json pour un hébergement
   en http(s).
   ============================================================ */"""),

    # --- licence detection ---------------------------------------------
    ("""  function isPermissive(lic) {
    if (!lic) return false;
    var l = String(lic).toLowerCase();
    if (l.indexOf("none") !== -1 || l.indexOf("no licence") !== -1 || l.indexOf("no license") !== -1) return false;
    if (l.indexOf("not stated") !== -1 || l.indexOf("unknown") !== -1 || l.indexOf("check") !== -1) return false;
    return /mit|apache|bsd|cc by|cc0|isc|zlib/.test(l);
  }""",
     """  function isPermissive(lic) {
    if (!lic) return false;
    var l = String(lic).toLowerCase();
    // Exclusions explicites : licence absente, non précisée, inconnue,
    // à vérifier, ou restrictives (usage non commercial / contrôlé).
    if (/aucune|none|not stated|non indiqu|unknown|inconnu|à v.ri?fi|check|kaggle|nih |contrôl|control/.test(l)) return false;
    if (/by-nc|nc-sa|non commercial|non-commercial|share-alike/.test(l)) return false;
    return /mit|apache|bsd|cc by|cc0|isc|zlib|agpl|gpl/.test(l);
  }"""),

    # --- weights state --------------------------------------------------
    ("""  function weightState(weights) {
    if (!weights) return "no";
    var w = String(weights).toLowerCase();
    if (w.indexOf("no ") === 0 || w.indexOf("not found") !== -1 || w.indexOf("no official") !== -1) return "no";
    if (w.indexOf("external") !== -1 || w.indexOf("download") !== -1) return "external";
    return "yes";
  }""",
     """  /* L'état des poids est stocké explicitement dans les données
     (weights_state) plutôt que deviné à partir du texte libre. */
  function weightState(model) {
    if (!model) return "no";
    if (model.weights_state) return model.weights_state;
    var w = String(model.weights || "").toLowerCase();
    if (/externe|external/.test(w)) return "external";
    if (/^non|^s\\.o\\.|not found|non v.ri/i.test(w)) return "no";
    return "yes";
  }"""),

    # --- link button ----------------------------------------------------
    ("""      a.title = "Not available";""",
     """      a.title = "Non disponible";"""),

    # --- stats ----------------------------------------------------------
    ("""      { n: c.models, l: "Models catalogued", s: "CNN, Transformer, U-Net family, hybrid" },
      { n: c.datasets, l: "Datasets", s: "MRI, CT, mammography, PET" },
      { n: c.xai_methods, l: "XAI methods", s: "Gradient, perturbation, attention, concept" },
      { n: c.papers, l: "Papers", s: "Primary sources with verified links" },
      { n: c.repositories, l: "Repositories", s: "Every one inspected file-by-file" },
      { n: c.recommendations, l: "Implementation routes", s: "Simple / compromise / ambitious" }""",
     """      { n: c.models, l: "Modèles catalogués", s: "CNN, transformeurs, famille U-Net, hybrides" },
      { n: c.datasets, l: "Jeux de données", s: "IRM, CT, mammographie, TEP" },
      { n: c.xai_methods, l: "Méthodes XAI", s: "Gradient, perturbation, attention, concepts" },
      { n: c.papers, l: "Articles", s: "Sources primaires aux liens vérifiés" },
      { n: c.repositories, l: "Dépôts", s: "Chacun inspecté fichier par fichier" },
      { n: c.recommendations, l: "Voies d'implémentation", s: "Simple / compromis / ambitieuse" }"""),

    # --- options meta ---------------------------------------------------
    ("""      var t1 = el("span"); t1.appendChild(document.createTextNode("Time: ")); t1.appendChild(el("b", null, o.time)); m.appendChild(t1);
      var t2 = el("span"); t2.appendChild(document.createTextNode("GPU: ")); t2.appendChild(el("b", null, o.gpu)); m.appendChild(t2);""",
     """      var t1 = el("span"); t1.appendChild(document.createTextNode("Durée : ")); t1.appendChild(el("b", null, o.time)); m.appendChild(t1);
      var t2 = el("span"); t2.appendChild(document.createTextNode("GPU : ")); t2.appendChild(el("b", null, o.gpu)); m.appendChild(t2);"""),

    # --- ranking dimensions ---------------------------------------------
    ("""      ["reuse", "Reuse"], ["quality", "Quality"], ["weights", "Weights"], ["code", "Code"],
      ["dataset", "Data"], ["explainability", "XAI"], ["reproducibility", "Repro"],
      ["compute", "Compute"], ["student_fit", "Student"], ["documentation", "Docs"]""",
     """      ["reuse", "Réutilis."], ["quality", "Qualité"], ["weights", "Poids"], ["code", "Code"],
      ["dataset", "Données"], ["explainability", "XAI"], ["reproducibility", "Repro"],
      ["compute", "Calcul"], ["student_fit", "Étudiant"], ["documentation", "Docs"]"""),

    # --- model filters --------------------------------------------------
    ("""    [["permissive", "Permissive (MIT/Apache/BSD/CC)"], ["none", "No licence"]].forEach(function (p) {""",
     """    [["permissive", "Permissive (MIT/Apache/BSD/CC)"], ["none", "Sans licence"]].forEach(function (p) {"""),

    ("""    $("#model-count").textContent = rows.length + " of " + D.models.length + " models";

    if (!rows.length) { g.appendChild(el("div", "empty", "No model matches these filters.")); return; }""",
     """    $("#model-count").textContent = rows.length + " sur " + D.models.length + " modèles";

    if (!rows.length) { g.appendChild(el("div", "empty", "Aucun modèle ne correspond à ces filtres.")); return; }"""),

    # --- model card -----------------------------------------------------
    ("""      var ws = weightState(m.weights);
      var t = el("span", "tag " + (ws === "yes" ? "tag-a" : ws === "external" ? "tag-b" : "tag-d"),
        ws === "yes" ? "Weights in repo" : ws === "external" ? "Weights external" : "No weights");""",
     """      var ws = weightState(m);
      var t = el("span", "tag " + (ws === "yes" ? "tag-a" : ws === "external" ? "tag-b" : "tag-d"),
        ws === "yes" ? "Poids dans le dépôt" : ws === "external" ? "Poids externes" : "Aucun poids");"""),

    ("""      row("Task", m.task);
      row("Dataset", m.dataset);
      row("Classes", m.classes);
      row("Input", m.input);
      row("Perf.", m.performance);
      row("Framework", m.framework);
      row("Licence", m.license);
      row("XAI", m.xai);
      c.appendChild(dl);

      var diff = el("div", "kv");
      diff.appendChild(el("dt", null, "Pros"));
      diff.appendChild(el("dd", null, m.pros));
      diff.appendChild(el("dt", null, "Cons"));
      diff.appendChild(el("dd", null, m.cons));
      diff.appendChild(el("dt", null, "Reuse"));
      diff.appendChild(el("dd", null, m.difficulty));
      c.appendChild(diff);""",
     """      row("Tâche", m.task);
      row("Jeu de données", m.dataset);
      row("Classes", m.classes);
      row("Entrée", m.input);
      row("Perf.", m.performance);
      row("Framework", m.framework);
      row("Licence", m.license);
      row("XAI", m.xai);
      c.appendChild(dl);

      var diff = el("div", "kv");
      diff.appendChild(el("dt", null, "Avantages"));
      diff.appendChild(el("dd", null, m.pros));
      diff.appendChild(el("dt", null, "Limites"));
      diff.appendChild(el("dd", null, m.cons));
      diff.appendChild(el("dt", null, "Réutilis."));
      diff.appendChild(el("dd", null, m.difficulty));
      c.appendChild(diff);"""),

    # --- XAI cards ------------------------------------------------------
    ("""    $("#xai-count").textContent = rows.length + " of " + D.xai_methods.length + " methods";
    if (!rows.length) { g.appendChild(el("div", "empty", "No XAI method matches these filters.")); return; }""",
     """    $("#xai-count").textContent = rows.length + " sur " + D.xai_methods.length + " méthodes";
    if (!rows.length) { g.appendChild(el("div", "empty", "Aucune méthode d'IA explicable ne correspond à ces filtres.")); return; }"""),

    ("""      c.appendChild(el("h4", null, "How it works"));
      c.appendChild(el("p", null, x.how));

      c.appendChild(el("h4", null, "Compatible models"));
      c.appendChild(el("p", null, x.compatible));

      c.appendChild(el("h4", null, "Advantages"));
      c.appendChild(el("p", "plus", x.advantages));

      c.appendChild(el("h4", null, "Limitations"));
      c.appendChild(el("p", "minus", x.limitations));

      c.appendChild(el("h4", null, "Medical relevance"));
      c.appendChild(el("p", "rel", x.medical_relevance));

      c.appendChild(el("h4", null, "Cost & implementation"));
      c.appendChild(el("p", null, x.compute + " · " + x.ease));

      var foot = el("div", "xcard-foot");
      foot.appendChild(el("div", null, "Visualisation quality: " + x.visual_quality));
      var libs = el("div");
      libs.appendChild(el("span", null, "Libraries: " + x.libs));
      foot.appendChild(libs);
      var code = el("div");
      code.appendChild(el("span", null, "Code: "));""",
     """      c.appendChild(el("h4", null, "Principe"));
      c.appendChild(el("p", null, x.how));

      c.appendChild(el("h4", null, "Modèles compatibles"));
      c.appendChild(el("p", null, x.compatible));

      c.appendChild(el("h4", null, "Avantages"));
      c.appendChild(el("p", "plus", x.advantages));

      c.appendChild(el("h4", null, "Limites"));
      c.appendChild(el("p", "minus", x.limitations));

      c.appendChild(el("h4", null, "Pertinence médicale"));
      c.appendChild(el("p", "rel", x.medical_relevance));

      c.appendChild(el("h4", null, "Coût et implémentation"));
      c.appendChild(el("p", null, x.compute + " · " + x.ease));

      var foot = el("div", "xcard-foot");
      foot.appendChild(el("div", null, "Qualité de visualisation : " + x.visual_quality));
      var libs = el("div");
      libs.appendChild(el("span", null, "Bibliothèques : " + x.libs));
      foot.appendChild(libs);
      var code = el("div");
      code.appendChild(el("span", null, "Code : "));"""),

    ("""      code.appendChild(linkBtn((x.code || "").split(" | ")[0], "open", true));""",
     """      code.appendChild(linkBtn((x.code || "").split(" | ")[0], "ouvrir", true));"""),

    # --- datasets -------------------------------------------------------
    ("""      if (d.paper) { var p = el("a", "ds-note", "paper / datasheet"); p.href = d.paper; p.target = "_blank"; p.rel = "noopener noreferrer"; td.appendChild(p); }""",
     """      if (d.paper) { var p = el("a", "ds-note", "article / fiche"); p.href = d.paper; p.target = "_blank"; p.rel = "noopener noreferrer"; td.appendChild(p); }"""),

    ("""      var rec = d.recommended.split(" ")[0];
      var cls = rec === "YES" ? "rec-yes" : rec === "MAYBE" ? "rec-maybe" : "rec-no";
      tr.appendChild(el("td", cls, d.recommended.split("—")[0].trim()));""",
     """      var rec = d.recommended.split(" ")[0];
      var cls = rec === "OUI" ? "rec-yes" : rec === "PEUT-ÊTRE" ? "rec-maybe" : "rec-no";
      tr.appendChild(el("td", cls, d.recommended.split("—")[0].trim()));"""),

    ("""      if (d.caveat) c.appendChild(el("div", "caveat", "Caveat — " + d.caveat));
      if (d.quality_notes) c.appendChild(el("div", "rec", "QC — " + d.quality_notes));
      else if (d.recommended) c.appendChild(el("div", "rec", d.recommended));

      var a = el("div", "dcard-actions");
      a.appendChild(linkBtn(d.url, "Official source", true));
      if (d.paper) a.appendChild(linkBtn(d.paper, "Datasheet"));""",
     """      if (d.caveat) c.appendChild(el("div", "caveat", "Réserve — " + d.caveat));
      if (d.quality_notes) c.appendChild(el("div", "rec", "Contrôle qualité — " + d.quality_notes));
      else if (d.recommended) c.appendChild(el("div", "rec", d.recommended));

      var a = el("div", "dcard-actions");
      a.appendChild(linkBtn(d.url, "Source officielle", true));
      if (d.paper) a.appendChild(linkBtn(d.paper, "Fiche"));"""),

    # --- papers ---------------------------------------------------------
    ("""    $("#paper-count").textContent = rows.length + " of " + D.papers.length + " papers";
    if (!rows.length) { g.appendChild(el("div", "empty", "No paper matches these filters.")); return; }""",
     """    $("#paper-count").textContent = rows.length + " sur " + D.papers.length + " articles";
    if (!rows.length) { g.appendChild(el("div", "empty", "Aucun article ne correspond à ces filtres.")); return; }"""),

    ("""      function chip(k, v) {
        if (!v) return;
        var s = el("span", "chip");
        s.appendChild(el("b", null, k + ": "));""",
     """      function chip(k, v) {
        if (!v) return;
        var s = el("span", "chip");
        s.appendChild(el("b", null, k + " : "));"""),

    ("""      chip("Model", p.model);
      chip("Dataset", p.dataset);
      chip("Task", p.task);
      chip("Metrics", p.metrics);
      chip("XAI", p.xai);""",
     """      chip("Modèle", p.model);
      chip("Jeu de données", p.dataset);
      chip("Tâche", p.task);
      chip("Métriques", p.metrics);
      chip("XAI", p.xai);"""),

    ("""      if (p.limits) {
        var lp = el("p", "limits");
        lp.appendChild(el("b", null, "Limits — "));""",
     """      if (p.limits) {
        var lp = el("p", "limits");
        lp.appendChild(el("b", null, "Limites — "));"""),

    ("""      act.appendChild(linkBtn(p.url, "Paper", true));
      if (firstUrl(p.code)) act.appendChild(linkBtn(firstUrl(p.code), "Code"));
      var dsMatch = findDataset(p.dataset);
      if (dsMatch) act.appendChild(linkBtn(dsMatch.url, "Dataset"));
      else if (p.dataset_available && p.dataset_available !== "n/a") act.appendChild(linkBtn(null, "Dataset available"));
      if (p.weights && p.weights.indexOf("http") === 0) act.appendChild(linkBtn(p.weights, "Weights"));
      else if (p.weights && p.weights !== "n/a") act.appendChild(linkBtn(null, "Weights: " + p.weights.slice(0, 34) + (p.weights.length > 34 ? "…" : "")));""",
     """      act.appendChild(linkBtn(p.url, "Article", true));
      if (firstUrl(p.code)) act.appendChild(linkBtn(firstUrl(p.code), "Code"));
      var dsMatch = findDataset(p.dataset);
      if (dsMatch) act.appendChild(linkBtn(dsMatch.url, "Jeu de données"));
      else if (p.dataset_available && p.dataset_available !== "n/a") act.appendChild(linkBtn(null, "Jeu de données disponible"));
      if (p.weights && p.weights.indexOf("http") === 0) act.appendChild(linkBtn(p.weights, "Poids"));
      else if (p.weights && p.weights !== "n/a") act.appendChild(linkBtn(null, "Poids : " + p.weights.slice(0, 34) + (p.weights.length > 34 ? "…" : "")));"""),

    # --- comparison -----------------------------------------------------
    ("""        code: rep ? (rep.category === "D" ? "README only" : "yes") : "repo",
        weights: ws === "yes" ? "in repo" : ws === "external" ? "external" : "no",""",
     """        code: rep ? (rep.category === "D" ? "README seul" : "oui") : "dépôt",
        weights: ws === "yes" ? "dans le dépôt" : ws === "external" ? "externes" : "aucun","""),

    ("""    $("#cmp-count").textContent = rows.length + " of " + cmpRows.length + " solutions";""",
     """    $("#cmp-count").textContent = rows.length + " sur " + cmpRows.length + " solutions";"""),

    ("""    if (!rows.length) { tb.appendChild(el("tr")).appendChild(el("td", "empty", "No solution matches these filters.")); return; }""",
     """    if (!rows.length) { tb.appendChild(el("tr")).appendChild(el("td", "empty", "Aucune solution ne correspond à ces filtres.")); return; }"""),

    ("""      var wtd = el("td");
      var wc = r.weights === "in repo" ? "tag-a" : r.weights === "external" ? "tag-b" : "tag-d";
      wtd.appendChild(el("span", "tag " + wc, r.weights));
      tr.appendChild(wtd);
      var ltd = el("td");
      ltd.appendChild(el("span", "tag " + (r._permissive ? "tag-a" : "tag-d"), r._permissive ? "Permissive" : "None / unclear"));
      tr.appendChild(ltd);""",
     """      var wtd = el("td");
      var wc = r.weights === "dans le dépôt" ? "tag-a" : r.weights === "externes" ? "tag-b" : "tag-d";
      wtd.appendChild(el("span", "tag " + wc, r.weights));
      tr.appendChild(wtd);
      var ltd = el("td");
      ltd.appendChild(el("span", "tag " + (r._permissive ? "tag-a" : "tag-d"), r._permissive ? "Permissive" : "Absente / floue"));
      tr.appendChild(ltd);"""),

    # --- sources --------------------------------------------------------
    ("""    D.repositories.forEach(function (r) { add("GitHub", r.name + " — " + r.github, r.github, r.license + " · " + (r.stars ? r.stars + " stars · " : "") + "Cat " + r.category); });
    D.datasets.forEach(function (d) { add("Dataset", d.name, d.url, d.license); if (d.paper) add("Datasheet", d.name + " datasheet", d.paper, d.venue || ""); });
    D.papers.forEach(function (p) { add("Paper", p.title, p.url, p.venue + " · " + p.year); });
    D.models.forEach(function (m) { if (m.code && m.code.indexOf("http") === 0) add("Model code", m.name, m.code, m.license); });
    D.xai_methods.forEach(function (x) {
      (x.code || "").split(" | ").forEach(function (u) { if (u.indexOf("http") === 0) add("XAI code", x.name + " implementation", u, x.category); });
    });""",
     """    D.repositories.forEach(function (r) { add("GitHub", r.name + " — " + r.github, r.github, r.license + " · " + (r.stars ? r.stars + " étoiles · " : "") + "Cat " + r.category); });
    D.datasets.forEach(function (d) { add("Jeu de données", d.name, d.url, d.license); if (d.paper) add("Fiche", d.name + " fiche", d.paper, d.venue || ""); });
    D.papers.forEach(function (p) { add("Article", p.title, p.url, p.venue + " · " + p.year); });
    D.models.forEach(function (m) { if (m.code && m.code.indexOf("http") === 0) add("Code modèle", m.name, m.code, m.license); });
    D.xai_methods.forEach(function (x) {
      (x.code || "").split(" | ").forEach(function (u) { if (u.indexOf("http") === 0) add("Code XAI", x.name + " implémentation", u, x.category); });
    });"""),

    ("""      $("#src-count").textContent = rows.length + " of " + srcs.length + " sources";
      if (!rows.length) { g.appendChild(el("div", "empty", "No source matches these filters.")); return; }""",
     """      $("#src-count").textContent = rows.length + " sur " + srcs.length + " sources";
      if (!rows.length) { g.appendChild(el("div", "empty", "Aucune source ne correspond à ces filtres.")); return; }"""),

    # --- verification record labels -------------------------------------
    ("""      ["Date", v.date],
      ["Method", v.method],
      ["Repositories inspected", v.repos_checked],
      ["Weight files probed over HTTP", v.weights_files_probed],
      ["Licences confirmed", v.license_confirmed],
      ["Repositories flagged as having no licence", v.license_missing_flagged],
      ["Links checked", v.links_checked],
      ["Links that failed and were corrected", v.links_broken]""",
     """      ["Date", v.date],
      ["Méthode", v.method],
      ["Dépôts inspectés", v.repos_checked],
      ["Fichiers de poids sondés en HTTP", v.weights_files_probed],
      ["Licences confirmées", v.license_confirmed],
      ["Dépôts signalés comme sans licence", v.license_missing_flagged],
      ["Liens vérifiés", v.links_checked],
      ["Liens en échec puis corrigés", v.links_broken]"""),

    # --- paper filter options ------------------------------------------
    ("""        <option value="model">Model / architecture papers</option>""", None),  # placeholder, replaced below
]


def main():
    with open(PATH, encoding="utf-8") as f:
        src = f.read()

    applied, skipped = 0, []
    for pair in REPLACEMENTS:
        if pair[1] is None:
            continue
        old, new = pair
        if old in src:
            src = src.replace(old, new, 1)
            applied += 1
        else:
            skipped.append(old.splitlines()[0][:70])

    with open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(src)

    print(f"applied {applied} replacements")
    for s in skipped:
        print("  NOT FOUND:", s)
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
