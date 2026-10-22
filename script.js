/* ============================================================
   IA explicable — Recherche sur la détection de tumeurs
   JavaScript pur. Aucun framework, aucun appel réseau à l'exécution.
   Les données proviennent de research-data.js (window.RESEARCH_DATA),
   avec un repli fetch() vers research-data.json pour un hébergement
   en http(s).
   ============================================================ */
(function () {
  "use strict";

  var D = null;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  function el(tag, cls, txt) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt !== undefined && txt !== null) n.textContent = String(txt);
    return n;
  }

  function esc(s) {
    return String(s === undefined || s === null ? "" : s);
  }

  function isPermissive(lic) {
    if (!lic) return false;
    var l = String(lic).toLowerCase();
    // Exclusions explicites : licence absente, non précisée, inconnue, à
    // vérifier, ou restrictive (usage non commercial / accès contrôlé).
    if (/aucune|none|not stated|non indiqu|unknown|inconnu|à v.ri?fi|check|kaggle|nih |contrôl|control/.test(l)) return false;
    if (/by-nc|nc-sa|non commercial|non-commercial|share-alike/.test(l)) return false;
    return /mit|apache|bsd|cc by|cc0|isc|zlib|odbl|gpl/.test(l);
  }

  /* L'état des poids est stocké explicitement dans les données
     (weights_state) plutôt que deviné à partir du texte libre. */
  function weightState(model) {
    if (!model) return "no";
    if (model.weights_state) return model.weights_state;
    var w = String(model.weights || "").toLowerCase();
    if (/externe|external/.test(w)) return "external";
    if (/^non|^s\.o\.|not found|non v.ri/i.test(w)) return "no";
    return "yes";
  }

  function linkBtn(href, label, primary) {
    var a = el("a", "btn" + (primary ? " btn-primary" : ""), label);
    if (href) {
      a.href = href;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    } else {
      a.className += " is-disabled";
      a.setAttribute("aria-disabled", "true");
      a.title = "Non disponible";
    }
    return a;
  }

  /* Some entries store a URL inside a sentence ("... available at https://...").
     Pull out the first absolute URL, or return null if there is none. */
  function firstUrl(s) {
    if (!s) return null;
    var str = String(s);
    if (/^https?:\/\/\S+$/.test(str.trim())) return str.trim();
    var m = str.match(/https?:\/\/[^\s"'<>)\]]+/);
    return m ? m[0] : null;
  }

  /* ---------------- Boot ---------------- */

  function boot(data) {
    D = data;
    renderStats();
    renderRecommended();
    renderOptions();
    renderTopModels();
    renderModels();
    renderXai();
    renderDatasets();
    renderPapers();
    renderComparison();
    renderArchitecture();
    renderSources();
  }

  function start() {
    if (window.RESEARCH_DATA) { boot(window.RESEARCH_DATA); return; }
    fetch("research-data.json")
      .then(function (r) { return r.json(); })
      .then(boot)
      .catch(function () {
        document.querySelector("main").insertAdjacentHTML("afterbegin",
          '<div class="wrap"><div class="callout callout-warn">Could not load <code>research-data.js</code> or <code>research-data.json</code>. ' +
          'Run <code>python tools/build_data.py</code> to regenerate them.</div></div>');
      });
  }

  /* ---------------- 1. Stats ---------------- */

  function renderStats() {
    var c = D.meta.counts;
    var items = [
      { n: c.models, l: "Modèles catalogués", s: "U-Net, nnU-Net, MIL, prototypes, attention" },
      { n: c.datasets, l: "Jeux de données", s: "IRM prostatique, histopathologie H&E" },
      { n: c.xai_methods, l: "Méthodes XAI", s: "Gradient, perturbation, concepts, prototypes" },
      { n: c.papers, l: "Articles", s: "Sources primaires aux liens vérifiés" },
      { n: c.repositories, l: "Dépôts", s: "Chacun inspecté fichier par fichier" },
      { n: c.recommendations, l: "Voies d'implémentation", s: "Simple / compromis / ambitieuse" }
    ];
    var grid = $("#stats");
    grid.innerHTML = "";
    items.forEach(function (it) {
      var d = el("div", "stat");
      d.appendChild(el("b", null, it.n));
      d.appendChild(el("span", null, it.l));
      d.appendChild(el("small", null, it.s));
      grid.appendChild(d);
    });
  }

  /* ---------------- 2. Recommended ---------------- */

  function renderRecommended() {
    var r = D.recommended;
    $("#rec-model").textContent = r.model;
    $("#rec-dataset").textContent = r.dataset;
    $("#rec-xai").textContent = r.xai_method;
    $("#rec-why").textContent = r.why;
    $("#rec-time").textContent = r.train_time;
    $("#rec-gpu").textContent = r.gpu;

    function fill(id, arr) {
      var u = $(id); u.innerHTML = "";
      arr.forEach(function (x) { u.appendChild(el("li", null, x)); });
    }
    fill("#rec-pros", r.pros);
    fill("#rec-cons", r.cons);
  }

  function renderOptions() {
    var g = $("#options"); g.innerHTML = "";
    D.recommendations.forEach(function (o) {
      var c = el("div", "opt-card");
      c.appendChild(el("h3", null, o.label));

      var rows = [
        ["Model", o.model], ["Dataset", o.dataset], ["XAI", o.xai]
      ];
      rows.forEach(function (r) {
        var w = el("div", "opt-row");
        w.appendChild(el("span", null, r[0]));
        w.appendChild(el("div", null, r[1]));
        c.appendChild(w);
      });

      c.appendChild(el("p", "opt-why", o.why));

      var ul = el("ul");
      (o.pros || []).slice(0, 3).forEach(function (x) { ul.appendChild(el("li", null, x)); });
      c.appendChild(ul);

      var m = el("div", "opt-meta");
      var t1 = el("span"); t1.appendChild(document.createTextNode("Durée : ")); t1.appendChild(el("b", null, o.time)); m.appendChild(t1);
      var t2 = el("span"); t2.appendChild(document.createTextNode("GPU : ")); t2.appendChild(el("b", null, o.gpu)); m.appendChild(t2);
      c.appendChild(m);
      g.appendChild(c);
    });
  }

  function renderTopModels() {
    var g = $("#top-models"); g.innerHTML = "";
    var dims = [
      ["reuse", "Réutilis."], ["quality", "Qualité"], ["weights", "Poids"], ["code", "Code"],
      ["dataset", "Données"], ["explainability", "XAI"], ["reproducibility", "Repro"],
      ["compute", "Calcul"], ["student_fit", "Étudiant"], ["documentation", "Docs"]
    ];
    D.top_models.forEach(function (t) {
      var row = el("div", "rank-row");
      row.appendChild(el("div", "rank-badge", t.rank));
      var body = el("div");
      body.appendChild(el("h4", null, t.name));
      body.appendChild(el("p", null, t.one_line));

      var bars = el("div", "score-bars");
      dims.forEach(function (d) {
        var v = t[d[0]] || 0;
        var b = el("div", "score-bar");
        b.appendChild(el("span", null, d[1]));
        var i = el("i"); i.style.setProperty("--w", (v / 10 * 100) + "%");
        b.appendChild(i);
        b.appendChild(el("b", null, v + "/10"));
        bars.appendChild(b);
      });
      body.appendChild(bars);
      row.appendChild(body);
      g.appendChild(row);
    });
  }

  /* ---------------- 3. Models ---------------- */

  var modelState = { q: "", task: "", weights: "", license: "" };

  function renderModels() {
    var ts = $("#model-task"), ls = $("#model-license");
    var tasks = {};
    D.models.forEach(function (m) { tasks[m.task] = 1; });
    Object.keys(tasks).sort().forEach(function (t) {
      var o = el("option", null, t); o.value = t; ts.appendChild(o);
    });
    [["permissive", "Permissive (MIT/Apache/BSD/CC)"], ["none", "Sans licence"]].forEach(function (p) {
      var o = el("option", null, p[1]); o.value = p[0]; ls.appendChild(o);
    });

    $("#model-search").addEventListener("input", function (e) { modelState.q = e.target.value.toLowerCase(); drawModels(); });
    ts.addEventListener("change", function (e) { modelState.task = e.target.value; drawModels(); });
    $("#model-weights").addEventListener("change", function (e) { modelState.weights = e.target.value; drawModels(); });
    ls.addEventListener("change", function (e) { modelState.license = e.target.value; drawModels(); });
    drawModels();
  }

  function drawModels() {
    var g = $("#models-grid"); g.innerHTML = "";
    var rows = D.models.filter(function (m) {
      if (modelState.task && m.task !== modelState.task) return false;
      if (modelState.license === "permissive" && !isPermissive(m.license)) return false;
      if (modelState.license === "none" && isPermissive(m.license)) return false;
      var ws = weightState(m);
      if (modelState.weights === "yes" && ws !== "yes") return false;
      if (modelState.weights === "no" && ws === "yes") return false;
      if (modelState.weights === "external" && ws !== "external") return false;
      if (modelState.q) {
        var hay = [m.name, m.architecture, m.dataset, m.framework, m.license, m.xai, m.task].join(" ").toLowerCase();
        if (hay.indexOf(modelState.q) === -1) return false;
      }
      return true;
    });

    $("#model-count").textContent = rows.length + " sur " + D.models.length + " modèles";

    if (!rows.length) { g.appendChild(el("div", "empty", "Aucun modèle ne correspond à ces filtres.")); return; }

    rows.forEach(function (m) {
      var c = el("div", "mcard");

      var head = el("div", "mcard-head");
      var titles = el("div");
      titles.appendChild(el("h3", null, m.name));
      titles.appendChild(el("p", "arch", m.architecture));
      head.appendChild(titles);
      var ws = weightState(m);
      var t = el("span", "tag " + (ws === "yes" ? "tag-a" : ws === "external" ? "tag-b" : "tag-d"),
        ws === "yes" ? "Poids dans le dépôt" : ws === "external" ? "Poids externes" : "Aucun poids");
      head.appendChild(t);
      c.appendChild(head);

      var dl = el("dl", "kv");
      function row(k, v) {
        dl.appendChild(el("dt", null, k));
        var dd = el("dd");
        if (v) dd.textContent = v;
        dl.appendChild(dd);
      }
      row("Tâche", m.task);
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
      c.appendChild(diff);

      var actions = el("div", "mcard-actions");
      actions.appendChild(linkBtn(firstUrl(m.code), "GitHub", true));
      var rep = D.repositories.filter(function (r) { return r.repo_id === m.repo_id; })[0];
      if (rep) {
        var b = el("a", "btn", "Paper");
        b.href = "#papers";
        actions.appendChild(b);
      }
      var ds = findDataset(m.dataset);
      actions.appendChild(linkBtn(ds ? ds.url : null, "Dataset"));
      c.appendChild(actions);
      g.appendChild(c);
    });
  }

  /* ---------------- 4. XAI ---------------- */

  var xaiState = { q: "", cat: "" };

  function renderXai() {
    var cats = {};
    D.xai_methods.forEach(function (x) { cats[x.category] = 1; });
    var sel = $("#xai-category");
    Object.keys(cats).sort().forEach(function (c) {
      var o = el("option", null, c); o.value = c; sel.appendChild(o);
    });
    $("#xai-search").addEventListener("input", function (e) { xaiState.q = e.target.value.toLowerCase(); drawXai(); });
    sel.addEventListener("change", function (e) { xaiState.cat = e.target.value; drawXai(); });
    drawXai();
  }

  function drawXai() {
    var g = $("#xai-grid"); g.innerHTML = "";
    var rows = D.xai_methods.filter(function (x) {
      if (xaiState.cat && x.category !== xaiState.cat) return false;
      if (xaiState.q) {
        var hay = [x.name, x.category, x.principle, x.compatible, x.advantages, x.limitations, x.libraries].join(" ").toLowerCase();
        if (hay.indexOf(xaiState.q) === -1) return false;
      }
      return true;
    });
    $("#xai-count").textContent = rows.length + " sur " + D.xai_methods.length + " méthodes";
    if (!rows.length) { g.appendChild(el("div", "empty", "Aucune méthode d'IA explicable ne correspond à ces filtres.")); return; }

    rows.forEach(function (x) {
      var c = el("div", "xcard");
      var head = el("div", "xcard-head");
      head.appendChild(el("h3", null, x.name));
      head.appendChild(el("span", "cat", x.category));
      c.appendChild(head);

      c.appendChild(el("p", "principle", x.principle));

      c.appendChild(el("h4", null, "Principe"));
      c.appendChild(el("p", null, x.how));

      if (x.prospect_prostate) {
        c.appendChild(el("h4", null, "Persistance prostatique"));
        c.appendChild(el("p", "rel", x.prospect_prostate));
      }

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
      code.appendChild(el("span", null, "Code : "));
      code.appendChild(linkBtn(firstUrl((x.code || "").split(" | ")[0]), "ouvrir", true));
      foot.appendChild(code);
      c.appendChild(foot);
      g.appendChild(c);
    });
  }

  /* ---------------- 5. Datasets ---------------- */

  function renderDatasets() {
    var tb = $("#datasets-table tbody"); tb.innerHTML = "";
    D.datasets.forEach(function (d) {
      var tr = el("tr");
      var td = el("td");
      td.appendChild(el("span", "ds-name", d.name));
      if (d.paper) { var p = el("a", "ds-note", "article / fiche"); p.href = d.paper; p.target = "_blank"; p.rel = "noopener noreferrer"; td.appendChild(p); }
      tr.appendChild(td);
      tr.appendChild(el("td", null, d.type));
      var im = el("td", "num", d.images);
      if (d.images && String(d.images).length > 22) { var s = el("span", "ds-note", d.images); im.textContent = ""; im.appendChild(s); }
      tr.appendChild(im);
      tr.appendChild(el("td", "num", d.patients));
      tr.appendChild(el("td", null, d.classes));
      tr.appendChild(el("td", null, d.masks));
      tr.appendChild(el("td", null, d.access));
      tr.appendChild(el("td", null, d.license));
      var rec = d.recommended.split(" ")[0];
      var cls = rec === "OUI" ? "rec-yes" : rec === "PEUT-ÊTRE" ? "rec-maybe" : "rec-no";
      tr.appendChild(el("td", cls, d.recommended.split("—")[0].trim()));
      tb.appendChild(tr);
    });

    var g = $("#dataset-cards"); g.innerHTML = "";
    D.datasets.forEach(function (d) {
      var c = el("div", "dcard");
      c.appendChild(el("h3", null, d.name));
      c.appendChild(el("p", "dtype", d.type));

      var dl = el("dl", "kv");
      function row(k, v) { dl.appendChild(el("dt", null, k)); dl.appendChild(el("dd", null, v)); }
      row("Images", d.images);
      row("Patients", d.patients);
      row("Classes", d.classes);
      row("Masks", d.masks);
      row("Split", d.split);
      row("Licence", d.license);
      row("Access", d.access);
      row("Format", d.format);
      row("Used by", d.used_by);
      c.appendChild(dl);

      if (d.caveat) c.appendChild(el("div", "caveat", "Réserve — " + d.caveat));
      if (d.quality_notes) c.appendChild(el("div", "rec", "Contrôle qualité — " + d.quality_notes));
      else if (d.recommended) c.appendChild(el("div", "rec", d.recommended));

      var a = el("div", "dcard-actions");
      a.appendChild(linkBtn(d.url, "Source officielle", true));
      if (d.paper) a.appendChild(linkBtn(d.paper, "Fiche"));
      c.appendChild(a);
      g.appendChild(c);
    });
  }

  /* ---------------- 6. Papers ---------------- */

  var paperState = { q: "", f: "" };

  function classifyPaper(p) {
    var t = (p.title + " " + p.venue + " " + p.authors).toLowerCase();
    if (t.indexOf("survey") !== -1 || t.indexOf("review") !== -1) return "survey";
    if (t.indexOf("dataset") !== -1 || t.indexOf("benchmark") !== -1 || t.indexOf("brisc") !== -1 || t.indexOf("challenge") !== -1) return "dataset";
    if (p.xai && p.xai !== "None" && t.indexOf("model") === -1) return "xai";
    if (p.xai && p.xai !== "None") return "xai";
    return "model";
  }

  function renderPapers() {
    $("#paper-search").addEventListener("input", function (e) { paperState.q = e.target.value.toLowerCase(); drawPapers(); });
    $("#paper-filter").addEventListener("change", function (e) { paperState.f = e.target.value; drawPapers(); });
    drawPapers();
  }

  function drawPapers() {
    var g = $("#papers-list"); g.innerHTML = "";
    var rows = D.papers.filter(function (p) {
      if (paperState.f && classifyPaper(p) !== paperState.f) return false;
      if (paperState.q) {
        var hay = [p.title, p.venue, p.authors, p.model, p.dataset, p.metrics, p.xai].join(" ").toLowerCase();
        if (hay.indexOf(paperState.q) === -1) return false;
      }
      return true;
    });
    $("#paper-count").textContent = rows.length + " sur " + D.papers.length + " articles";
    if (!rows.length) { g.appendChild(el("div", "empty", "Aucun article ne correspond à ces filtres.")); return; }

    rows.forEach(function (p) {
      var c = el("div", "paper");
      var head = el("div", "paper-head");
      var left = el("div");
      var h = el("h3"); var a = el("a", null, p.title); a.href = p.url; a.target = "_blank"; a.rel = "noopener noreferrer";
      h.appendChild(a); left.appendChild(h);
      left.appendChild(el("p", "venue", p.venue + " · " + p.authors));
      head.appendChild(left);
      head.appendChild(el("span", "year", p.year));
      c.appendChild(head);

      var meta = el("div", "paper-meta");
      function chip(k, v) {
        if (!v) return;
        var s = el("span", "chip");
        s.appendChild(el("b", null, k + " : "));
        s.appendChild(document.createTextNode(v));
        meta.appendChild(s);
      }
      chip("Modèle", p.model);
      chip("Jeu de données", p.dataset);
      chip("Tâche", p.task);
      chip("Métriques", p.metrics);
      chip("XAI", p.xai);
      c.appendChild(meta);

      if (p.limits) {
        var lp = el("p", "limits");
        lp.appendChild(el("b", null, "Limites — "));
        lp.appendChild(document.createTextNode(p.limits));
        c.appendChild(lp);
      }

      var act = el("div", "paper-actions");
      act.appendChild(linkBtn(p.url, "Article", true));
      if (firstUrl(p.code)) act.appendChild(linkBtn(firstUrl(p.code), "Code"));
      var dsMatch = findDataset(p.dataset);
      if (dsMatch) act.appendChild(linkBtn(dsMatch.url, "Jeu de données"));
      else if (p.dataset_available && p.dataset_available !== "n/a") act.appendChild(linkBtn(null, "Jeu de données disponible"));
      if (p.weights && p.weights.indexOf("http") === 0) act.appendChild(linkBtn(p.weights, "Poids"));
      else if (p.weights && p.weights !== "n/a") act.appendChild(linkBtn(null, "Poids : " + p.weights.slice(0, 34) + (p.weights.length > 34 ? "…" : "")));
      if (p.pmid) act.appendChild(linkBtn("https://pubmed.ncbi.nlm.nih.gov/" + p.pmid + "/", "PubMed"));
      c.appendChild(act);
      g.appendChild(c);
    });
  }

  /* Best-effort match between a free-text dataset mention and a dataset record. */
  function findDataset(mention) {
    if (!mention) return null;
    var m = String(mention).toLowerCase();
    var best = null, bestLen = 0;
    D.datasets.forEach(function (d) {
      var key = String(d.name).toLowerCase().split(" (")[0].split(" 20")[0].trim();
      if (key.length > 3 && m.indexOf(key) !== -1 && key.length > bestLen) { best = d; bestLen = key.length; }
    });
    return best;
  }

  /* ---------------- 7. Comparison ---------------- */

  var cmpRows = [];
  var cmpSort = { key: "name", asc: true };
  var cmpState = { q: "", license: "" };

  function buildCmpRows() {
    cmpRows = D.models.map(function (m) {
      var rep = D.repositories.filter(function (r) { return r.repo_id === m.repo_id; })[0];
      var ds = D.datasets.filter(function (x) {
        return m.dataset && m.dataset.toLowerCase().indexOf(x.name.toLowerCase().split(" (")[0].split(" 20")[0].trim()) !== -1;
      })[0];
      var ws = weightState(m);
      var perf = (m.performance || "").split(".")[0].split(",")[0];
      return {
        name: m.name,
        task: m.task,
        model: m.architecture,
        dataset: ds ? ds.name : m.dataset,
        xai: m.xai,
        performance: perf,
        code: rep ? (rep.category === "D" ? "vide / non vérifié" : rep.category === "C" ? "code partiel" : "oui") : "dépôt",
        weights: ws === "yes" ? "dans le dépôt" : ws === "external" ? "externes" : "aucun",
        license: m.license,
        difficulty: m.difficulty,
        recommendation: rep ? ("Cat " + rep.category) : "-",
        _permissive: isPermissive(m.license),
        _rec: rep ? rep.category : "C"
      };
    });
  }

  function renderComparison() {
    buildCmpRows();
    $("#cmp-search").addEventListener("input", function (e) { cmpState.q = e.target.value.toLowerCase(); drawCmp(); });
    $("#cmp-license").addEventListener("change", function (e) { cmpState.license = e.target.value; drawCmp(); });

    $$("#cmp-table thead th[data-sort]").forEach(function (th) {
      th.addEventListener("click", function () {
        var k = th.getAttribute("data-sort");
        if (cmpSort.key === k) cmpSort.asc = !cmpSort.asc; else { cmpSort.key = k; cmpSort.asc = true; }
        drawCmp();
      });
    });
    drawCmp();
  }

  function drawCmp() {
    var tb = $("#cmp-table tbody"); tb.innerHTML = "";
    var rows = cmpRows.filter(function (r) {
      if (cmpState.license === "permissive" && !r._permissive) return false;
      if (cmpState.license === "none" && r._permissive) return false;
      if (cmpState.q) {
        var hay = [r.name, r.task, r.model, r.dataset, r.xai, r.performance, r.code, r.weights, r.license, r.difficulty, r.recommendation].join(" ").toLowerCase();
        if (hay.indexOf(cmpState.q) === -1) return false;
      }
      return true;
    });

    var k = cmpSort.key;
    rows.sort(function (a, b) {
      var x = String(a[k] || "").toLowerCase(), y = String(b[k] || "").toLowerCase();
      return cmpSort.asc ? (x < y ? -1 : x > y ? 1 : 0) : (x > y ? -1 : x < y ? 1 : 0);
    });

    $("#cmp-count").textContent = rows.length + " sur " + cmpRows.length + " solutions";

    $$("#cmp-table thead th[data-sort]").forEach(function (th) {
      th.classList.remove("sorted", "asc");
      if (th.getAttribute("data-sort") === k) { th.classList.add("sorted"); if (cmpSort.asc) th.classList.add("asc"); }
    });

    if (!rows.length) { tb.appendChild(el("tr")).appendChild(el("td", "empty", "Aucune solution ne correspond à ces filtres.")); return; }

    rows.forEach(function (r) {
      var tr = el("tr");
      var td0 = el("td");
      td0.appendChild(el("span", "ds-name", r.name));
      tr.appendChild(td0);
      tr.appendChild(el("td", null, r.task));
      tr.appendChild(el("td", null, r.model));
      tr.appendChild(el("td", null, r.dataset));
      tr.appendChild(el("td", null, r.xai));
      tr.appendChild(el("td", null, r.performance));
      tr.appendChild(el("td", null, r.code));
      var wtd = el("td");
      var wc = r.weights === "dans le dépôt" ? "tag-a" : r.weights === "externes" ? "tag-b" : "tag-d";
      wtd.appendChild(el("span", "tag " + wc, r.weights));
      tr.appendChild(wtd);
      var ltd = el("td");
      ltd.appendChild(el("span", "tag " + (r._permissive ? "tag-a" : "tag-d"), r._permissive ? "Permissive" : "Absente / floue"));
      tr.appendChild(ltd);
      tr.appendChild(el("td", null, r.difficulty));
      var rtd = el("td");
      rtd.appendChild(el("span", "tag tag-" + (r._rec === "A" ? "a" : r._rec === "B" ? "b" : r._rec === "C" ? "c" : "d"), r.recommendation));
      tr.appendChild(rtd);
      tb.appendChild(tr);
    });
  }

  /* ---------------- 8. Architecture ---------------- */

  function renderArchitecture() {
    var g = $("#pipeline"); g.innerHTML = "";
    D.project_architecture.forEach(function (s, i) {
      var row = el("div", "pipe-step");
      var h = el("div", "pipe-head");
      h.appendChild(el("div", "pipe-idx", "STEP " + (i + 1)));
      h.appendChild(el("div", "pipe-name", s.name));
      row.appendChild(h);
      row.appendChild(el("div", "pipe-body", s.detail));
      g.appendChild(row);
      if (i < D.project_architecture.length - 1) g.appendChild(el("div", "pipe-arrow", "↓"));
    });
  }

  /* ---------------- 9. Sources ---------------- */

  function renderSources() {
    var seen = {}, srcs = [];
    function add(type, title, url, meta) {
      if (!url || seen[url]) return;
      seen[url] = 1;
      srcs.push({ type: type, title: title, url: url, meta: meta || "" });
    }
    D.repositories.forEach(function (r) { add(r.type || "GitHub", r.name + " — " + r.github, r.github, r.license + " · " + (r.stars ? r.stars + " étoiles · " : "") + "Cat " + r.category); });
    D.datasets.forEach(function (d) { add("Jeu de données", d.name, d.url, d.license); if (d.paper) add("Fiche", d.name + " fiche", d.paper, d.venue || ""); });
    D.papers.forEach(function (p) { add("Article", p.title, p.url, p.venue + " · " + p.year); });
    D.models.forEach(function (m) { if (m.code && m.code.indexOf("http") === 0) add("Code modèle", m.name, m.code, m.license); });
    D.xai_methods.forEach(function (x) {
      (x.code || "").split(" | ").forEach(function (u) { if (u.indexOf("http") === 0) add("Code XAI", x.name + " implémentation", u, x.category); });
    });

    var sel = $("#src-type");
    var types = {};
    srcs.forEach(function (s) { types[s.type] = 1; });
    Object.keys(types).sort().forEach(function (t) { var o = el("option", null, t); o.value = t; sel.appendChild(o); });

    var state = { q: "", t: "" };
    $("#src-search").addEventListener("input", function (e) { state.q = e.target.value.toLowerCase(); draw(); });
    sel.addEventListener("change", function (e) { state.t = e.target.value; draw(); });

    function draw() {
      var g = $("#sources-list"); g.innerHTML = "";
      var rows = srcs.filter(function (s) {
        if (state.t && s.type !== state.t) return false;
        if (state.q && (s.title + " " + s.url + " " + s.meta).toLowerCase().indexOf(state.q) === -1) return false;
        return true;
      });
      $("#src-count").textContent = rows.length + " sur " + srcs.length + " sources";
      if (!rows.length) { g.appendChild(el("div", "empty", "Aucune source ne correspond à ces filtres.")); return; }
      rows.forEach(function (s) {
        var c = el("div", "src");
        c.appendChild(el("span", "src-type", s.type));
        var a = el("a", null, s.title); a.href = s.url; a.target = "_blank"; a.rel = "noopener noreferrer";
        c.appendChild(a);
        c.appendChild(el("div", "src-meta", s.meta));
        g.appendChild(c);
      });
    }
    draw();

    var v = D.verification;
    var box = $("#verification");
    var ul = el("ul", "tight");
    [
      ["Date", v.date],
      ["Méthode", v.method],
      ["Dépôts inspectés", v.repos_checked],
      ["Fichiers de poids sondés en HTTP", v.weights_files_probed],
      ["Licences confirmées", v.license_confirmed],
      ["Dépôts signalés comme sans licence", v.license_missing_flagged],
      ["Liens vérifiés", v.links_checked],
      ["Liens en échec puis corrigés", v.links_broken]
    ].forEach(function (p) {
      var li = el("li");
      li.appendChild(el("b", null, p[0] + ": "));
      li.appendChild(document.createTextNode(p[1]));
      ul.appendChild(li);
    });
    box.appendChild(ul);
    var note = el("p", "note", v.note);
    box.appendChild(note);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
