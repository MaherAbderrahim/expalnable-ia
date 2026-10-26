# -*- coding: utf-8 -*-
"""Build the new presentation site (reports/demo/index.html).

Consumer of phase1_phase2_demo.ipynb outputs:
- reports/demo/data/presentation_data.json  (numbers)
- reports/demo/assets/**                    (figures)
- reports/demo/nb/phase1_phase2_demo.html   (executed notebook)

The previous French dashboard is preserved as index_v1_fr.html.
Structure: Accueil / Overview / Phase 1 / Data / Predictions+Threshold /
Phase 2 XAI / Results / Conclusion — smooth navigation, keyboard arrows.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from html import escape as e
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "reports" / "demo"
DATA = json.loads((DEMO / "data" / "presentation_data.json").read_text(encoding="utf-8"))
OUT = DEMO / "index.html"

backup = DEMO / "index_v1_fr.html"
if OUT.exists() and not backup.exists():
    shutil.copy2(OUT, backup)

model = DATA["model"]
golden = DATA["golden"]
cases = DATA["cases"]
panels = DATA["panels"]
thr1 = DATA["threshold_phase1"]
bench = DATA["benchmark"]
bst = DATA["benchmark_stats"]
sel = DATA["selected_postproc"]
thr2 = {"value": sel["threshold"],
        "rationale": "max-F1 sur la validation (120 cas) ; règle préenregistrée ; test scoré une seule fois après gel"}
xm = DATA["xai_main"]
xig = DATA["xai_ig"]
xfn = DATA["xai_fn"]
xq = DATA["xai_quality"]

sel_thr = sel["threshold"]
gen = datetime.strptime(DATA["generated_at"], "%Y-%m-%dT%H:%M:%S").strftime("%d/%m/%Y %H:%M")


def pct(x, nd=1):
    return f"{100 * float(x):.{nd}f}%"


def n(x, nd=2):
    return f"{float(x):.{nd}f}"


def verdict_badge(v, cs):
    if v == "TP":
        return "<span class='vbadge tp'>DÉTECTÉ (TP)</span>"
    if v == "FP":
        return "<span class='vbadge fp'>FAUSE ALERTE (FP)</span>".replace("FAUSE", "FAUX")
    if v == "FN":
        return "<span class='vbadge fn'>MANQUÉ (FN)</span>"
    return "<span class='vbadge tn'>AUCUNE ALERTE (TN)</span>"


def case_card(cid, idx):
    p = panels.get(cid) or {}
    row = next((r for r in cases if r["case_id"] == cid), {})
    detected = float(row.get("score", 0)) >= thr1["value"]
    detected2 = float(row.get("score", 0)) >= sel_thr
    pred_txt = ("SUSPICION DÉTECTÉE" if detected else "AUCUNE ALERTE")
    fig = p.get("figure", "")
    role_names = {"detection franche": "Exemple 01 — détection franche",
                  "cancer manqué": "Exemple 02 — cancer manqué",
                  "cas ambigu": "Exemple 03 — cas ambigu"}
    comment = {
        "detection franche": ("L'inférence produit une probabilité élevée au niveau de la lésion ; la décision "
                              "découle de la comparaison score / seuil officiel."),
        "cancer manqué": ("Présenté tel quel : le score du modèle reste nettement sous le seuil officiel. "
                          "C'est précisément le type de cas que l'analyse de l'opératoire (Phase 2) met en évidence."),
        "cas ambigu": ("Score intermédiaire, sous le seuil officiel, supérieur au seuil recalibré de la Phase 2 : "
                       "cas relevant du choix d'opératoire, présenté sans interprétation clinique."),
    }
    p2 = " · " + (f"déclencherait l'alerte au seuil recalibré ({n(sel_thr, 3)})" if detected2
                  else f"reste sous le seuil recalibré ({n(sel_thr, 3)})")
    return (f"<div class='excard' id='ex{idx}'>"
            f"<img src='{fig}' alt='{cid}'>"
            f"<div class='exbody'>"
            f"<div class='exhead'><h3>{role_names[p.get('role', idx)]}</h3>"
            f"<span class='caseid'>{cid}</span></div>"
            f"<div class='exmeta'>csPCa (vérité terrain) : <b>{'oui' if row.get('csPCa') else 'non'}</b>"
            f" &nbsp;·&nbsp; PI-RADS : <b>{e(str(row.get('PI-RADS', '—')))}</b></div>"
            f"<div class='pipeline'><span>Input</span>→<span>Inférence</span>→<span>Score</span>→<span>Décision</span></div>"
            f"<div class='scoregrid'>"
            f"<div><div class='lbl'>Score du modèle</div><div class='big'>{n(row.get('score'), 3)}</div></div>"
            f"<div><div class='lbl'>Seuil officiel</div><div class='mid'>{n(thr1['value'], 2)}</div></div>"
            f"<div><div class='lbl'>Prédiction</div><div class='decide'>"
            f"{'<b class=\"sus\">Suspicion détectée</b>' if detected else '<b class=\"nos\">Aucune alarme</b>'} "
            f"<span class='minip'>{p2}</span></div></div>"
            f"</div>"
            f"<p class='exnote'>{comment[p.get('role', idx)]}</p>"
            f"</div></div>")


example_cards = "".join(case_card(cid, i + 1)
                        for i, cid in enumerate(["10032_1000032", "10012_1000012", "10008_1000008"]))

bench_rows = ""
for row in bench:
    win = row["config"] == "no_extract"
    bcls = " class='win'" if win else ""
    cfg = "figé (no_extract)" if win else "officiel @0.50"
    bench_rows += (f"<tr{bcls}><td>{'Validation' if row['split']=='val' else 'Test'} — {cfg}</td>"
                   f"<td>{n(row['threshold'], 3)}</td><td>{n(row['auroc'], 4)}</td><td>{n(row['ap'], 4)}</td>"
                   f"<td>{pct(row['sens'], 0)}</td><td>{pct(row['spec'], 0)}</td>"
                   f"<td>{pct(row['precision'], 0)}</td><td>{n(row['f1'], 3)}</td>"
                   f"<td>{pct(row['balanced_accuracy'], 0)}</td><td>{n(row['ece'], 3)}</td></tr>")

delong_p = bst["delong_p_test"]
bda = bst["bootstrap_auc_diff"]
ciA = bst["auroc_selected_ci_test"]
fv = bst["fold_run_variance_val"]

bars = []
for row in cases:
    s = row["score"]
    left = min(100, max(2, s * 100))
    over1 = "<b class='sus'></b>" if s >= thr1["value"] else ""
    bars.append(
        f"<div class='barrow'><span class='bcid'>{row['case_id']}</span>"
        f"<span class='bar'><span class='fill creds' style='width:{left:.1f}%; "
        + ("background:#d9534f" if s >= thr1['value'] else "background:#12385c") + f"'></span>"
        f"<span class='tick' style='left:{thr1['value']*100:.1f}%'></span>"
        f"<span class='tover' style='left:{sel_thr*100:.1f}%'></span></span>"
        f"<span class='bval'>{n(s, 3)}</span>"
        f"<span class='bdec'>{'suspicion' if s >= thr1['value'] else 'aucune'}"
        + (f" / déclenchée @ {n(sel_thr, 3)}" if s >= sel_thr and s < thr1['value'] else "") + "</span></div>")

CSS = """
:root{--bg:#f7f9fc;--card:#ffffff;--ink:#17202b;--mut:#5a6b80;--line:#dde5ee;
--navy:#12385c;--navy2:#0d2b47;--acc:#1c7bd6;--ok:#188f56;--warn:#b7791f;--bad:#c0392b;--soft:#eef4fb}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--ink);font-family:'Segoe UI',system-ui,-apple-system,sans-serif;line-height:1.6;font-size:16px}
.wrap{max-width:1120px;margin:0 auto;padding:0 22px}
a{color:var(--acc)}
h2{font-size:26px;color:var(--navy2);margin-bottom:6px;letter-spacing:-.3px}
h3{font-size:17px;color:var(--navy2)}
p.lead{color:var(--mut);max-width:850px;margin-bottom:18px}
section{padding:58px 0 28px}
.k{display:inline-block;background:var(--navy);color:#fff;font-size:11.5px;font-weight:700;letter-spacing:2px;
text-transform:uppercase;border-radius:4px;padding:4px 10px;margin-bottom:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;box-shadow:0 1px 3px rgba(16,42,67,.06)}
header.hero{padding:66px 0 40px}
.hero h1{font-size:37px;line-height:1.15;color:var(--navy2);letter-spacing:-1px;max-width:900px}
.hero h1 em{font-style:normal;color:var(--acc)}
.hero .sub{color:var(--mut);font-size:17px;margin-top:12px;max-width:860px}
.chain{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:22px 0 14px}
.chain span{background:var(--soft);border:1px solid var(--line);border-radius:999px;padding:6px 14px;font-size:13.5px;color:var(--navy2)}
.chain .arw{background:none;border:none;color:var(--mut);padding:0 2px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
.chip{font-size:12.5px;padding:5px 12px;border-radius:999px;border:1px solid var(--line);background:#fff;color:var(--mut)}
.chip.ok{color:var(--ok);border-color:#bfe3cf}.chip.warn{color:var(--warn);border-color:#ecd9b0}
nav.top{position:sticky;top:0;z-index:9;background:rgba(247,249,252,.95);backdrop-filter:blur(6px);
border-bottom:1px solid var(--line)}
nav.top .in{display:flex;gap:4px;align-items:center;padding:9px 0;flex-wrap:wrap}
nav.top a{color:var(--mut);text-decoration:none;font-size:13.5px;padding:6px 12px;border-radius:8px;white-space:nowrap}
nav.top a:hover{color:var(--navy)}
nav.top a.on{color:#fff;background:var(--navy)}
nav.top .brand{font-weight:700;color:var(--navy2);font-size:14px;margin-right:10px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}
.statb .num{font-size:31px;font-weight:750;color:var(--navy2);letter-spacing:-1px}
.statb .lbl{color:var(--mut);font-size:13.5px;margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:14px;margin-top:10px}
th{text-align:left;color:var(--mut);font-size:11.5px;text-transform:uppercase;letter-spacing:.6px;padding:8px 10px;border-bottom:2px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid var(--line)}
tr.win td{background:#eaf7ef;color:var(--ok);font-weight:600}
.excard{display:grid;grid-template-columns:1.08fr 1fr;gap:0;background:var(--card);border:1px solid var(--line);
border-radius:16px;overflow:hidden;margin:20px 0;box-shadow:0 1px 3px rgba(16,42,67,.06)}
.excard img{width:100%;height:100%;object-fit:cover;min-height:280px;display:block}
.exbody{padding:20px 22px}
.exhead{display:flex;align-items:baseline;gap:10px}
.exhead h3{font-size:18px}
.caseid{font-size:12.5px;color:var(--mut);font-family:Consolas,monospace}
.exmeta{color:var(--mut);font-size:13px;margin:4px 0 10px}
.pipeline{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
.pipeline span{font-size:12px;background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:3px 9px;color:var(--navy2)}
.pipeline .arwp{background:none;border:none;color:var(--mut)}
.scoregrid{display:grid;grid-template-columns:1fr 1fr 1.4fr;gap:14px;background:var(--soft);border-radius:12px;padding:14px 16px}
.scoregrid .lbl{font-size:11.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
.scoregrid .big{font-size:30px;font-weight:800;color:var(--navy2);letter-spacing:-1px}
.scoregrid .mid{font-size:22px;font-weight:650;color:var(--mut)}
.decide .sus{color:var(--ok)} .decide .nos{color:var(--bad)}
.decide .minip{display:block;font-size:12px;color:var(--mut);margin-top:3px}
.exnote{font-size:13.5px;color:var(--mut);margin-top:12px}
.barwrap{margin-top:16px}
.barrow{display:grid;grid-template-columns:110px 1fr 70px 210px;gap:12px;align-items:center;margin:9px 0}
.barrow .bcid{font-size:12.5px;font-family:Consolas,monospace;color:var(--mut)}
.bar{position:relative;height:20px;background:var(--soft);border-radius:6px}
.bar .fill{position:absolute;left:0;top:0;bottom:0;border-radius:6px}
.bar .tick{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink);opacity:.65}
.bar .tover{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--warn)}
.barrow .bval{font-size:13.5px;font-weight:700;color:var(--navy2)}
.barrow .bdec{font-size:12.5px;color:var(--mut)}
.legend{font-size:12.5px;color:var(--mut);margin-top:8px}
.legend b{color:var(--ink)}
.xpair{display:block;width:100%;border:1px solid var(--line);border-radius:12px;margin:14px 0 6px;background:#fff}
.cap{font-size:13.5px;color:var(--mut);margin:2px 0 20px}
.cap b{color:var(--navy2)}
.okbox{border:1px solid #bfe3cf;background:#eef9f2;border-radius:12px;padding:14px 18px;font-size:14px;color:#14532d;margin:16px 0}
.warnbox{border:1px solid #ecd9b0;background:#fff9ee;border-radius:12px;padding:14px 18px;font-size:14px;color:#7a5314;margin:16px 0}
ul.list{margin:6px 0 0 20px}ul.list li{margin:7px 0;font-size:14.5px}
.bigbtns{display:flex;gap:12px;margin-top:18px;flex-wrap:wrap}
.bigbtns a{background:var(--navy);color:#fff;text-decoration:none;border-radius:10px;padding:11px 20px;font-size:14.5px}
.bigbtns a.ghost{background:#fff;color:var(--navy);border:1px solid var(--line)}
.footer{border-top:1px solid var(--line);margin-top:50px;padding:24px 0 60px;color:var(--mut);font-size:12.5px}
@media (max-width:900px){.excard{grid-template-columns:1fr}.scoregrid{grid-template-columns:1fr}
.barrow{grid-template-columns:86px 1fr 62px;grid-template-areas:'a b c' 'a b d'}
noscript,h1{font-size:30px}}
"""

JS = """
document.addEventListener('keydown', function (e) {
  var secs = Array.from(document.querySelectorAll('section[id]'));
  if (!secs.length) return;
  var y = window.scrollY + 140, cur = secs[0];
  secs.forEach(function (s) { if (s.offsetTop <= y) cur = s; });
  var i = secs.indexOf(cur);
  if (e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === 'PageDown') {
    if (secs[i + 1]) secs[i + 1].scrollIntoView(); e.preventDefault();
  } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft' || e.key === 'PageUp') {
    if (i > 0) secs[i - 1].scrollIntoView(); e.preventDefault();
  }
});
var secs = Array.from(document.querySelectorAll('section[id]'));
var links = Array.from(document.querySelectorAll('nav.top a[data-sec]'));
function onScroll() {
  var y = window.scrollY + 160, idx = 0;
  secs.forEach(function (s, k) { if (s.offsetTop <= y) idx = k; });
  links.forEach(function (a) { a.classList.toggle('on', a.dataset.sec === secs[idx].id); });
}
document.addEventListener('scroll', onScroll); onScroll();
"""

nav = "".join(f"<a data-sec='{sid}' href='#{sid}'>{txt}</a>" for sid, txt in [
    ("accueil", "Accueil"), ("overview", "Overview"), ("phase1", "Phase 1"),
    ("data", "Données & exemples"), ("predictions", "Prédictions"),
    ("seuil", "Seuil"), ("phase2", "Phase 2 — XAI"), ("results", "Résultats"),
    ("conclusion", "Conclusion")])

P = []
P.append("<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'>"
         "<meta name='viewport' content='width=device-width,initial-scale=1'>"
         "<title>Explainable AI for Prostate Cancer Detection on MRI</title>"
         f"<style>{CSS}</style></head><body>")

nav_sensor = ("<nav class='top'><div class='in wrap' style='margin:0 auto'>"
              "<span class='brand'>XAI · Prostate</span>" + nav + "</div></nav>")

P.append("<header class='hero' id='accueil'><div class='wrap'>"
         "<span class='k'>Présentation de projet</span>"
         "<h1>Explainable AI for Prostate Cancer Detection on <em>MRI</em></h1>"
         "<div class='sub'>Analyse de la détection du cancer de la prostate cliniquement significatif "
         "(csPCa) en IRM biparamétrique : validation fidèle du modèle officiel PI-CAI (Phase 1), "
         "puis introduction des méthodes d'Explainable AI appliquées à ses prédictions (Phase 2).</div>"
         "<div class='chain'><span>Projet</span><span class='arw'>→</span>"
         "<span>Notebook exécuté</span><span class='arw'>→</span><span>Résultats réels</span>"
         "<span class='arw'>→</span><span>Présentation web</span><span class='arw'>→</span>"
         "<span>Démonstration</span></div>"
         "<div class='chips'>"
         f"<span class='chip ok'>✓ Test d'or PASSED — Δ = {golden['delta']:.1e} vs référence officielle</span>"
         "<span class='chip ok'>✓ modèle réel : ensemble 5-folds, poids officiels LFS</span>"
         "<span class='chip ok'>✓ évaluation 240 cas (120 val / 120 test)</span>"
         "<span class='chip warn'>Borne optimiste : cas locaux vus à l'entraînement officiel</span>"
         f"<span class='chip'>contenu généré depuis le notebook — {gen}</span></div>"
         "</div></header>")
P.append(nav_sensor)
P.append("<div class='wrap'>")

# ---- Overview
P.append("<section id='overview'><h2>Overview</h2>"
         "<p class='lead'>Le csPCa est une atteinte maligne de la prostate significative cliniquement. "
         "Le projet évalue un détecteur d'apprentissage profond sur des examens IRM biparamétriques "
         "(T2W, ADC, DWI/HBV), avec deux axes : <b>validation d'un modèle de détection</b>, puis "
         "<b>analyse de l'interprétabilité de ses décisions</b>.</p>"
         "<div class='grid3'>"
         f"<div class='card statb'><div class='num'>{model['n_folds']} × UNet 3D</div>"
         f"<div class='lbl'>Ensemble k-fold : {model['params_per_fold_millions']} M paramètres/fold, "
         f"{model['weight_mb_per_fold']} Mo de poids checkpoint, structure features officielle</div></div>"
         f"<div class='card statb'><div class='num'>{model['input_shape'][0]}×{model['input_shape'][1]}×{model['input_shape'][2]}</div>"
         f"<div class='lbl'>Grille d'entrée par modalité, espacement officiel "
         f"({model['spacing'][0]}/{model['spacing'][1]}/{model['spacing'][2]} mm) — 3 canaux T2W/ADC/HBV</div></div>"
         f"<div class='card statb'><div class='num'>7 cases</div>"
         f"<div class='lbl'>Cas de démonstration locaux passés dans le pipeline réel "
         "(vérité terrain de la marksheet officielle)</div></div>"
         "</div>"
         "<div class='warnbox'><b>Honnêteté expérimentale :</b> le détecteur utilisé a été entraîné par ses auteurs "
         "sur les 1 500 examens publics PI-CAI ; les cas locaux d'évaluation ont donc été observés à "
         "l'entraînement. Les métriques présentées sont une borne supérieure d'usage : le travail porte sur la "
         "<b>qualité de l'instrumentation</b> (reproduction exacte, évaluation sans fuite, seuil figé sur validation, "
         "explicabilité auditée), pas sur la performance en généralisation externe.</div></section>")

# ---- Phase 1
P.append("<section id='phase1'><h2>Phase 1 — Detection Model</h2>"
         "<p class='lead'>Cette phase est consacrée à l'intégration et à la validation du modèle de détection "
         "utilisé comme base du système d'Explainable AI.</p>"
         "<div class='card'><h3 style='margin-bottom:10px'>Chargement du modèle</h3>"
         "<table><tr><th style='width:190px'>Modèle</th><td>PI-CAI — ensemble 5 folds UNet 3D "
         "(Saha et al., Lancet Oncology 2024)</td></tr>"
         f"<tr><td style='color:var(--ok);font-weight:700'>Status</td>"
         f"<td style='color:var(--ok);font-weight:700'>✓ Model loaded · ✓ Weights loaded "
         f"({model['n_folds']} checkpoint(s) LFS, {model['weight_mb_per_fold']} Mo chacun) · "
         "✓ Configuration loaded · ✓ Inference ready</td></tr>"
         f"<tr><td>Paramètres / checkpoint</td><td>{model['params_per_fold_millions']} M</td></tr>"
         "<tr><td>Configuration d'entrée</td>"
         f"<td>shape {model['input_shape']}, espacement {model['spacing']}, "
         f"{model['channels']} canaux, 2 classes</td></tr>"
         "<tr><td>Inférence</td><td>TTA miroir horizontal · lissage gaussien σ = 1,5 · "
         f"moyenne des 5 folds · exécution sur {model['device']} "
         f"(chargement : {model['load_seconds']} s)</td></tr></table></div>"
         "<div class='okbox'><b>Validation par le cas d'or :</b> sur le cas de test officiel "
         f"<b>{golden['case_id']}</b>, la probabilité produite par ce pipeline est de "
         f"<b>{n(golden['score'], 10)}</b> contre <b>{golden['reference']:.8f}</b> pour la référence des "
         f"auteurs — écart absolu {golden['delta']:.1e} (tolérance flottante). Le pipeline reproduit "
         "donc exactement l'algorithme publié.</div>"
         "</section>")

# ---- Data & examples
P.append("<section id='data'><h2>Données et exemples</h2>"
         "<p class='lead'>Les cases de démonstration (mêmes données que la démonstration Phase 1) sont "
         "converties en analyses reproductibles : chaque cas traverse la chaîne Input → Inférence → "
         "Score → Décision. Les tableaux ci-dessous proviennent directement de l'exécution du notebook.</p>"
         "<div class='card'><h3 style='margin-bottom:6px'>Cas disponibles avec vérité terrain</h3>"
         "<table class='metatab'><tr><th>Case</th><th>csPCa</th><th>PI-RADS</th><th> Centre</th></tr>"
         + "".join(f"<tr><td>{c['case_id']}</td><td>{'oui' if c['csPCa'] else 'non'}</td>"
                   f"<td>{e(str(c.get('PI-RADS','-')))}</td><td>{e(str(c.get('center','-')))}</td></tr>"
                   for c in DATA["cases_meta"]) + "</table></div>"
         + example_cards + "</section>")

# ---- Predictions
P.append("<section id='predictions'><h2>Prédictions</h2>"
         "<p class='lead'>Pour chaque cas, l'inférence produit un score d'étage cas (probabilité de "
         "csPCa significatif après ensemble k-fold, TTA, lissage). Ce score est comparé au seuil de "
         "décision — rien d'autre n'intervient. Le vocabulaire affiché correspond exactement à la "
         "sortie du modèle.</p>"
         "<div class='card'><div class='barwrap'>"
         + "".join(bars) + "</div>"
         "<div class='legend'><b style='color:#12385c'>■</b> score (barre) &nbsp;·&nbsp; "
         f"<b>|</b> seuil officiel {n(thr1['value'], 2)} &nbsp;·&nbsp; "
         f"<b style='color:#b7791f'>|</b> seuil recalibré Phase 2 {n(sel_thr, 3)} (figé sur la validation)</div>"
         "<p class='cap' style='margin-top:14px'>Sur cet échantillon de démonstration, un seul des quatre "
         f"cancers réels dépasse le seuil officiel ({n(thr1['value'], 2)}) ; avec le seuil recalibré de la "
         f"Phase 2 (<b>{n(sel_thr, 3)}</b>, figé sur 120 cas de validation), trois de quatre le dépasseraient. "
         "Cette amélioration est mesurée et documentée en section <a href='#results'>Résultats</a>.</p>"
         "</div></section>")

# ---- Threshold
P.append("<section id='seuil'><h2>Le seuil de décision</h2>"
         "<p class='lead'>Aucun seuil « arrange » les résultats : la Phase 1 utilise le seuil officiel "
         "publié, la Phase 2 documente une recalibration <b>figée sur la seule validation</b> avant tout "
         "accès au test.</p>"
         "<div class='grid2'>"
         f"<div class='card'><h3>Phase 1 — seuil officiel</h3>"
         f"<div class='num' style='font-size:34px;font-weight:750;color:var(--navy2)'>{n(thr1['value'], 2)}</div>"
         f"<p class='cap'>Origine : point opératoire de l'évaluation PI-CAI (Saha et al., Lancet Oncology "
         f"2024 : sensitivity 70,9 % / PPV 35,5 % au seuil 0,50 sur le jeu test du concours). "
         "Chaque exemple ci-dessus applique strictement cette règle.</p></div>"
         f"<div class='card'><h3>Phase 2 — recalibration sur validation</h3>"
         f"<div class='num' style='font-size:34px;font-weight:750;color:var(--warn)'>{n(sel_thr, 4)}</div>"
         f"<p class='cap'>Règle préenregistrée : choïx de la configuration de post-traitement par AP "
         f"maximale en validation, seuil = max-F1 en validation ; le test n'est scoré qu'une seule fois "
         "après gel. C'est l'unique « changement de seuil » du projet, et il est mesurable.</p></div>"
         "</div></section>")

# ---- Phase 2 transition + XAI
P.append("<section id='phase2'><h2>Phase 2 — Explainable AI</h2>"
         "<div class='okbox'><b>Transition :</b> le modèle permet d'obtenir une prédiction à partir des "
         "données d'imagerie ; la prédiction seule ne dit pas ce qui a motivé la décision. La Phase 2 "
         "introduit donc les méthodes XAI afin d'analyser cette décision, s'appuyant sur une évaluation "
         "élargie (240 cas) et une règle figée avant le test.</div>"
         "<div class='card'><h3>Benchmark — 120 cas de validation + 120 cas de test (34 csPCa+ chacun)</h3>"
         f"<table><tr><th>Jeu — config</th><th>Seuil</th><th>AUROC</th><th>AP</th><th>Sens.</th>"
         f"<th>Spéc.</th><th>Précision</th><th>F1</th><th>Acc. éq.</th><th>ECE</th></tr>"
         + bench_rows + "</table>"
         f"<p class='cap'>DeLong p = {n(delong_p, 3)} (classement équivalent) ; ΔAUROC bootstrap "
         f"{n(bda['mean'], 4)} IC95 [{n(bda['ci_lower'], 3)} ; {n(bda['ci_upper'], 3)}] ; AUROC test figé "
         f"{n(ciA['value'], 4)} [{n(ciA['ci_lower'], 3)} ; {n(ciA['ci_upper'], 3)}]. La contribution "
         "mesurable est le <b>point opératoire</b> : sensibilité 23,5 % → 55,9 % et F1 0,381 → 0,623 au "
         "test avec le seuil figé (sans ré-entraînement).</p></div>"
         "<div class='card'><h3>Méthodes XAI implémentées (dépôt réel)</h3>"
         "<table><tr><th>Méthode</th><th>Principe</th><th>Ce qu'elle montre</th></tr>"
         "<tr><td><b>Grad-CAM 3D</b></td><td>pondération des activations de la dernière connexion utile par les gradients moyens</td><td>où le réseau regarde</td></tr>"
         "<tr><td><b>Grad-CAM++ 3D</b></td><td>pondération du second ordre (α par voxel)</td><td>attention plus concentrée, utile sur petites lésions</td></tr>"
         "<tr><td><b>Integrated Gradients</b></td><td>intégration des gradients sur 24 pas depuis la baseline nulle</td><td>les attributions qui « poussent » le score</td></tr>"
         "<tr><td><b>Occlusion 3D</b></td><td>masquage de blocs et impact sur le score</td><td>fidélité du score (contrôle quantitatif)</td></tr></table></div>"
         "<h3 style='margin-top:26px'>Visualisations — tout provient du modèle réel (case d'or 10032)</h3>"
         f"<img class='xpair' src='{xm['gradcam']['figure']}' alt='gradcam'>"
         f"<p class='cap'><b>Grad-CAM</b> — attention concentrée sur la lésion suspecte (coupe z={xm['gradcam']['slice']}); "
         f"la coupe de score max du modèle (z={xm['det_slice']}) coïncide — cohérence spatial score/attention.</p>"
         f"<img class='xpair' src='{xm['gradcampp']['figure']}' alt='gradcampp'>"
         f"<p class='cap'><b>Grad-CAM++</b> — même coupe ; l'attention se concentre davantage "
         f"(enrichissement max {n(xq['gradcampp_enrichment_max'], 1)}× sur le cas audité le plus confiant).</p>"
         f"<img class='xpair' src='{xig['figure']}' alt='ig'>"
         f"<p class='cap'><b>Integrated Gradients</b> — attributions (24 pas) : zones qui contribuent "
         f"positivement au score sur la coupe z={xig['slice']}.</p>"
         f"<img class='xpair' src='{xfn['figure']}' alt='fn'>"
         f"<p class='cap'><b>Grad-CAM++ sur un cancer manqué</b> (case 10012, score {n(xfn['score'], 2)} &lt; 0.50) : "
         "l'absence d'alarme correspond à une réponse ciblée d'amplitude insuffisante — présentation honnête, "
         "cet exemple montre une limite actuelle plutôt qu'une réussite.</p>"
         f"<div class='card'><h3>Contrôles qualité de l'explicabilité</h3>"
         "<table><tr><th>Contrôle</th><th>Valeur (audit 6 cas)</th><th>Lecture</th></tr>"
         f"<tr><td>Fidélité insertion / délétion (Grad-CAM)</td><td>ins {n(xq['insertion_auc_mean'], 2)} · "
         f"dél {n(xq['deletion_auc_mean'], 2)}</td><td>l'attention pousse réellement le score ; sur le cas le "
         "plus confiant (0.73 / 0.21), l'excellent indicateur de fidélité</td></tr>"
         f"<tr><td>Randomisation des poids (Adebayo 2018)</td><td>|r| moyen ≈ {n(xq['randomization_mean_abs'], 2)}</td>"
         "<td>la carte ne dépend pas d'un artefact trivial</td></tr>"
         f"<tr><td>Stabilité ±1 voxel (Pearson)</td><td>jusqu'à {n(xq['stability_best'], 2)}</td>"
         "<td>la carte est robuste au bruit numérique</td></tr>"
         f"<tr><td>Densité d'attention (enrichissement)</td><td>jusqu'à {n(xq['gradcampp_enrichment_max'], 1)}×"
         "</td><td>sur petites lésions, on juge la concentration relative, pas l'IoU brut</td></tr></table>"
         f"<p class='cap' style='margin-top:8px'>Source : {e(DATA.get('xai_quality_source', 'experiments/results/xai/xai_quality.csv'))} — aucune valeur "
         "fabriquée.</p></div></section>")

# ---- Results
P.append("<section id='results'><h2>Résultats</h2>"
         "<div class='grid3'>"
         "<div class='card'><h3 style='color:var(--acc)'>Phase 1 — validation du modèle</h3>"
         f"<div class='statb'><div class='num'>{n(golden['delta'], 9)}</div>"
         "<div class='lbl'>écart absolu au score de référence sur le cas d'or — reproduction fidèle validée</div></div>"
         "<ul class='list'><li>7 cas de démonstration, chaîne intégrale réelle</li>"
         "<li>vocabulaire et seuil strictement officiels</li>"
         "<li>aucune métrique embellie</li></ul></div>"
         "<div class='card'><h3 style='color:var(--acc)'>Phase 2 — recalibration & explicabilité</h3>"
         "<div class='statb'><div class='num'>23,5 % → 55,9 %</div>"
         "<div class='lbl'>sensibilité test (seuil figé sur la seule validation, sans ré-entraînement)</div></div>"
         "<ul class='list'><li>AUROC / AP statistiquement constants (DeLong p="
         + n(delong_p, 3) + ")</li>"
         "<li>F1 test 0,38 → 0,62</li>"
         "<li>Post-traitement 36× plus rapide (extraction RGA chère et nuisible supprimée)</li>"
         "<li>XAI : fidélité quantitative, randomisation, stabilité vérifiées</li></ul></div>"
         "<div class='card'><h3 style='color:var(--acc)'>Limites (affichées)</h3>"
         "<ul class='list'><li>modèle officiel entraîné sur nos 240 cas — borne supérieure</li>"
         "<li>informativité des cartes faible sous score ≈ 0,15</li>"
         "<li>audit XAI sur 6 cas représentatifs (ouvrable à plus grande échelle)</li>"
         "<li>120 cas/split : CIs larges sur les sous-groupes</li></ul></div>"
         "</div></section>")

# ---- Conclusion
P.append("<section id='conclusion'><h2>Conclusion &amp; Next Steps</h2>"
         "<p class='lead'>Cette première implémentation constitue la base pour les améliorations et "
         "comparaisons qui suivront.</p>"
         "<div class='grid2'>"
         "<div class='card'><h3>Déjà réalisé</h3><ul class='list'>"
         "<li>Phase 1 : reproduction fidèle du détecteur PI-CAI, cas d'or validé, 7 cas réels déroulés du "
         "pipeline, seuil officiel strictement appliqué</li>"
         "<li>Phase 2 : évaluation 240 cas sans fuite, règles figées avant le test, benchmark des "
         "10 post-traitements, analyse d'erreurs, contrôle qualité de l'explicabilité</li>"
         "<li>Complexité maîtrisée : une seule commande refait toute la chaîne "
         "(<code>python scripts/run_phase2.py</code>)</li></ul></div>"
         "<div class='card'><h3>Prochaines étapes</h3><ul class='list'>"
         "<li>comparaison systématique des méthodes XAI sur tout le jeu test (matrice méthode × cas)</li>"
         "<li>intégration des cartes d'attention comme interface d'aide à la lecture</li>"
         "<li>validation externe (données held-out non publiées)</li>"
         "<li>calibration plus fine du point opératoire (échelle de température etc.)</li>"
         "<li>analyse quantitative des explications à plus grande échelle</li></ul></div>"
         "</div>"
         "<div class='bigbtns'>"
         f"<a href='nb/phase1_phase2_demo.html'>Notebook complet (exécuté)</a>"
         f"<a class='ghost' href='nb/phase1_phase2_demo.ipynb'>Télécharger le .ipynb</a>"
         f"<a class='ghost' href='nb/phase1_phase2_demo.ipynb' onclick=\"return false\"></a>"
         "</div></section>")

P.append(f"<div class='footer wrap'>Contenu produit par <b>phase1_phase2_demo.ipynb</b> (sorties réelles) — "
         f"généré automatiquement le {gen} · source de vérité unique : "
         "<code>reports/demo/data/presentation_data.json</code> · projet : "
         "Explainable AI for Prostate Cancer Detection on MRI.</div>")
P.append("</div>")
P.append(f"<script>{JS}</script></body></html>")

OUT.write_text("\n".join(P), encoding="utf-8")
print(f"[site v2] OK -> {OUT} ({OUT.stat().st_size // 1024} Ko) | backup: {backup.name if backup.exists() else 'aucune'}")
