/* Render the research site in jsdom and assert the DOM actually populates.
   Usage: node tools/render_test.js
*/
const fs = require("fs");
const path = require("path");
const { JSDOM } = require(require("path").join(process.env.TEMP || "/tmp", "opencode/domtest/node_modules/jsdom"));

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const dataJs = fs.readFileSync(path.join(root, "research-data.js"), "utf8");
const appJs = fs.readFileSync(path.join(root, "script.js"), "utf8");

const errors = [];
const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true });
const { window } = dom;
window.addEventListener("error", e => errors.push("window error: " + e.message));

/* Wait for the document to finish parsing before running the app script,
   so its DOMContentLoaded / readyState branch behaves as it does in a browser. */
function ready() {
  return new Promise(res => {
    if (window.document.readyState === "complete") return res();
    window.addEventListener("load", res);
    setTimeout(res, 3000);
  });
}

let d;
let D;
const q = s => d.querySelectorAll(s);
const text = s => (d.querySelector(s) ? d.querySelector(s).textContent.trim() : "<missing>");

let fails = 0;
function check(name, cond, extra) {
  if (cond) console.log("ok   " + name);
  else { console.log("FAIL " + name + (extra ? " — " + extra : "")); fails++; }
}

ready().then(() => {
  d = window.document;
  try {
    window.eval(dataJs);
    window.eval(appJs);
    D = window.RESEARCH_DATA;
  } catch (e) {

    errors.push("script threw: " + e.message + "\n" + (e.stack || "").split("\n").slice(0, 4).join("\n"));
  }
  run();
});

function run() {

check("no runtime errors", errors.length === 0, errors.join(" | "));

/* 1. Overview */
check("6 stat tiles", q("#stats .stat").length === 6, q("#stats .stat").length + " rendered");
check("stat values are numeric", /^\d+$/.test(text("#stats .stat b")));

/* 2. Recommended */
check("recommended model filled", text("#rec-model").length > 10);
check("recommended dataset filled", text("#rec-dataset").length > 10);
check("recommended XAI filled", text("#rec-xai").length > 10);
check("recommended pros listed", q("#rec-pros li").length >= 3);
check("recommended cons listed", q("#rec-cons li").length >= 3);
check("3 option cards", q("#options .opt-card").length === 3, q("#options .opt-card").length + "");
check("5 ranked models", q("#top-models .rank-row").length === 5, q("#top-models .rank-row").length + "");

/* 3. Models */
const modelCards = q("#models-grid .mcard");
check("model cards rendered", modelCards.length >= 20, modelCards.length + " cards");
check("model filter populated", q("#model-task option").length >= 5);
check("model count label set", /\d+\s+sur\s+\d+\s+modèles/.test(text("#model-count")), text("#model-count"));

/* 4. XAI */
const xaiCards = q("#xai-grid .xcard");
check("xai cards rendered", xaiCards.length === 14, xaiCards.length + " cards");
check("xai category filter populated", q("#xai-category option").length >= 4);
check("grad-cam card present", /Grad-CAM/.test(xaiCards[0].textContent));

/* 5. Datasets */
const dsRows = q("#datasets-table tbody tr");
check("dataset rows rendered", dsRows.length === D.datasets.length, dsRows.length + " rows");
check("dataset cards rendered", q("#dataset-cards .dcard").length === D.datasets.length);

/* 6. Papers */
const papers = q("#papers-list .paper");
check("paper cards rendered", papers.length === D.papers.length, papers.length + " papers");
check("paper filter populated", q("#paper-filter option").length === 5);

/* 7. Comparison */
const cmpRows = q("#cmp-table tbody tr");
check("comparison rows rendered", cmpRows.length >= 20, cmpRows.length + " rows");
check("comparison headers sortable", q("#cmp-table thead th[data-sort]").length === 10);

/* 8. Architecture */
check("7 pipeline steps", q("#pipeline .pipe-step").length === 7, q("#pipeline .pipe-step").length + "");

/* 9. Sources */
const srcs = q("#sources-list .src");
check("sources rendered", srcs.length >= 60, srcs.length + " sources");
check("source type filter populated", q("#src-type option").length >= 5);
check("verification block filled", text("#verification").length > 100);

/* Interaction smoke tests */
const search = d.querySelector("#model-search");
search.value = "nnunet";
search.dispatchEvent(new window.Event("input"));
check("model search filters", q("#models-grid .mcard").length < modelCards.length,
  q("#models-grid .mcard").length + " after filter");
search.value = "";
search.dispatchEvent(new window.Event("input"));

const sel = d.querySelector("#xai-category");
const firstRealCat = [...sel.options].map(o => o.value).find(Boolean);
sel.value = firstRealCat;
sel.dispatchEvent(new window.Event("change"));
check("xai category filter works", q("#xai-grid .xcard").length < xaiCards.length,
  q("#xai-grid .xcard").length + " after filter on '" + firstRealCat + "'");
sel.value = "";
sel.dispatchEvent(new window.Event("change"));

const th = d.querySelector('#cmp-table thead th[data-sort="model"]');
th.dispatchEvent(new window.Event("click"));
check("comparison sorts on header click", q("#cmp-table thead th.sorted").length === 1);

/* No stray placeholder text left.
   Em-dashes are legitimate prose punctuation here ("Limits — ...", the H1 title),
   so only genuinely broken markers are checked. "null" is checked as a standalone
   token, not a substring, because legitimate French words such as "nulle" and
   "annulé" contain it. */
const body = d.body.textContent;
["undefined", "NaN", "[object Object]"].forEach(bad => {
  const n = body.split(bad).length - 1;
  check("no stray '" + bad + "' in the rendered page", n === 0, n + " occurrence(s)");
});
const nullTokens = (body.match(/(^|[^A-Za-zÀ-ÿ])null([^A-Za-zÀ-ÿ]|$)/g) || []).length;
check("no stray standalone 'null' in the rendered page", nullTokens === 0, nullTokens + " occurrence(s)");

/* Every link button that is NOT disabled must carry a real http(s) href. */
const badHrefs = [...d.querySelectorAll("a.btn")].filter(a => {
  if (a.classList.contains("is-disabled")) return false;
  const h = a.getAttribute("href") || "";
  return !(h.startsWith("http://") || h.startsWith("https://") || h.startsWith("#"));
});
check("every enabled link button has a valid href", badHrefs.length === 0,
  badHrefs.length + " bad href(s): " + badHrefs.map(a => a.getAttribute("href")).join(", "));

console.log("\n" + (fails ? fails + " FAILURE(S)" : "ALL RENDER CHECKS PASSED"));
process.exit(fails ? 1 : 0);
}
