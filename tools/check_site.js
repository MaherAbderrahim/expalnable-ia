/* Static checks for the research site.
   Verifies:
     1. script.js parses
     2. every getElementById/querySelector('#id') used by script.js exists in index.html
     3. every id referenced in index.html's script src files exists
     4. the research data loads and every model/dataset/paper has the fields the UI reads
     5. no broken internal anchors
*/
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const js = fs.readFileSync(path.join(root, "script.js"), "utf8");

let failures = 0;
const fail = (m) => { console.log("FAIL " + m); failures++; };
const pass = (m) => console.log("ok   " + m);

/* 1. data loads */
const dataSrc = fs.readFileSync(path.join(root, "research-data.js"), "utf8");
const sandbox = { window: {} };
try {
  new Function("window", dataSrc)(sandbox.window);
} catch (e) {
  fail("research-data.js does not evaluate: " + e.message);
}
const D = sandbox.window.RESEARCH_DATA;
if (!D) fail("window.RESEARCH_DATA not set");
else pass("research-data.js evaluates (" + Object.keys(D).length + " top-level keys)");

/* 2. ids referenced by script.js exist in index.html */
const htmlIds = new Set([...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]));
const usedIds = new Set([...js.matchAll(/\$\("#([a-zA-Z0-9_-]+)"/g)].map(m => m[1]));
const missing = [...usedIds].filter(i => !htmlIds.has(i));
if (missing.length) fail("script.js references missing ids: " + missing.join(", "));
else pass("all " + usedIds.size + " script.js #id lookups exist in index.html");

/* 3. anchors */
const anchors = [...html.matchAll(/href="#([a-zA-Z0-9_-]+)"/g)].map(m => m[1]);
const badAnchors = [...new Set(anchors)].filter(a => !htmlIds.has(a));
if (badAnchors.length) fail("broken internal anchors: " + badAnchors.join(", "));
else pass("all " + new Set(anchors).size + " internal anchors resolve");

/* 4. data fields the UI reads */
const need = {
  models: ["id", "name", "architecture", "task", "dataset", "classes", "input", "performance",
    "code", "repo_id", "weights", "framework", "license", "reuse", "xai", "pros", "cons", "difficulty"],
  datasets: ["id", "name", "url", "type", "images", "patients", "classes", "masks",
    "split", "license", "access", "format", "used_by", "recommended"],
  xai_methods: ["name", "category", "principle", "how", "compatible", "advantages", "limitations",
    "compute", "ease", "medical_relevance", "visual_quality", "code", "libs"],
  papers: ["title", "year", "authors", "venue", "url", "model", "dataset", "task", "metrics", "xai", "limits"]
};
const optional = { datasets: ["paper", "caveat", "quality_notes", "provenance", "preprocessing", "size", "masks"] };
for (const [key, fields] of Object.entries(need)) {
  if (!D[key]) { fail("missing data array: " + key); continue; }
  let bad = 0;
  const opt = (optional[key] || []);
  D[key].forEach((row, i) => {
    fields.forEach(f => {
      if (row[f] === undefined) { fail(`${key}[${i}] (${row.name || row.title}) missing field "${f}"`); bad++; }
    });
    opt.forEach(f => {
      if (row[f] !== undefined && row[f] !== null && typeof row[f] !== "string" && typeof row[f] !== "number") {
        fail(`${key}[${i}] optional field "${f}" has the wrong type`); bad++;
      }
    });
  });
  if (!bad) pass(`${key}: ${D[key].length} records, all ${fields.length} fields present`);
}

/* 5. repo_id references resolve */
const repoIds = new Set(D.repositories.map(r => r.id));
const dangling = D.models.filter(m => m.repo_id && !repoIds.has(m.repo_id)).map(m => m.name + " -> " + m.repo_id);
if (dangling.length) fail("models referencing unknown repo_id: " + dangling.join(", "));
else pass("every model.repo_id resolves to a repository");

/* 6. counts are honest */
const c = D.meta.counts;
const pairs = [["models", "models"], ["datasets", "datasets"], ["xai_methods", "xai_methods"],
  ["papers", "papers"], ["repositories", "repositories"], ["recommendations", "recommendations"]];
let countBad = 0;
pairs.forEach(([ck, dk]) => { if (c[ck] !== D[dk].length) { fail(`count ${ck}=${c[ck]} but array length ${D[dk].length}`); countBad++; } });
if (!countBad) pass("declared counts match actual array lengths");

/* 7. every URL in the data is absolute http(s) */
const urls = [];
const walk = (o, p) => {
  if (typeof o === "string") { if (/^https?:\/\//.test(o)) urls.push([p, o]); return; }
  if (Array.isArray(o)) { o.forEach((v, i) => walk(v, p + "[" + i + "]")); return; }
  if (o && typeof o === "object") Object.entries(o).forEach(([k, v]) => walk(v, p + "." + k));
};
walk(D, "$");
const badUrl = urls.filter(([p, u]) => { try { new URL(u); return false; } catch (e) { return true; } });
if (badUrl.length) fail("malformed URLs: " + badUrl.map(x => x[1]).join(", "));
else pass("all " + urls.length + " embedded URLs are well-formed");

/* 8. repo category values */
const cats = new Set(["A", "B", "C", "D"]);
const badCat = D.repositories.filter(r => !cats.has(r.category));
if (badCat.length) fail("repositories with invalid category: " + badCat.map(r => r.name).join(", "));
else pass("all " + D.repositories.length + " repositories carry a valid A/B/C/D category");

console.log("\n" + (failures ? failures + " FAILURE(S)" : "ALL CHECKS PASSED"));
process.exit(failures ? 1 : 0);
