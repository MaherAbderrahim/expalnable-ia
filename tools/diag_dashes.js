/* Diagnostic: locate every em-dash in the rendered page and list disabled link buttons. */
const fs = require("fs");
const path = require("path");
const { JSDOM } = require(require("path").join(process.env.TEMP || "/tmp", "opencode/domtest/node_modules/jsdom"));

const root = path.resolve(__dirname, "..");
const dom = new JSDOM(fs.readFileSync(path.join(root, "index.html"), "utf8"),
  { runScripts: "outside-only", pretendToBeVisual: true });
const w = dom.window;

new Promise(r => { w.addEventListener("load", r); setTimeout(r, 2000); }).then(() => {
  w.eval(fs.readFileSync(path.join(root, "research-data.js"), "utf8"));
  w.eval(fs.readFileSync(path.join(root, "script.js"), "utf8"));
  const d = w.document;

  const hits = [];
  d.body.querySelectorAll("*").forEach(n => {
    if (n.children.length === 0 && n.textContent.includes("—")) {
      hits.push(n.tagName + "." + (n.className || "-") + ": " + n.textContent.trim().slice(0, 120));
    }
  });
  console.log("LEAF NODES CONTAINING AN EM-DASH: " + hits.length);
  hits.slice(0, 15).forEach(h => console.log("   " + h));

  const btns = [...d.querySelectorAll("a.btn.is-disabled")].map(a => a.textContent.trim());
  const uniq = [...new Set(btns)];
  console.log("\nDISABLED LINK BUTTONS: " + btns.length + " total, distinct labels: " + JSON.stringify(uniq));

  const emptyHrefs = [...d.querySelectorAll("a.btn")].filter(a => !a.getAttribute("href")).length;
  console.log("LINK BUTTONS WITH NO href: " + emptyHrefs);
  process.exit(0);
});
