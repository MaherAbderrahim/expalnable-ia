/* Diagnostic : recherche de chaînes anglaises résiduelles dans la page rendue. */
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
  const text = w.document.body.textContent.replace(/\s+/g, " ").toLowerCase();

  // Jetons anglais qui ne devraient plus apparaitre dans l'interface.
  const suspects = ["the ", " and ", " of ", " to ", " with ", " model ", " dataset ",
    "results ", " weights ", " paper ", " task ", " input ", " output ",
    "repository ", " available ", " for ", " this ", " that ", "from ",
    "grad-cam", "xai", "no ", "yes "];
  const found = suspects.filter(s => text.includes(s));
  console.log("JETONS ANGLAIS RESIDUELS :", JSON.stringify(found));

  // Contrôle positif : le contenu français est bien présent.
  ["modèles", "jeux de données", "articles", "sources", "poids", "licence",
    "tâche", "explicabilité", "classification", "segmentation"].forEach(f => {
    console.log("  contient « " + f + " » :", text.includes(f));
  });

  process.exit(0);
});
