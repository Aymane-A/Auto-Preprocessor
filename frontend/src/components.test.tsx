/** Test key presentation behavior without requiring a browser process. */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Boxplot, DataTable, display, Distribution, effectiveChoice, Heatmap, visibleFeatures } from "./components";
import type { Feature } from "./api";

const features: Feature[] = [
  { column: "Catégorie", type: "catégoriel", dtype: "string", missing_pct: 0, missing_count: 0, unique: 3, top: [] },
  { column: "Montant", type: "numérique", dtype: "float64", missing_pct: 20, missing_count: 2, unique: 8, top: [], mean: 5 },
  { column: "Durée", type: "numérique", dtype: "float64", missing_pct: 10, missing_count: 1, unique: 9, top: [], mean: 9 },
];

describe("inventaire", () => {
  it("distingue une petite probabilité, zéro et une statistique indisponible", () => {
    expect(display(1e-90)).toBe("1.000e-90");
    expect(display(0)).toBe("0");
    expect(display(null)).toBe("—");
    expect(display(Number.NaN)).toBe("indisponible");
    expect(display("< 0.001")).toBe("< 0.001");
  });
  it("filtre par type et trie sans modifier l’ordre initial", () => {
    expect(visibleFeatures(features, "NUMÉRIQUE", "missing_pct", true).map(x => x.column)).toEqual(["Montant", "Durée"]);
    expect(visibleFeatures(features, "", "mean", true).map(x => x.column)).toEqual(["Durée", "Montant", "Catégorie"]);
    expect(features[0].column).toBe("Catégorie");
  });

  it("rend les manquants et échappe les cellules", () => {
    const view = renderToStaticMarkup(<DataTable rows={[{ a: "<test>", b: null }]} />);
    expect(view).toContain("&lt;test&gt;");
    expect(view).toContain("—");
  });

  it("rend les colonnes multilingues, les booléens et les statistiques indisponibles", () => {
    const view = renderToStaticMarkup(<DataTable rows={[{ "年齢 / âge": null, "booléen": true, "بلد": "Maroc" }]} />);
    expect(view).toContain("年齢 / âge");
    expect(view).toContain("بلد");
    expect(view).toContain("Oui");
    expect(view).toContain("Maroc");
    expect(visibleFeatures([{ ...features[1], mean: null }], "", "mean", true)[0].mean).toBeNull();
  });
});

describe("prétraitement et graphiques", () => {
  it("affiche des états vides sans inventer de valeurs", () => {
    expect(renderToStaticMarkup(<Distribution data={[]} />)).toContain("Aucune valeur à tracer.");
    expect(renderToStaticMarkup(<Heatmap title="Pearson" matrix={{ columns: [], values: [], eligible: 0, limit: 40 }} />)).toContain("Aucune colonne éligible.");
    expect(renderToStaticMarkup(<DataTable rows={[]} />)).toContain("Aucune ligne à afficher.");
  });
  it("applique la recommandation par défaut, y compris pour les noms réservés", () => {
    expect(effectiveChoice({}, "constructor")).toBe("recommended");
    expect(effectiveChoice({ montant: "none" }, "montant")).toBe("none");
  });

  it("rend les valeurs de la matrice et les quartiles accessibles", () => {
    const matrix = renderToStaticMarkup(<Heatmap title="Pearson" matrix={{ columns: ["a"], values: [[1]], limit: 40, eligible: 1 }} />);
    expect(matrix).toContain('title="a × a · 1"');
    expect(renderToStaticMarkup(<Boxplot box={{ low: 0, q1: 1, median: 2, q3: 3, high: 4 }} />)).toContain("Médiane 2");
  });
});
