import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RecommendationExplanation } from "./RecommendationExplanation";
import type { Recommendation } from "./api";

const recommendation: Recommendation = {
  column: "record_id", type: "DROP_ID", is_target: false, anomalies: ["Identifiant potentiel"],
  action: "DROP", label: "Supprimer", justification: {
    raison: "Identifiant détecté sur record_id.", preuve: "uniques = 80 / 80 valeurs observées (100%)",
    explication: "Cette valeur reconnaît chaque ligne et risque de ne pas généraliser."
  },
  alternatives: [{ value: "none", label: "Conserver brute" }],
  steps: { type: { action: "DROP", measure: "80 valeurs uniques", threshold: "ratio > 90%",
    explanation: "La colonne est retirée avant l'encodage.", confidence: "élevée",
    comparison: { none: { avertissement: "L'identifiant peut surajuster." } } } },
  diagnostics: { uniques: 80, manquants_pct: 0, ratio_uniques_lignes: 1, exemples: ["1", "2"] },
};

describe("explanations de recommandation", () => {
  it("utilise les preuves et exemples de la variable sans inventer de sortie", () => {
    const html = renderToStaticMarkup(<RecommendationExplanation row={recommendation} choice="recommended" />);
    expect(html).toContain("Pourquoi cette recommandation ?");
    expect(html).toContain("80 valeurs uniques");
    expect(html).toContain("Valeurs manquantes (%) : 0");
    expect(html).toContain("1 · 2");
    expect(html).toContain("valeur après transformation est disponible dans l’aperçu");
    expect(html).toContain("traitement non encore exécuté");
  });

  it("distingue le choix final de la recommandation", () => {
    const html = renderToStaticMarkup(<RecommendationExplanation row={recommendation} choice="none" />);
    expect(html).toContain("Conserver la colonne brute");
    expect(html).toContain("Identifiant détecté sur record_id");
  });
});
