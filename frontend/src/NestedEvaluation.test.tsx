import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, validateResponse } from "./api";
import type { Capabilities, NestedResult } from "./api";
import { buildNestedRequest, NestedResults } from "./NestedEvaluation";

const catalog: Capabilities = {
  defaults: {}, formats: [], options: { properties: { selection_method: { title: "Sélection", enum: ["auto", "kbest", "chi2", "rfecv", "percentile"] } } },
  nested_evaluation: { properties: {
    family: { title: "Modèle", enum: ["linear", "logistic", "random_forest"] },
    outer_folds: { title: "Plis externes", minimum: 2, maximum: 5 },
    inner_folds: { title: "Plis internes", minimum: 2, maximum: 5 },
    scalers: { title: "Échelles", items: { enum: ["none", "standard", "minmax"] } },
  } },
};
const draft = { sessionId: "s", task: "classification", options: { random_state: 7 }, family: "logistic",
  outerFolds: 2, innerFolds: 3, split: "stratified", splitColumn: "", method: "kbest",
  counts: "2, 4", scalers: ["none", "minmax"], modelParams: '[{"C": 0.1}, {"C": 1}]',
  percentile: 50, selectorFolds: 2 };
const columns = ["signal", "category", "group", "when"];

afterEach(() => vi.unstubAllGlobals());

describe("Nested CV contract", () => {
  it("builds exact binary, group, time and regression requests", () => {
    const binary = buildNestedRequest(draft, catalog, columns);
    expect(binary).toEqual({ session_id: "s", options: { random_state: 7, selection_method: "kbest" },
      family: "logistic", outer_folds: 2, inner_folds: 3, time_column: null, group_column: null,
      selection_counts: [2, 4], scalers: ["none", "minmax"], model_params: [{ C: .1 }, { C: 1 }] });
    expect(buildNestedRequest({ ...draft, split: "group", splitColumn: "group" }, catalog, columns).group_column).toBe("group");
    expect(buildNestedRequest({ ...draft, split: "time", splitColumn: "when" }, catalog, columns).time_column).toBe("when");
    const regression = buildNestedRequest({ ...draft, task: "regression", family: "linear", split: "kfold",
      method: "percentile", counts: "", scalers: [], modelParams: "[{}]", percentile: 30 }, catalog, columns);
    expect(regression.options.selection_percentile).toBe(30);
    expect(regression.selection_counts).toEqual([]);
  });

  it("rejects invalid and unsupported configurations before sending them", () => {
    expect(() => buildNestedRequest({ ...draft, task: "auto" }, catalog, columns)).toThrow("explicitement");
    expect(() => buildNestedRequest({ ...draft, task: "regression" }, catalog, columns)).toThrow("modèle");
    expect(() => buildNestedRequest({ ...draft, outerFolds: 6 }, catalog, columns)).toThrow("plis");
    expect(() => buildNestedRequest({ ...draft, split: "unknown" }, catalog, columns)).toThrow("Stratégie");
    expect(() => buildNestedRequest({ ...draft, split: "group", splitColumn: "unknown" }, catalog, columns)).toThrow("colonne");
    expect(() => buildNestedRequest({ ...draft, counts: "2,4,6,8,10" }, catalog, columns)).toThrow("quatre");
    expect(() => buildNestedRequest({ ...draft, modelParams: '[{"bad": []}]' }, catalog, columns)).toThrow("scalaires");
    expect(() => buildNestedRequest({ ...draft, modelParams: "oops" }, catalog, columns)).toThrow("JSON");
    expect(() => buildNestedRequest({ ...draft, counts: "2,4,6,8", scalers: ["none", "standard", "minmax"], modelParams: '[{}, {}]' }, catalog, columns)).toThrow("16");
    expect(() => buildNestedRequest({ ...draft, method: "chi2", counts: "", task: "regression", family: "linear", split: "kfold" }, catalog, columns)).toThrow("classification");
  });

  it("posts to the nested endpoint and refuses a fabricated successful result", async () => {
    const request = buildNestedRequest(draft, catalog, columns);
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ protocol: "nested_cross_validation" })));
    vi.stubGlobal("fetch", fetch);
    await expect(api("/evaluate-nested", request)).rejects.toMatchObject({ status: 502 });
    expect(fetch.mock.calls[0][0]).toContain("/api/evaluate-nested");
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual(request);
  });

  it("surfaces a backend configuration error for the nested endpoint", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Classe absente du train." }), { status: 422 })));
    await expect(api("/evaluate-nested", buildNestedRequest(draft, catalog, columns)))
      .rejects.toMatchObject({ status: 422, message: "Classe absente du train." });
  });
});

const result: NestedResult = {
  protocol: "nested_cross_validation", task: "classification", split_mode: "stratified",
  scoring_inner: "balanced_accuracy", classes: [0, 1],
  predicted_row_positions: [0, 1, 2, 3], unscored_row_positions: [],
  global_metrics: { accuracy: .75, confusion_matrix: [[2, 0], [1, 1]] },
  outer_summary: { accuracy: { mean: .75, std: .35, folds: 2 } },
  outer_folds: [1, 2].map((fold, index) => ({ fold, train_row_positions: index ? [0, 1] : [2, 3],
    validation_row_positions: index ? [2, 3] : [0, 1], predictions: [0, 1], targets: [0, 1],
    metrics: { accuracy: index ? .5 : 1, confusion_matrix: [[1, 0], [0, 1]] },
    selected_config: { preprocessing: { selection_count: 2 }, model: { C: 1 } }, selected_features: ["signal"],
    inner_search: [{ preprocessing: { selection_count: 2 }, model: { C: 1 }, scores: [.6, .8], mean: .7 }],
    metric_warnings: [], elapsed_seconds: 1.2, pipeline_fit_count: 3 })),
};

describe("Nested CV results", () => {
  it("separates inner search from external metrics, per-fold values and confusion", () => {
    expect(() => validateResponse("/evaluate-nested", result)).not.toThrow();
    const html = renderToStaticMarkup(<NestedResults result={result} selectionMethod="kbest" family="logistic" />);
    expect(html).toContain("Résultats des plis externes");
    expect(html).toContain("moyennes et écarts-types sont descriptifs");
    expect(html).toContain("plis répétés partagent des observations");
    expect(html).toContain("Recherche interne · pli externe 1");
    expect(html).toContain("Moyenne et écart-type");
    expect(html).toContain("Matrice de confusion globale");
    expect(html).toContain("Configuration retenue");
    expect(html).toContain("4 observations évaluées");
    expect(html).toContain("logistic");
    expect(html).not.toContain("données fictives");
    const protocol = renderToStaticMarkup(<NestedResults result={result} selectionMethod="rfecv" family="logistic"
      request={{ ...buildNestedRequest({ ...draft, method: "rfecv", counts: "" }, catalog, columns), outer_folds: 5,
        options: { selection_method: "rfecv", selection_folds: 2 } }} />);
    expect(protocol).toContain("5 plis externes ; 3 plis internes ; 2 plis du sélecteur enveloppe");
  });

  it("shows unavailable temporal positions and missing-class warnings without inventing scores", () => {
    const temporal = { ...result, split_mode: "time", predicted_row_positions: [2, 3], unscored_row_positions: [0, 1],
      global_metrics: { accuracy: .5 }, outer_folds: [{ ...result.outer_folds[0],
        metric_warnings: ["Validation sans toutes les classes"] }] };
    expect(() => validateResponse("/evaluate-nested", temporal)).not.toThrow();
    const html = renderToStaticMarkup(<NestedResults result={temporal} selectionMethod="variance" family="linear" />);
    expect(html).toContain("2 observations évaluées");
    expect(html).toContain("2 non évaluées");
    expect(html).toContain("ROC-AUC et PR-AUC indisponibles");
    expect(html).toContain("Validation sans toutes les classes");
    expect(renderToStaticMarkup(<NestedResults result={temporal} selectionMethod="variance" family="linear" />))
      .toContain("aucune performance de prévision opérationnelle n’est validée");
    expect(() => validateResponse("/evaluate-nested", { ...result, predicted_row_positions: [0, 0] })).toThrow();
  });

  it("renders multiclass macro F1 and a three-class confusion matrix", () => {
    const multiclass: NestedResult = { ...result, classes: [0, 1, 2],
      global_metrics: { accuracy: .75, f1_macro: .7, f1_weighted: .72,
        balanced_accuracy: .68, confusion_matrix: [[2, 0, 0], [0, 1, 1], [0, 0, 1]] } };
    const html = renderToStaticMarkup(<NestedResults result={multiclass} selectionMethod="kbest" family="logistic" />);
    expect(html).toContain("F1 macro");
    expect(html).toContain("F1 pondéré");
    expect(html).toContain("Prédit 2");
  });

  it("renders regression metrics without classification claims", () => {
    const regression: NestedResult = { ...result, task: "regression", classes: null,
      global_metrics: { mae: 12, rmse: 15, r2: .4 },
      outer_summary: { mae: { mean: 12, std: 3, folds: 2 } },
      outer_folds: result.outer_folds.map(fold => ({ ...fold, metrics: { mae: 12, rmse: 15, r2: .4 } })) };
    const html = renderToStaticMarkup(<NestedResults result={regression} selectionMethod="variance" family="linear" />);
    expect(html).toContain("RMSE");
    expect(html).toContain("MAE");
    expect(html).toContain("R²");
    expect(html).not.toContain("Matrice de confusion globale");
  });
});
