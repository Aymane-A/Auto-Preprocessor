import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { AppliedReport } from "./PipelinePanel";
import { actionLabel, operationLabel } from "./labels";
import { ImportPanel, suggestedTask } from "./panels";
import type { TransformationReport, Upload } from "./api";

const dataset: Upload = {
  session_id: "sample", filename: "train.csv", shape: [20, 3], columns: ["PassengerId", "Survived", "Fare"],
  dtypes: { PassengerId: "int64", Survived: "int64", Fare: "float64" },
  preview: Array.from({ length: 20 }, (_, index) => ({ PassengerId: index + 1, Survived: index % 2, Fare: index + .25 })),
};

describe("Task 8 presentation", () => {
  it("starts without an ID target and infers an editable task on selection", () => {
    const html = renderToStaticMarkup(<ImportPanel dataset={dataset} config={{ mode: "supervised", target: "", task: "auto" }} busy={false}
      onFile={() => {}} onChange={() => {}} onConfigure={() => {}} />);
    expect(html).toContain("Choisir la cible");
    expect(html).toContain('<option value="" selected="">Choisir la cible</option>');
    expect(html).not.toContain('<option selected="">PassengerId</option>');
    expect(suggestedTask(dataset, "Survived")).toBe("classification");
    expect(suggestedTask(dataset, "Fare")).toBe("regression");
  });

  it("uses French operation labels and readable imbalance text", () => {
    expect(operationLabel("clean_missing_tokens")).toBe("Faux manquants");
    expect(actionLabel("TO_NAN")).toBe("Convertir en valeur manquante");
    const report: TransformationReport = { summary: { "Gestion du déséquilibre": {
      method: "none", before: { counts: { "0": 12, "1": 8 } }, test: { counts: { "0": 3, "1": 2 } }, synthetic_rows: 0,
    } }, features: [] };
    const html = renderToStaticMarkup(<AppliedReport report={report} />);
    expect(html).toContain("Aucun rééquilibrage");
    expect(html).toContain("test intact");
    expect(html).toContain("<td>Aucun rééquilibrage · train");
  });
});
