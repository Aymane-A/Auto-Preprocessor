import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { EdaPanel, ExecutionPanel } from "./panels";
import type { Eda, Result } from "./api";

const common = { sessionId:"s", busy:false, result:null, onRun:()=>{}, testSize:20, onTestSize:()=>{}, onImbalance:()=>{} };
describe("imbalance controls", () => {
  it("defaults to no resampling and reveals relevant parameters", () => {
    const imbalance={counts:{"0":98,"1":2},minority_ratio:.02,imbalance_ratio:49,level:"sévère"};
    const base=renderToStaticMarkup(<ExecutionPanel {...common} imbalance={imbalance} options={{}} />);
    expect(base).toContain('Gestion du déséquilibre');expect(base).toContain('Aucun');
    expect(base).not.toContain('Voisins SMOTE');expect(base).not.toContain('SMOTENC');
    const chosen=renderToStaticMarkup(<ExecutionPanel {...common} imbalance={imbalance} options={{imbalance_method:"smote"}} rawCategorical />);
    expect(chosen).toContain('SMOTENC');expect(chosen).toContain('Voisins SMOTE');expect(chosen).toContain('Graine aléatoire');
    expect(chosen).toContain('Ratio minoritaire / majoritaire visé');
    const regression=renderToStaticMarkup(<ExecutionPanel {...common} options={{}} />);
    expect(regression).not.toContain('Gestion du déséquilibre');
  });
  it("shows measured class counts and an accuracy warning", () => {
    const eda={shape:[20000,2],features:[],missing_pct:0,duplicate_count:0,duplicate_examples:[],
      pearson:{columns:[],values:[],limit:40,eligible:0},cramers_v:{columns:[],values:[],limit:40,eligible:0},target_relations:[],relation_sample_size:100,
      imbalance:{counts:{"0":19800,"1":200},minority_ratio:.01,imbalance_ratio:99,level:"sévère"}} as Eda;
    const html=renderToStaticMarkup(<EdaPanel eda={eda} feature={null} busy={false} onFeature={()=>{}} />);
    expect(html).toContain('Cible déséquilibrée');expect(html).toContain('19');expect(html).toContain('PR-AUC');
    expect(html).toContain('width:99%');expect(html).toContain('width:1%');
    expect(html).toContain('aria-valuemax="20000" aria-valuenow="19800"');
    expect(html).toContain('aria-valuemax="20000" aria-valuenow="200"');
  });
  it("labels an ID-like feature as an identifier in the inventory", () => {
    const eda = { shape: [80, 1], features: [{ column: "record_id", dtype: "int64", type: "numérique",
      missing_pct: 0, missing_count: 0, unique: 80, top: [], detected_type: "ID_LIKE" }],
      missing_pct: 0, duplicate_count: 0, duplicate_examples: [],
      pearson: { columns: [], values: [], limit: 40, eligible: 0 },
      cramers_v: { columns: [], values: [], limit: 40, eligible: 0 },
      target_relations: [], relation_sample_size: 80 } as Eda;
    const html = renderToStaticMarkup(<EdaPanel eda={eda} feature={null} busy={false} onFeature={() => {}} />);
    expect(html).toContain("identifiant");
  });
  it("offers a separate provenance download without adding a feature", () => {
    const result={input_shape:[100,3],train_shape:[95,3],test_shape:[20,3],full_shape:[115,3],
      output_columns:["amount","is_fraud"],warnings:[],downloads:["train","test","full","train_provenance"],
      worker:{elapsed_seconds:1,peak_rss_bytes:1024}} as Result;
    const html=renderToStaticMarkup(<ExecutionPanel {...common} result={result} view="exports" onDownload={()=>{}} />);
    expect(html).toContain('Provenance du train CSV');
    expect(html).not.toContain('__synthetic__');
  });
  it("summarizes resampling counts in French without raw JSON", () => {
    const profile = (counts: Record<string, number>) => ({ counts, minority_ratio: .2, imbalance_ratio: 4, level: "modéré" });
    const result = { input_shape: [100, 2], train_shape: [80, 2], test_shape: [20, 2], full_shape: [100, 2],
      output_columns: ["amount", "target"], warnings: [], downloads: [], worker: { elapsed_seconds: 1, peak_rss_bytes: 1024 },
      imbalance: { method: "SMOTENC", before: profile({ "0": 60, "1": 20 }), after: profile({ "0": 60, "1": 30 }),
        test: profile({ "0": 16, "1": 4 }), synthetic_rows: 10, warnings: [], parameters: {} } } as Result;
    const html = renderToStaticMarkup(<ExecutionPanel {...common} result={result} view="exports" />);
    expect(html).toContain("Train avant : 0 : 60, 1 : 20");
    expect(html).toContain("test intact : 0 : 16, 1 : 4");
    expect(html).toContain("10 lignes synthétiques");
    expect(html).not.toContain("{&quot;");
  });
});
