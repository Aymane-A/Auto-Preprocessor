import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Feature } from "./api";
import { validateResponse } from "./api";
import { FeatureVisualization } from "./FeatureVisualization";
import { EdaPanel } from "./panels";
import passengerId from "./__fixtures__/passengerId.json";
import type { Eda } from "./api";

const base: Feature = { column: "flag", type: "numérique", dtype: "int64", missing_count: 0, missing_pct: 0,
  unique: 2, top: [], detected_type: "BINARY", target_type: "NONE", relationship_note: "Aucune cible sélectionnée.",
  recommended_visualization: [{kind:"binary_bar", reason:"2 modalités", data:[{label:"0",count:60,percent:60},{label:"1",count:40,percent:40}]}] };

describe("backend-selected feature visualization", () => {
  it("accepts the exact Titanic ID payload and renders its summary", () => {
    expect(() => validateResponse("/eda/session/feature/PassengerId", passengerId)).not.toThrow();
    const html = renderToStaticMarkup(<FeatureVisualization feature={passengerId as Feature} />);
    expect(html).toContain('data-chart-kind="summary"');
    expect(html).toContain("Cardinalité : 891");
    expect(html).not.toContain("Réponse du serveur incompatible");
  });
  it("keeps the EDA page and shows a local message for an invalid column", () => {
    const invalid = { ...passengerId, recommended_visualization: [{ kind: "invalid", reason: "incorrect" }] };
    expect(() => validateResponse("/eda/session/feature/PassengerId", invalid)).toThrow();
    const matrix = { columns: [], values: [], limit: 0, eligible: 0 };
    const eda = { shape: [891, 12], features: [], missing_pct: 0, duplicate_count: 0, duplicate_examples: [],
      pearson: matrix, cramers_v: matrix, relation_sample_size: 0, target_relations: [] } as unknown as Eda;
    const html = renderToStaticMarkup(<EdaPanel eda={eda} feature={null} featureError="Visualisation indisponible pour cette colonne : PassengerId." busy={false} onFeature={() => {}} />);
    expect(html).toContain("Visualisation indisponible pour cette colonne : PassengerId.");
    expect(html).toContain("Inventaire des features");
    expect(html).not.toContain("Réponse du serveur incompatible");
  });
  it("renders the selected binary chart even if legacy boxplot statistics exist", () => {
    const html=renderToStaticMarkup(<FeatureVisualization feature={{...base,boxplot:{low:0,q1:0,median:0,q3:1,high:1}}} />);
    expect(html).toContain('data-chart-kind="binary_bar"');
    expect(html).not.toContain('class="boxplot"');expect(html).toContain('60,0 %');expect(html).toContain('40,0 %');
  });
  it("renders an explicit summary and never invents a distribution", () => {
    const html=renderToStaticMarkup(<FeatureVisualization feature={{...base,detected_type:"ID_LIKE",recommended_visualization:[{kind:"summary",reason:"Aucune distribution pertinente",cardinality:80,examples:["R001","R002"]}]}} />);
    expect(html).toContain('Cardinalité : 80');expect(html).toContain('R001');expect(html).not.toContain('class="chart"');
  });
  it("does not guess chart kinds when the backend omitted its selection", () => {
    const html=renderToStaticMarkup(<FeatureVisualization feature={{...base,recommended_visualization:undefined}} />);
    expect(html).toContain('Aucun choix de visualisation');expect(html).not.toContain('data-chart-kind');
    expect(()=>validateResponse('/eda/session/feature/flag',{...base,distribution:[],recommended_visualization:[{kind:"made_up",reason:"bad"}]})).toThrow();
    expect(()=>validateResponse('/eda/session/feature/flag',{...base,distribution:[]})).not.toThrow();
  });
  it("keeps counts available for grouped small-sample target rates", () => {
    const html=renderToStaticMarkup(<FeatureVisualization feature={{...base,recommended_visualization:[{
      kind:"target_rate",reason:"Modalités d'effectif < 5 regroupées",data:[{label:"Autres",count:1,total:2,percent:50,is_other:true}]
    }]}} />);
    expect(html).toContain('Effectifs et taux par catégorie');
    expect(html).toContain('Dénominateur');expect(html).toContain('Autres');
  });
  it("renders the backend's calendar order and labels an existing type override", () => {
    const html=renderToStaticMarkup(<FeatureVisualization feature={{...base,detected_type:"ORDINAL",display_type_override:"FORCE_CATEGORICAL",recommended_visualization:[{
      kind:"ordered_bar",reason:"janvier → décembre",data:[{label:"January",count:10,percent:25},{label:"December",count:30,percent:75}]
    }]}} />);
    expect(html).toContain('data-chart-kind="ordered_bar"');expect(html.indexOf('January')).toBeLessThan(html.indexOf('December'));
    expect(html).toContain('Type affiché (forcé)');
  });
});
