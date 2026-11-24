/** Render only chart descriptors returned by the backend feature profiler. */
import type { Feature, Visualization } from "./api";
import { Boxplot, DataTable, display, Distribution, percentDisplay, TimeSeries } from "./components";

const titles: Record<Visualization["kind"], string> = {
  binary_bar: "Deux modalités", frequency_bar: "Fréquences par modalité", ordered_bar: "Modalités dans l’ordre naturel", top_categories: "Modalités principales et autres",
  value_counts: "Effectifs par valeur", dominant_share: "Part de la valeur dominante", missingness: "Valeurs manquantes et présentes",
  histogram: "Histogramme", boxplot: "Boîte à moustaches", time_series: "Observations par mois", summary: "Cardinalité et exemples",
  target_boxplot: "Distribution par classe cible", target_distribution: "Effectifs discrets par classe cible", target_rate: "Taux par catégorie",
};

export function FeatureVisualization({ feature }: { feature: Feature }) {
  const charts = feature.recommended_visualization;
  return <div data-feature-visualization={feature.column}>
    <p className="muted">{feature.display_type_override && feature.display_type_override !== "AUTO" ? "Type affiché (forcé)" : "Type détecté"} : {feature.detected_type ?? "indisponible"} · {display(feature.unique)} valeurs distinctes</p>
    <p className="muted">{display(feature.missing_count)} valeurs manquantes ({percentDisplay(feature.missing_pct)} %)</p>
    {!charts?.length && <p>Aucun choix de visualisation fourni par le profilage backend.</p>}
    {charts?.map((chart, index) => <section key={`${chart.kind}-${index}`} data-chart-kind={chart.kind} aria-label={titles[chart.kind]}>
      <h3>{titles[chart.kind]}</h3><p className="muted">{chart.reason}</p>
      {chart.kind === "summary" ? <>
        <p>Cardinalité : {display(chart.cardinality)} · Exemples : {chart.examples?.join(" · ") || "aucune valeur observée"}</p>
      </> : chart.kind === "boxplot" && chart.box ? <>
        <Boxplot box={chart.box} /><p>{display(chart.outlier_count)} extrêmes IQR sur {display(chart.sample_size)} valeurs observées</p>
        <DataTable rows={[
          { Méthode: "IQR × 1,5", Extrêmes: feature.iqr?.count ?? null, "Borne basse": feature.iqr?.lower ?? null, "Borne haute": feature.iqr?.upper ?? null },
          { Méthode: "Score z robuste · 3,5 MAD", Extrêmes: feature.robust_z?.count ?? null, "Borne basse": feature.robust_z?.lower ?? null, "Borne haute": feature.robust_z?.upper ?? null },
        ]} />
      </> : chart.kind === "target_boxplot" || chart.kind === "target_distribution" ? chart.groups?.map(group => <div key={group.label}>
        <h4>Classe {group.label} · {display(group.sample_size)} observations</h4>
        {group.box ? <Boxplot box={group.box} domain={chart.domain} /> : group.data ? <>
          <Distribution data={group.data} horizontal={chart.horizontal} rate unit="Part de la classe (%)" />
          <details><summary>Effectifs et pourcentages de la classe {group.label}</summary><DataTable rows={group.data.map(row => ({ Valeur: row.label, Effectif: row.count, "%": row.percent ?? null }))} /></details>
        </> : <p>Données par classe indisponibles.</p>}
      </div>) : chart.data ? <>
        {chart.kind === "time_series" ? <TimeSeries data={chart.data} /> :
          <Distribution data={chart.data} horizontal={chart.horizontal} histogram={chart.kind === "histogram"} rate={chart.kind === "target_rate"} unit={chart.kind === "target_rate" ? "Taux (%)" : "Effectif"} />}
        {chart.data.length <= 2 && <p>{chart.data.map(row => `${row.label} : ${display(row.count)} (${percentDisplay(row.percent)} %)` ).join(" · ")}</p>}
        <details><summary>{chart.kind === "target_rate" ? "Effectifs et taux par catégorie" : "Effectifs et pourcentages"}</summary>
          <DataTable rows={chart.data.map(row => ({ Valeur: row.label, Effectif: row.count, ...(row.total === undefined ? {} : { Dénominateur: row.total }), "%": row.percent ?? null }))} />
        </details>
      </> : <p>Données de visualisation indisponibles.</p>}
    </section>)}
    {feature.text_length && <p>Longueur moyenne des valeurs {display(feature.text_length.mean)} caractères</p>}
    {feature.relationship_note && <p className="muted">{feature.relationship_note}</p>}
  </div>;
}
