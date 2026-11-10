import { useState } from "react";
import type { Preview } from "./api";
import { DataTable, display, Distribution } from "./components";
import { AnimatedNumber } from "./animation";

/** Compare aligned observations and statistics returned by the preview worker. */
export function PreviewComparison({ preview, column }: { preview: Preview; column: string }) {
  const [output, setOutput] = useState("");
  const chart = preview.after_charts.find(item => item.title === output) ?? preview.after_charts[0];
  const outputs = preview.samples?.[0] ? Object.keys(preview.samples[0].values) : [];
  const outputCount = preview.output_columns?.length ?? preview.after_charts.length;
  const statistics = [
    ["mean", "Moyenne"], ["std", "Écart-type"], ["missing_pct", "Manquants (%)"], ["skew", "Asymétrie"],
  ].flatMap(([key, label]) => {
    const before = preview.before?.[key];
    const after = preview.after_scaled?.[key];
    return typeof before === "number" && typeof after === "number" ? [{ label, before, after }] : [];
  });
  return <div className="comparison">
    <div className="comparison-summary">
      <div><strong>{column}</strong><span>{display(preview.rows)} lignes dans l’aperçu · {outputCount} {outputCount === 1 ? "colonne" : "colonnes"} de sortie</span></div>
      <div className="operation-tags">{Object.entries(preview.applied ?? {}).map(([step, action]) =>
        <span key={step} title={step}>{step}<b>{display(action)}</b></span>)}</div>
    </div>
    {!!statistics.length && <div className="comparison-statistics" aria-label="Différences statistiques">
      {statistics.map(stat => <div key={stat.label}><span>{stat.label}</span>
        <p><span className="original-value"><AnimatedNumber value={stat.before} /></span><span aria-hidden="true">→</span><strong><AnimatedNumber value={stat.after} /></strong></p>
        <small>Écart {stat.after > stat.before ? "+" : ""}<AnimatedNumber value={stat.after - stat.before} /></small>
      </div>)}
    </div>}
    {preview.affected && <p className="comparison-impact">
      <strong>{display(preview.affected.missing_filled)}</strong> valeurs manquantes imputées
      <span>·</span><strong>{display(preview.affected.outside_bounds)}</strong> valeurs hors bornes
      <span>·</span><strong>{display(preview.affected.rows_changed)}</strong> lignes modifiées avant mise à l’échelle
    </p>}
    <div className="two-col comparison-charts">
      <div><h3>Avant</h3><p className="chart-caption">Distribution avant transformation</p><Distribution data={preview.before_chart} color="var(--beige)" /></div>
      <div>{preview.after_charts.length > 1 && <select aria-label="Sortie transformée" value={chart?.title ?? ""}
        onChange={event => setOutput(event.target.value)}>{preview.after_charts.map(item => <option key={item.title}>{item.title}</option>)}</select>}
        {chart ? <><h3>Après · {chart.title}</h3><p className="chart-caption">Sortie du traitement sélectionné</p><Distribution data={chart.data} /></>
          : <p className="empty-chart">Aucune sortie pour cette feature.</p>}
      </div>
    </div>
    {!!preview.samples?.length && <section className="comparison-values">
      <div className="panel-head"><h3>Valeurs originales et transformées</h3><span>Jusqu’à 12 lignes alignées · 6 sorties</span></div>
      <div className="table-scroll" tabIndex={0}><table>
        <thead><tr><th scope="col">Ligne source</th><th scope="col" className="original-value">Original · {column}</th>
          {outputs.map(name => <th scope="col" key={name}>Transformé · {name}</th>)}
          {!outputs.length && <th scope="col">Traitement</th>}</tr></thead>
        <tbody>{preview.samples.map((sample, index) => <tr key={index}><th scope="row">{display(sample.row)}</th>
          <td className="original-value">{display(sample.original)}</td>
          {outputs.map(name => <td className={sample.values[name] !== sample.original ? "changed-value" : ""} key={name}>{display(sample.values[name])}</td>)}
          {!outputs.length && <td>Variable retirée</td>}</tr>)}</tbody>
      </table></div>
    </section>}
    {preview.notes.map(note => <p className="comparison-note" key={note}>{note}</p>)}
    {!!preview.table.length && <details className="comparison-details"><summary>Toutes les statistiques et les effets du traitement</summary><DataTable rows={preview.table} /></details>}
  </div>;
}
