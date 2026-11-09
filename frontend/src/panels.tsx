/** French workflow panels for upload, exploration, choices and export. */
import { useState } from "react";
import type { Bin, Eda, Feature, Preview, Recommendation, Result, Upload } from "./api";
import { apiRoot } from "./api";
import { Boxplot, DataTable, display, Distribution, effectiveChoice, Heatmap, Metric, shape, visibleFeatures } from "./components";
import type { SortKey } from "./components";

export type Configuration = { mode: "supervised" | "unsupervised"; target: string; task: string };

/** Collect a local dataset and its learning configuration. */
export function ImportPanel({ dataset, config, busy, onFile, onChange, onConfigure }: {
  dataset: Upload | null; config: Configuration; busy: boolean; onFile: (file: File) => void;
  onChange: (next: Configuration) => void; onConfigure: () => void;
}) {
  return <>
    <div className="two-col import-grid">
      <section className="panel">
        <h2>Fichier source</h2>
        <label className="upload">
          <svg className="upload-icon" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M12 16V3m-5 5 5-5 5 5M4 16v4h16v-4" /></svg>
          <strong>{dataset?.filename ?? "Choisir un fichier"}</strong>
          <span>{dataset ? shape(dataset.shape) + " / remplacer le fichier" : "CSV, TSV, TXT, Excel, JSON ou Parquet"}</span>
          <input aria-label="Fichier source" type="file" accept=".csv,.tsv,.txt,.xlsx,.xls,.json,.parquet"
            disabled={busy} onChange={event => {
              const file = event.currentTarget.files?.[0];
              event.currentTarget.value = "";
              if (file) onFile(file);
            }} />
        </label>
      </section>
      <section className="panel">
        <h2>Apprentissage</h2>
        <form onSubmit={event => { event.preventDefault(); onConfigure(); }}>
          <fieldset disabled={!dataset || busy}>
            <legend className="sr-only">Configuration de l’apprentissage</legend>
            <div className="segment">
              <label><input type="radio" name="mode" checked={config.mode === "supervised"}
                onChange={() => onChange({ ...config, mode: "supervised" })} /> Supervisé</label>
              <label><input type="radio" name="mode" checked={config.mode === "unsupervised"}
                onChange={() => onChange({ ...config, mode: "unsupervised" })} /> Non supervisé</label>
            </div>
            {config.mode === "supervised" && <>
              <label className="field">Colonne cible
                <select required value={config.target} onChange={event => onChange({ ...config, target: event.target.value })}>
                  <option value="">Choisir une colonne</option>
                  {dataset?.columns.map(name => <option key={name}>{name}</option>)}
                </select>
              </label>
              <label className="field">Type de tâche
                <select value={config.task} onChange={event => onChange({ ...config, task: event.target.value })}>
                  <option value="auto">Détection automatique</option><option value="classification">Classification</option>
                  <option value="regression">Régression</option>
                </select>
              </label>
            </>}
            <button className="primary" type="submit">Valider et explorer</button>
          </fieldset>
        </form>
      </section>
    </div>
    {dataset && <section className="panel">
      <h2>Aperçu du fichier</h2><p className="muted">{dataset.preview.length} premières lignes</p>
      <DataTable rows={dataset.preview} columns={dataset.columns} />
    </section>}
  </>;
}

const statsColumns: [SortKey, string][] = [
  ["column", "Feature"], ["type", "Type"], ["missing_pct", "Manquants %"], ["unique", "Uniques"],
  ["mean", "Moyenne"], ["std", "Écart-type"], ["min", "Min"], ["max", "Max"], ["skew", "Asymétrie"], ["kurtosis", "Kurtosis"],
];

/** Provide sortable feature statistics, distribution views and association matrices. */
export function EdaPanel({ eda, feature, busy, onFeature }: {
  eda: Eda; feature: (Feature & { distribution: Bin[] }) | null; busy: boolean; onFeature: (name: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortKey>("column");
  const [desc, setDesc] = useState(false);
  const [matrix, setMatrix] = useState<"pearson" | "cramers_v">("pearson");
  const ranked = eda.target_relations.filter(row => row.effect !== null)
    .sort((a, b) => (a.method ?? "").localeCompare(b.method ?? "") || Math.abs(b.effect ?? 0) - Math.abs(a.effect ?? 0));
  const missing = [...eda.features].sort((a, b) => b.missing_pct - a.missing_pct).slice(0, 20)
    .map(row => ({ label: row.column, count: row.missing_pct }));
  return <>
    <div className="metrics">
      <Metric label="Lignes" value={display(eda.shape[0])} /><Metric label="Colonnes" value={display(eda.shape[1])} />
      <Metric label="Cellules manquantes" value={display(eda.missing_pct) + " %"} />
      <Metric label="Doublons exacts" value={display(eda.duplicate_count)} />
    </div>
    <section className="panel">
      <div className="panel-head"><h2>Inventaire des features</h2>
        <input aria-label="Rechercher une feature" placeholder="Rechercher une feature…" value={query} onChange={e => setQuery(e.target.value)} />
      </div>
      <div className="table-scroll" tabIndex={0}><table>
        <thead><tr>{statsColumns.map(([key, label]) => <th key={key} scope="col"
          aria-sort={key === sort ? (desc ? "descending" : "ascending") : "none"}>
          <button className="sort" onClick={() => { setDesc(key === sort ? !desc : false); setSort(key); }}>
            {label}{sort === key && (desc ? " ↓" : " ↑")}</button></th>)}</tr></thead>
        <tbody>{visibleFeatures(eda.features, query, sort, desc).map(row => <tr key={row.column}>
          <th scope="row"><button className="text-button" disabled={busy} onClick={() => onFeature(row.column)}>{row.column}</button></th>
          {statsColumns.slice(1).map(([key]) => <td key={key}>{display(row[key])}</td>)}</tr>)}</tbody>
      </table></div>
    </section>
    <div className="two-col">
      <section className="panel">
        <label className="field">Distribution par feature
          <select disabled={busy} value={feature?.column ?? ""} onChange={e => onFeature(e.target.value)}>
            {!feature && <option value="">Choisir une feature</option>}
            {eda.features.map(row => <option key={row.column}>{row.column}</option>)}
          </select>
        </label>
        {feature && <>
          <Distribution data={feature.distribution} />
          <p className="muted">{display(feature.missing_count)} valeurs manquantes ({display(feature.missing_pct)} %)</p>
          {feature.boxplot && <>
            <h3>Boîte à moustaches</h3><Boxplot box={feature.boxplot} />
            <DataTable rows={[
              { Méthode: "IQR × 1,5", Extrêmes: feature.iqr?.count ?? null, "Borne basse": feature.iqr?.lower ?? null, "Borne haute": feature.iqr?.upper ?? null },
              { Méthode: "Score z robuste · 3,5 MAD", Extrêmes: feature.robust_z?.count ?? null, "Borne basse": feature.robust_z?.lower ?? null, "Borne haute": feature.robust_z?.upper ?? null },
            ]} />
          </>}
          {feature.text_length && <p>Longueur moyenne des valeurs {display(feature.text_length.mean)} caractères</p>}
        </>}
      </section>
      <section className="panel"><h2>Valeurs manquantes</h2><p className="muted">Jusqu’à 20 features, classées par proportion décroissante</p>
        <Distribution data={missing} unit="Manquants (%)" />
        <details><summary>Exemples de doublons ({display(eda.duplicate_count)} au total)</summary>
          <DataTable rows={eda.duplicate_examples} /></details>
      </section>
    </div>
    <div className="panel-head">
      <h2>Corrélations</h2>
      <select aria-label="Mesure de corrélation" value={matrix} onChange={e => setMatrix(e.target.value as typeof matrix)}>
        <option value="pearson">Pearson (numérique)</option><option value="cramers_v">V de Cramér (catégoriel)</option>
      </select>
    </div>
    <Heatmap matrix={eda[matrix]} title={matrix === "pearson" ? "Corrélation de Pearson" : "V de Cramér · modalités limitées à 40"} />
    {!!eda.target_relations.length && <section className="panel">
      <h2>Relations avec la cible</h2>
      <p className="muted">Effets classés par mesure · échantillon de {display(eda.relation_sample_size)} lignes</p>
      <DataTable rows={[...ranked, ...eda.target_relations.filter(row => row.effect === null)].map(row =>
        ({ Feature: row.column, Mesure: row.method, Effet: row.effect, Effectif: row.sample_size, Note: row.note }))} />
    </section>}
  </>;
}

/** Show evidence, editable alternatives and a before/after chart for one feature. */
export function CleaningPanel({ rows, choices, selected, preview, busy, onChoice, onReset, onPreview }: {
  rows: Recommendation[]; choices: Record<string, string>; selected: string | null; preview: Preview | null; busy: boolean;
  onChoice: (name: string, choice: string) => void; onReset: () => void; onPreview: (name: string) => void;
}) {
  const row = rows.find(item => item.column === selected);
  return <>
    <section className="panel">
      <div className="panel-head"><h2>Choix par feature</h2>
        <button disabled={busy} onClick={onReset}>Accepter toutes les recommandations</button></div>
      <div className="table-scroll" tabIndex={0}><table className="clean-table">
        <thead><tr><th>Feature</th><th>Anomalies</th><th>Recommandation et preuve</th><th>Votre choix</th></tr></thead>
        <tbody>{rows.map(item => <tr key={item.column}>
          <th scope="row">{item.is_target ? item.column : <button className="text-button" disabled={busy}
            aria-pressed={selected === item.column} onClick={() => onPreview(item.column)}>{item.column}</button>}
            <small>{item.type}</small></th>
          <td>{item.anomalies.length ? item.anomalies.join(" · ") : "—"}</td>
          <td><strong>{item.label}</strong><small>{item.justification.preuve}</small>
            <small>{item.justification.explication || item.justification.raison}</small></td>
          <td>{item.is_target ? "Cible" : <select aria-label={`Transformation pour ${item.column}`} disabled={busy}
            value={effectiveChoice(choices, item.column)} onChange={event => onChoice(item.column, event.target.value)}>
            <option value="recommended">Recommandation</option><option value="none">Ne rien appliquer</option>
            {item.alternatives.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>}</td></tr>)}</tbody>
      </table></div>
    </section>
    <section className="panel">
      <div className="panel-head"><h2>Aperçu avant / après</h2>
        <select aria-label="Feature à comparer" disabled={busy} value={selected ?? ""} onChange={event => onPreview(event.target.value)}>
          {!selected && <option value="">Choisir une feature</option>}
          {rows.filter(item => !item.is_target).map(item => <option key={item.column}>{item.column}</option>)}
        </select>
      </div>
      {row && <details><summary>Détail des étapes recommandées pour {row.column}</summary>
        <DataTable rows={Object.entries(row.steps).map(([step, detail]) =>
          ({ Étape: step, Action: detail.action, Mesure: detail.measure, Seuil: detail.threshold, Explication: detail.explanation }))} />
      </details>}
      {preview && <>
        <div className="two-col">
          <div><h3>Avant</h3><Distribution data={preview.before_chart} /></div>
          <div>{preview.after_charts.length ? preview.after_charts.map(chart => <div key={chart.title}>
            <h3>Après · {chart.title}</h3><Distribution data={chart.data} /></div>) : <p>Aucune sortie pour cette feature.</p>}</div>
        </div>
        {preview.notes.map(note => <p className="muted" key={note}>{note}</p>)}
        {!!preview.table.length && <DataTable rows={preview.table} />}
      </>}
    </section>
  </>;
}

const exports = [["train", "Train CSV"], ["test", "Test CSV"], ["full", "Complet CSV"], ["report_xlsx", "Rapport Excel"]];

/** Launch a train/test fit and present its downloadable artifacts. */
export function ExecutionPanel({ sessionId, busy, result, onRun }: {
  sessionId: string; busy: boolean; result: Result | null; onRun: (testSize: number) => void;
}) {
  const [testSize, setTestSize] = useState(20);
  return <>
    <section className="panel">
      <h2>Ajustement et découpage</h2>
      <p className="muted">Les paramètres sont appris sur le train. Le test est transformé avec ces paramètres.</p>
      <form className="run-form" onSubmit={event => { event.preventDefault(); onRun(testSize); }}>
        <label className="field">Part du test (%)
          <input type="number" min={1} max={50} required disabled={busy} value={Number.isNaN(testSize) ? "" : testSize}
            onChange={e => setTestSize(e.currentTarget.valueAsNumber)} />
        </label>
        <button className="primary" type="submit" disabled={busy}>Exécuter le prétraitement</button>
      </form>
    </section>
    {result && <section className="panel">
      <h2>Résultats et fichiers</h2>
      <div className="metrics">
        <Metric label="Entrée" value={shape(result.input_shape)} /><Metric label="Train" value={shape(result.train_shape)} />
        <Metric label="Test" value={shape(result.test_shape)} /><Metric label="Complet" value={shape(result.full_shape)} />
      </div>
      <p className="muted">Processus isolé en {display(result.worker.elapsed_seconds)} s, pic de {display(result.worker.peak_rss_bytes / 1024 ** 2)} Mio</p>
      {result.warnings.length > 0 && <details open><summary>Notes et avertissements ({result.warnings.length})</summary>
        <ul>{result.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul></details>}
      <details><summary>Colonnes de sortie ({result.output_columns.length})</summary>
        <p>{result.output_columns.join(" · ")}</p></details>
      <div className="downloads">{exports.filter(([key]) => result.downloads.includes(key)).map(([key, label]) =>
        <a key={key} className={key === "report_xlsx" ? "primary" : "button"} href={apiRoot + `/api/download/${sessionId}/${key}`}>{label} ↓</a>)}</div>
    </section>}
  </>;
}
