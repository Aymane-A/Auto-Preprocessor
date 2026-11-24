/** French workflow panels for upload, exploration, choices and export. */
import { useState } from "react";
import type { Choices, Choice, Bin, Eda, Feature, Preview, Recommendation, Result, Upload, Imbalance, Options } from "./api";
import { DataTable, display, Distribution, effectiveChoice, Heatmap, Metric, percentDisplay, shape, visibleFeatures } from "./components";
import type { SortKey } from "./components";
import { PreviewComparison } from "./PreviewComparison";
import { FeatureVisualization } from "./FeatureVisualization";
import { actionLabel, operationLabel } from "./labels";
import { RecommendationExplanation } from "./RecommendationExplanation";

export type Configuration = { mode: "supervised" | "unsupervised"; target: string; task: string };

function classCounts(counts: Record<string, number>): string {
  return Object.entries(counts).map(([label, count]) => `${label} : ${count}`).join(", ");
}

export function suggestedTask(dataset: Upload | null, column: string): string {
  if (!dataset || !column) return "auto";
  const values = dataset.preview.map(row => row[column]).filter(value => value !== null && value !== "");
  const unique = new Set(values.map(String)).size;
  const dtype = dataset.dtypes[column] ?? "";
  if (!values.length) return "auto";
  if (!/int|float|double|decimal/i.test(dtype)) return "classification";
  return unique <= 10 && unique / values.length <= .5 ? "classification" : "regression";
}

/** Collect a local dataset and its learning configuration. */
export function ImportPanel({ dataset, config, busy, onFile, onChange, onConfigure }: {
  dataset: Upload | null; config: Configuration; busy: boolean; onFile: (file: File) => void;
  onChange: (next: Configuration) => void; onConfigure: () => void;
}) {
  const [dragging, setDragging] = useState(false);
  return <>
    <div className="two-col import-grid">
      <section className="panel">
        <h2>Fichier source</h2>
        <label className={dragging ? "upload dragging" : "upload"}
          onDragOver={event => { event.preventDefault(); if (!busy) setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={event => {
            event.preventDefault();
            setDragging(false);
            const file = event.dataTransfer.files[0];
            if (file && !busy) onFile(file);
          }}>
          <svg className="upload-icon" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M12 16V3m-5 5 5-5 5 5M4 16v4h16v-4" /></svg>
          <strong>{dataset?.filename ?? "Déposer un fichier ou parcourir"}</strong>
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
                 <select required value={config.target} onChange={event => onChange({ ...config, target: event.target.value, task: suggestedTask(dataset, event.target.value) })}>
                   <option value="">Choisir la cible</option>
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
      <details><summary>Types du fichier source</summary>
        <DataTable rows={dataset.columns.map(column => ({ Colonne: column, Type: dataset.dtypes[column] }))} />
      </details>
    </section>}
  </>;
}

const statsColumns: [SortKey, string][] = [
  ["column", "Feature"], ["type", "Type"], ["missing_pct", "Manquants %"], ["unique", "Uniques"],
  ["mean", "Moyenne"], ["std", "Écart-type"], ["min", "Min"], ["max", "Max"], ["skew", "Asymétrie"], ["kurtosis", "Kurtosis"],
];

/** Provide sortable feature statistics, distribution views and association matrices. */
export function EdaPanel({ eda, feature, featureError = "", busy, onFeature }: {
  eda: Eda; feature: Feature | null; featureError?: string; busy: boolean; onFeature: (name: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortKey>("column");
  const [desc, setDesc] = useState(false);
  const [matrix, setMatrix] = useState<"pearson" | "cramers_v">("pearson");
  const ranked = eda.target_relations.filter(row => row.effect !== null)
    .sort((a, b) => (a.method ?? "").localeCompare(b.method ?? "") || Math.abs(b.effect ?? 0) - Math.abs(a.effect ?? 0));
  const missing = [...eda.features].sort((a, b) => b.missing_pct - a.missing_pct).slice(0, 20)
    .map(row => ({ label: row.column, count: row.missing_pct }));
  const classTotal = Math.max(1, Object.values(eda.imbalance?.counts ?? {}).reduce((a, b) => a + b, 0));
  return <>
    {eda.imbalance && <section className="panel"><h2>Répartition de la cible</h2>
      <div aria-label="Répartition des classes de la cible">{Object.entries(eda.imbalance.counts).map(([label,count]) =>
        <p key={label}>{label} : {display(count)} ({percentDisplay(count / classTotal * 100)} %)
          <span className="target-class-scale" role="meter" aria-label={`Effectif de la classe ${label}`}
            aria-valuemin={0} aria-valuemax={classTotal} aria-valuenow={count}>
            <span style={{ width: `${count / classTotal * 100}%` }} /></span></p>)}</div>
      {eda.imbalance.level && <p role="status">Cible déséquilibrée ({eda.imbalance.level}) : {percentDisplay((eda.imbalance.minority_ratio ?? 0)*100)} % dans la classe minoritaire. L’accuracy peut être trompeuse ; comparer PR-AUC, rappel et F1.</p>}
    </section>}
    <div className="metrics">
      <Metric label="Lignes" value={eda.shape[0]} /><Metric label="Colonnes" value={eda.shape[1]} />
      <Metric label="Cellules manquantes" value={eda.missing_pct} suffix=" %" />
      <Metric label="Doublons exacts" value={eda.duplicate_count} />
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
        <tbody data-motion-key={`${query}|${sort}|${desc}`}>{visibleFeatures(eda.features, query, sort, desc).map(row => <tr key={row.column}>
          <th scope="row"><button className="text-button" disabled={busy} onClick={() => onFeature(row.column)}>{row.column}</button></th>
           {statsColumns.slice(1).map(([key]) => <td key={key}>{key === "type" && row.detected_type === "ID_LIKE" ? "identifiant" : key === "missing_pct" ? new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1, minimumFractionDigits: 1 }).format(row.missing_pct) : display(row[key])}</td>)}</tr>)}</tbody>
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
        {featureError && <p role="alert" className="notice error">{featureError}</p>}
        {feature && <FeatureVisualization feature={feature} />}
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
    {eda.sample_note && <p className="muted">{eda.sample_note}</p>}
    <Heatmap matrix={eda[matrix]} title={matrix === "pearson" ? "Corrélation de Pearson" : "V de Cramér" + (eda.cramers_v.max_categories === undefined ? "" : ` · modalités limitées à ${eda.cramers_v.max_categories}`)} />
    {!!eda.target_relations.length && <section className="panel">
      <h2>Relations avec la cible</h2>
      <p className="muted">Effets classés par mesure · échantillon de {display(eda.relation_sample_size)} lignes</p>
      <DataTable rows={[...ranked, ...eda.target_relations.filter(row => row.effect === null)].map(row =>
        ({ Feature: row.column, Mesure: row.method, Effet: row.effect, "p-value": row.p_value ?? null, "q-value BH": row.q_value ?? null, Effectif: row.sample_size, Note: row.note }))} />
      <details><summary>Tous les tests de relation à la cible</summary><DataTable rows={eda.target_relations.map(row => ({ Variable: row.column, Tests: JSON.stringify(row.tests) }))} /></details>
    </section>}
  </>;
}

/** Show evidence, editable alternatives and a before/after chart for one feature. */
export function CleaningPanel({ rows, choices, selected, preview, busy, onChoice, onReset, onPreview, view = "all" }: {
  rows: Recommendation[]; choices: Choices; selected: string | null; preview: Preview | null; busy: boolean;
  onChoice: (name: string, choice: Choice) => void; onReset: () => void; onPreview: (name: string) => void;
  view?: "all" | "recommendations" | "preview";
}) {
  const row = rows.find(item => item.column === selected);
  return <>
    {view !== "preview" && <section className="panel">
      <div className="panel-head"><h2>Choix par feature</h2>
        <button disabled={busy} onClick={onReset}>Accepter toutes les recommandations</button></div>
      <div className="table-scroll" tabIndex={0}><table className="clean-table">
        <thead><tr><th>Feature</th><th>Anomalies</th><th>Recommandation et preuve</th><th>Votre choix</th></tr></thead>
        <tbody>{rows.map(item => <tr key={item.column}>
          <th scope="row">{item.is_target || view === "recommendations" ? item.column : <button className="text-button" disabled={busy}
            aria-pressed={selected === item.column} onClick={() => onPreview(item.column)}>{item.column}</button>}
            <small>{item.type}</small></th>
          <td>{item.anomalies.length ? item.anomalies.join(" · ") : "—"}</td>
          <td><strong>{item.label}</strong><small>{item.justification.preuve}</small>
            <small>{item.justification.explication || item.justification.raison}</small>
            <RecommendationExplanation row={item} choice={choices[item.column] ?? "recommended"} preview={selected === item.column ? preview : null} />
            <details><summary>Diagnostics avancés de {item.column}</summary><pre className="json-report">{JSON.stringify(item.diagnostics, null, 2)}</pre></details>
            <details><summary>Preuves et confiance par opération</summary>
               {(item.action === "DROP" ? Object.entries(item.steps).filter(([name]) => name === "type") : Object.entries(item.steps)).map(([name, detail]) => <p key={name}>
                 <strong title={name}>{operationLabel(name)}</strong> / <span title={detail.action}>{actionLabel(detail.action)}</span> / confiance {detail.confidence || "non fournie"}<br />
                {detail.measure} / {detail.threshold}
              </p>)}
            </details></td>
          <td>{item.is_target ? "Cible" : <select aria-label={`Transformation pour ${item.column}`} disabled={busy}
            value={effectiveChoice(choices, item.column)} onChange={event => onChoice(item.column, event.target.value)}>
            <option value="custom" disabled>Personnalisé par opération</option><option value="recommended">Recommandation</option><option value="none">Ne rien appliquer</option>
            {item.alternatives.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>}</td></tr>)}</tbody>
      </table></div>
    </section>}
    {view !== "recommendations" && <section className="panel">
      <div className="panel-head"><h2>Aperçu avant / après</h2>
        <select aria-label="Feature à comparer" disabled={busy} value={selected ?? ""} onChange={event => onPreview(event.target.value)}>
          {!selected && <option value="">Choisir une feature</option>}
          {rows.filter(item => !item.is_target).map(item => <option key={item.column}>{item.column}</option>)}
        </select>
      </div>
      {row && <details><summary>Détail des étapes recommandées pour {row.column}</summary>
         <DataTable rows={(row.action === "DROP" ? Object.entries(row.steps).filter(([step]) => step === "type") : Object.entries(row.steps)).map(([step, detail]) =>
           ({ Étape: operationLabel(step), Action: actionLabel(detail.action), Mesure: detail.measure, Seuil: detail.threshold, Explication: detail.explanation }))} />
      </details>}
      {preview && <PreviewComparison key={selected} preview={preview} column={selected ?? ""} />}
      {!selected && <p className="muted">Choisissez une variable pour comparer le traitement sélectionné.</p>}
    </section>}
  </>;
}

const exports = [["train", "Train CSV"], ["test", "Test CSV"], ["full", "Complet CSV"], ["train_provenance", "Provenance du train CSV"], ["report_xlsx", "Rapport Excel"], ["report_json", "Rapport JSON"], ["full_xlsx", "Dataset Excel"], ["preprocessor", "Préprocesseur des variables traitées"]];

/** Launch a train/test fit and present its downloadable artifacts. */
export function ExecutionPanel({ sessionId, busy, runDisabled = false, result, onRun, view = "all", testSize, onTestSize, onDownload, imbalance, options, onImbalance, rawCategorical }: {
  sessionId: string; busy: boolean; result: Result | null; onRun: (testSize: number) => void;
  runDisabled?: boolean;
  onDownload?: (kind: string) => void;
  view?: "all" | "run" | "exports"; testSize: number; onTestSize: (value: number) => void;
  imbalance?: Imbalance | null; options?: Options; onImbalance?: (next: Options) => void;
  rawCategorical?: boolean;
}) {
  return <>
    {view !== "exports" && <section className="panel">
      <h2>Ajustement et découpage</h2>
      <p className="muted">Les paramètres sont appris sur le train. Le test est transformé avec ces paramètres.</p>
      <form className="run-form" onSubmit={event => { event.preventDefault(); if (!runDisabled) onRun(testSize); }}>
        <label className="field">Part du test (%)
          <input type="number" min={1} max={50} required disabled={busy} value={Number.isNaN(testSize) ? "" : testSize}
            onChange={e => onTestSize(e.currentTarget.valueAsNumber)} />
        </label>
        {imbalance && <><label className="field">Gestion du déséquilibre
          <select disabled={busy} value={String(options?.imbalance_method ?? "none")} onChange={e => onImbalance?.({imbalance_method:e.target.value})}>
            <option value="none">Aucun</option><option value="class_weight">Pondération des classes</option>
            <option value="random_over">Sur-échantillonnage aléatoire</option><option value="smote">SMOTE</option>
            {rawCategorical && <option value="smotenc">SMOTENC</option>}
            <option value="random_under">Sous-échantillonnage aléatoire</option>
          </select></label>
          <p className="muted">{imbalance.level === "sévère" ? (Math.min(...Object.values(imbalance.counts)) * (1 - testSize / 100) >= 6
            ? "Suggestion : SMOTE (moins de 5 % dans la classe rare, au moins six exemples estimés dans le train)."
            : "Suggestion : pondération des classes (moins de six exemples rares estimés dans le train).") :
            imbalance.level === "modéré" ? "Suggestion : pondération des classes." : "Répartition sans déséquilibre marqué."} Aucun rééquilibrage n’est appliqué sans sélection.</p>
          <details className="recommendation-explanation"><summary>Pourquoi cette recommandation ?</summary>
            <div className="explanation-content">
              <section><h4>Qu'est-ce que c'est ?</h4><p>La pondération augmente le poids des classes rares dans un modèle compatible. Le sur-échantillonnage ajoute des lignes au train ; le sous-échantillonnage en retire.</p></section>
              <section><h4>Pourquoi cette recommandation ?</h4><p>{imbalance.level ? `Déséquilibre ${imbalance.level} observé dans la cible.` : "Aucun déséquilibre marqué détecté."} Répartition observée : {classCounts(imbalance.counts)}.</p></section>
              <section><h4>Comment ça fonctionne ?</h4><p>Le découpage train/test précède tout rééquilibrage. La méthode choisie agit sur le train seulement ; le test reste intact.</p></section>
              <section><h4>Exemple concret</h4><p>Effectifs observés avant traitement : {classCounts(imbalance.counts)}. Les effectifs après traitement sont affichés après exécution, sans estimation présentée comme résultat.</p></section>
              <section><h4>Impact sur les données</h4><p>La pondération produit des poids à transmettre à un modèle compatible ; elle ne modifie pas les lignes exportées. Les méthodes de rééchantillonnage modifient les lignes du train. Le test reste intact.</p></section>
              <section><h4>Avantages et risques</h4><p>Une classe rare peut recevoir plus d’attention. La pondération exige un modèle compatible ; le sur-échantillonnage peut surajuster et le sous-échantillonnage retire des observations.</p></section>
              <section><h4>Pourquoi les autres méthodes ne sont pas choisies ?</h4><p>Le choix dépend des effectifs observés et du modèle visé. SMOTE requiert assez de voisins dans la classe rare ; la pondération ne crée pas de lignes.</p></section>
              <section><h4>Preuves statistiques</h4><p>Part de la classe minoritaire : {imbalance.minority_ratio == null ? "indisponible" : `${percentDisplay(imbalance.minority_ratio * 100)} %`}. Effectifs : {classCounts(imbalance.counts)}.</p></section>
              <p className="explanation-status"><strong>Votre choix :</strong> {String(options?.imbalance_method ?? "none")}. <strong>Exécution :</strong> non encore exécutée.</p>
            </div>
          </details>
          {!["none","class_weight"].includes(String(options?.imbalance_method ?? "none")) &&
            <label className="field">Ratio minoritaire / majoritaire visé<input type="number" min="0.01" max="1" step="0.01" value={Number(options?.imbalance_ratio ?? .5)}
              onChange={e => onImbalance?.({imbalance_ratio:e.currentTarget.valueAsNumber})} /></label>}
          {["smote","smotenc"].includes(String(options?.imbalance_method)) && <label className="field">Voisins SMOTE<input type="number" min="1" max="50" value={Number(options?.imbalance_neighbors ?? 5)}
            onChange={e => onImbalance?.({imbalance_neighbors:e.currentTarget.valueAsNumber})} /></label>}
          {String(options?.imbalance_method ?? "none") !== "none" && <label className="field">Graine aléatoire<input type="number" min="0" max="2147483647" value={Number(options?.random_state ?? 42)}
            onChange={e => onImbalance?.({random_state:e.currentTarget.valueAsNumber})} /></label>}
        </>}
        <button className="primary" type="submit" disabled={busy || runDisabled}>Exécuter le prétraitement</button>
      </form>
    </section>}
    {view !== "run" && result && <section className="panel">
      <h2>Résultats et fichiers</h2>
      <div className="metrics">
        <Metric label="Entrée" value={result.input_shape} /><Metric label="Train" value={result.train_shape} />
        <Metric label="Test" value={result.test_shape} /><Metric label="Complet" value={result.full_shape} />
      </div>
      <p className="muted">Processus isolé en {display(result.worker.elapsed_seconds)} s, pic de {display(result.worker.peak_rss_bytes / 1024 ** 2)} Mio</p>
      {result.imbalance && <section><h3>Rééquilibrage du train</h3><p>Méthode : {result.imbalance.method}. Train avant : {classCounts(result.imbalance.before.counts)} ; train après : {classCounts(result.imbalance.after.counts)} ; test intact : {classCounts(result.imbalance.test.counts)}. {result.imbalance.synthetic_rows} lignes synthétiques.</p></section>}
      {result.warnings.length > 0 && <details open><summary>Notes et avertissements ({result.warnings.length})</summary>
        <ul>{result.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul></details>}
      <details><summary>Colonnes de sortie ({result.output_columns.length})</summary>
        <p>{result.output_columns.join(" · ")}</p></details>
      <div className="downloads">{exports.filter(([key]) => result.downloads.includes(key)).map(([key, label]) =>
        <button key={key} disabled={busy || !onDownload} className={key === "report_xlsx" ? "primary" : "button"} onClick={() => onDownload?.(key)}>{label} ↓</button>)}</div>
    </section>}
  </>;
}
