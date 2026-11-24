import { useEffect, useState } from "react";
import { api } from "./api";
import type { Capabilities, EvaluationResult, Options } from "./api";
import { DataTable } from "./components";

const structured = ["value_bounds", "ordinal_orders", "keep_columns", "selection_include", "selection_exclude"];
const groups: Record<string, string[]> = {
  "Réglages généraux": ["test_size_pct", "scaler_type", "outlier_method", "imputation_strategy", "adaptive_recommendations", "random_state", "model_family", "sparse_output"],
  "Nettoyage et types": ["clean_column_names", "clean_string_values", "parse_numeric_strings", "numeric_string_threshold", "downcast_numeric", "encode_discrete_as_categorical", "value_bounds", "ordinal_orders"],
  "Sélection et prévention des fuites": ["selection_method", "selection_count", "selection_percentile", "selection_folds", "selection_scoring", "selection_include", "selection_exclude", "missing_threshold", "correlation_threshold", "variance_threshold", "constant_threshold", "identifier_unique_threshold", "min_features_for_collinearity", "drop_leaky_columns", "leakage_threshold", "leakage_min_gap", "min_features_for_auto_drop", "noise_guard", "noise_margin", "min_rows_for_noise_guard", "keep_columns"],
  "Texte, encodage et cible": ["cardinality_threshold", "rare_category_threshold", "max_text_features", "text_svd_components", "supervised_encoding", "transform_target", "target_skew_threshold"],
};

export function AdvancedSettings({ options, busy, onApply, onImport }: { options: Options; busy: boolean; onApply: (options: Options) => void; onImport: (file: File) => void }) {
  const [catalog, setCatalog] = useState<Capabilities | null>(null);
  const [draft, setDraft] = useState<Options>(options);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => { setDraft(options); }, [options]);
  useEffect(() => {
    let cancelled = false;
    api<Capabilities>("/capabilities").then(data => { if (!cancelled) { setCatalog(data); setError(""); } })
      .catch(reason => { if (!cancelled) setError(reason.message); });
    return () => { cancelled = true; };
  }, [retry]);
  const entries = Object.entries(catalog?.options.properties ?? {});
  const sections = { ...groups, "Seuils statistiques": entries.map(([key]) => key).filter(key => !Object.values(groups).flat().includes(key)) };
  const selectionMethod = String(draft.selection_method ?? catalog?.defaults.selection_method ?? "auto");
  const countMethods = new Set(["kbest", "chi2", "anova", "mutual_info", "rfe", "forward", "backward", "l1", "elastic_net", "select_from_model", "random_forest"]);
  const cvMethods = new Set(["rfecv", "forward", "backward"]);
  return <section className="panel"><details>
    <summary>Réglages avancés du moteur</summary>
    <p className="muted">Les mêmes réglages s’appliquent aux recommandations, aux aperçus et au traitement final. Les paramètres appris utilisent le train.</p>
    <label className="field">Importer une configuration JSON pour la cible et la tâche courantes<input type="file" accept=".json" disabled={busy} onChange={event => {
      const file = event.currentTarget.files?.[0];
      if (file) onImport(file);
      event.currentTarget.value = "";
    }} /></label>
    {error && <p role="alert">{error}</p>}
    {!catalog ? <button disabled={busy} onClick={() => setRetry(value => value + 1)}>Recharger les réglages disponibles</button> :
      <form onSubmit={event => {
        event.preventDefault();
        try {
          const next = { ...draft };
          structured.forEach(key => { if (typeof next[key] === "string") next[key] = JSON.parse(next[key] as string); });
          setError(""); onApply(next);
        } catch { setError("Vérifiez la syntaxe JSON des bornes, ordres et variables protégées."); }
      }}><fieldset disabled={busy}>
        {Object.entries(sections).map(([name, keys]) => <details className="settings-group" key={name}>
          <summary>{name}</summary><div className="advanced-grid">{keys.filter(key =>
            key !== "selection_count" || countMethods.has(selectionMethod)).filter(key =>
            key !== "selection_percentile" || selectionMethod === "percentile").filter(key =>
            !["selection_folds", "selection_scoring"].includes(key) || cvMethods.has(selectionMethod)).map(key => {
            const field = catalog.options.properties[key];
            if (!field) return null;
            const value = Object.hasOwn(draft, key) ? draft[key] : catalog.defaults[key];
            const choices = field.enum ?? field.anyOf?.find(item => item.enum)?.enum;
            return <label className="field" key={key}>{field.title}
              {field.type === "boolean" ? <input aria-label={field.title} type="checkbox" checked={!!value} onChange={event => setDraft(previous => ({ ...previous, [key]: event.target.checked }))} />
                : choices ? <select aria-label={field.title} value={value == null ? "" : String(value)} onChange={event => setDraft(previous => ({ ...previous, [key]: event.target.value || null }))}>
                  {field.anyOf?.some(item => item.type === "null") && <option value="">Statistique seule</option>}
                  {choices.map(option => <option key={option}>{option}</option>)}</select>
                : structured.includes(key) ? <textarea aria-label={field.title} rows={3} value={typeof value === "string" ? value : JSON.stringify(value, null, 2)}
                  onChange={event => setDraft(previous => ({ ...previous, [key]: event.target.value }))} />
                : <input aria-label={field.title} type="number" step={field.type === "integer" ? 1 : "any"} min={field.minimum ?? (field.exclusiveMinimum === undefined ? undefined : field.exclusiveMinimum + Number.EPSILON * Math.max(1, Math.abs(field.exclusiveMinimum)))}
                  max={field.maximum ?? (field.exclusiveMaximum === undefined ? undefined : field.exclusiveMaximum - Number.EPSILON * Math.max(1, Math.abs(field.exclusiveMaximum)))} required={!field.anyOf?.some(item => item.type === "null")} value={typeof value === "number" && Number.isFinite(value) ? value : ""}
                  onChange={event => setDraft(previous => ({ ...previous, [key]: event.target.value === "" ? null : event.target.valueAsNumber }))} />}
            </label>;
          })}</div></details>)}
        {selectionMethod === "chi2" && <p className="muted">Chi-Square requiert une cible de classification et des features encodées non négatives. Une mise à l’échelle MinMax peut être nécessaire.</p>}
        {cvMethods.has(selectionMethod) && <p className="muted">Chaque pli réajuste son prétraitement lorsque les opérations préservent les lignes. Le rapport signale toute restriction ; une évaluation externe reste nécessaire pour mesurer la performance finale.</p>}
        <p className="muted">Bornes métier au format {`{"montant": [0, null]}`}, ordres au format {`{"niveau": ["faible", "fort"]}`} et variables protégées au format {`["date"]`}.</p>
        <div className="downloads"><button className="primary" type="submit">Appliquer les réglages et recalculer</button>
          <button type="button" onClick={() => { setDraft({}); onApply({}); }}>Rétablir les réglages par défaut</button></div>
      </fieldset></form>}
  </details></section>;
}

export function EvaluationPanel({ columns, supervised, busy, result, onEvaluate }: {
  columns: string[]; supervised: boolean; busy: boolean; result: EvaluationResult | null;
  onEvaluate: (family: string, folds: number, timeColumn: string) => void;
}) {
  const [family, setFamily] = useState("linear");
  const [folds, setFolds] = useState(3);
  const [time, setTime] = useState("");
  const [catalog, setCatalog] = useState<Capabilities | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let cancelled = false;
    api<Capabilities>("/capabilities").then(data => {
      if (cancelled) return;
      if (!data.evaluation?.properties.family.enum?.length) throw new Error("Le serveur ne fournit pas les modèles disponibles. Redémarrez le serveur à jour.");
      setCatalog(data); setError("");
    }).catch(reason => { if (!cancelled) setError(reason.message); });
    return () => { cancelled = true; };
  }, [retry]);
  const evaluation = catalog?.evaluation?.properties;
  return <section className="panel"><details><summary>Validation croisée des mises à l’échelle</summary>
    <p>Cet essai fixe les règles globales pour comparer uniquement l’échelle et réajuste le préprocesseur dans chaque pli. Entre 20 et 5 000 cibles renseignées sont nécessaires. Une évaluation indépendante reste nécessaire pour mesurer le choix retenu.</p>
    {!supervised ? <p className="muted">Choisissez une cible pour réaliser une évaluation prédictive.</p> :
      <form onSubmit={event => { event.preventDefault(); onEvaluate(family, folds, time); }}><fieldset disabled={busy || !evaluation}>
        <div className="advanced-grid"><label className="field">Modèle de validation<select aria-label="Modèle de validation" value={family} onChange={event => setFamily(event.target.value)}>
          {evaluation?.family.enum?.map(name => <option key={name}>{name}</option>)}
        </select></label><label className="field">Nombre de plis<input type="number" min={evaluation?.folds.minimum} max={evaluation?.folds.maximum} value={folds} required onChange={event => setFolds(event.target.valueAsNumber)} /></label>
           <label className="field">Type de découpage<select value={time} onChange={event => setTime(event.target.value)}><option value="">Aléatoire reproductible</option>
             {columns.filter(column => /date|time|timestamp|jour|mois|année|year/i.test(column)).map(column => <option key={column} value={column}>Temporel · {column}</option>)}</select></label></div>
        <button type="submit">Comparer par validation croisée</button>
      </fieldset></form>}
    {error && <p role="alert">{error}</p>}
    {!catalog && <button disabled={busy} onClick={() => setRetry(value => value + 1)}>Recharger les modèles disponibles</button>}
    {result && <><p>Modèle évalué · {result.family}</p><p>Métrique {result.scoring} · {result.folds} plis</p><DataTable rows={result.results.map(row => ({
      Échelle: row.scaler, Moyenne: row.mean, "Écart-type": row.std, "Scores par pli": row.scores.join(", "),
    }))} /></>}
  </details></section>;
}
