/** Independent outer evaluation, separate from the exploratory /evaluate panel. */
import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { api } from "./api";
import type { Capabilities, NestedRequest, NestedResult, Options, Row } from "./api";
import { DataTable } from "./components";

const countMethods = new Set(["kbest", "chi2", "anova", "mutual_info", "rfe", "forward", "backward",
  "l1", "elastic_net", "select_from_model", "random_forest"]);
const wrapperMethods = new Set(["rfecv", "forward", "backward"]);

type Draft = { sessionId: string; task: string; options: Options; family: string; outerFolds: number; innerFolds: number;
  split: string; splitColumn: string; method: string; counts: string; scalers: string[]; modelParams: string;
  percentile: number; selectorFolds: number };

export function buildNestedRequest(draft: Draft, catalog: Capabilities, columns: string[]): NestedRequest {
  if (!["classification", "regression"].includes(draft.task)) throw new Error("Choisissez explicitement classification ou régression à l’import.");
  const schema = catalog.nested_evaluation?.properties;
  const methods = catalog.options.properties.selection_method?.enum ?? [];
  if (!schema || !methods.includes(draft.method)) throw new Error("Le serveur ne publie pas cette méthode de sélection.");
  if (draft.task === "regression" && draft.method === "chi2") throw new Error("Chi-Square exige une classification.");
  const families = schema.family?.enum ?? [];
  if (!families.includes(draft.family) || (draft.task === "regression" && draft.family === "logistic"))
    throw new Error("Ce modèle n’est pas disponible pour la tâche choisie.");
  const validFolds = (value: number, field: string) => {
    const limits = schema[field];
    if (!Number.isInteger(value) || value < (limits?.minimum ?? 2) || value > (limits?.maximum ?? 5))
      throw new Error("Le nombre de plis doit être dans les limites du serveur.");
  };
  validFolds(draft.outerFolds, "outer_folds");
  validFolds(draft.innerFolds, "inner_folds");
  if (!["stratified", "kfold", "group", "time"].includes(draft.split))
    throw new Error("Stratégie de plis inconnue.");
  if (draft.split === "stratified" && draft.task !== "classification"
      || draft.split === "kfold" && draft.task !== "regression") throw new Error("La stratégie de plis est incompatible avec la tâche.");
  if (draft.options.sparse_output && draft.family === "gradient_boosting")
    throw new Error("Gradient boosting exige une entrée dense.");
  if (["group", "time"].includes(draft.split) && (!draft.splitColumn || !columns.includes(draft.splitColumn)))
    throw new Error("Choisissez une colonne de groupe ou de date présente dans le fichier.");
  if (draft.splitColumn && !["group", "time"].includes(draft.split)) throw new Error("Colonne de découpage inattendue.");
  let params: unknown;
  try { params = JSON.parse(draft.modelParams); }
  catch { throw new Error("Les paramètres du modèle doivent être un tableau JSON valide."); }
  if (!Array.isArray(params) || params.length < 1 || params.length > 4 || params.some(item =>
    item === null || typeof item !== "object" || Array.isArray(item) || Object.values(item).some(value =>
      value !== null && !["string", "number", "boolean"].includes(typeof value)
      || typeof value === "number" && !Number.isFinite(value))))
    throw new Error("Indiquez de 1 à 4 objets JSON contenant uniquement des paramètres scalaires.");
  const counts = draft.counts.trim() ? draft.counts.split(",").map(value => Number(value.trim())) : [];
  if (counts.length > 4 || counts.some(value => !Number.isInteger(value) || value < 1 || value > 200)
      || counts.length > 0 && !countMethods.has(draft.method))
    throw new Error("Indiquez au plus quatre nombres de variables entre 1 et 200 pour cette méthode.");
  const availableScalers = schema.scalers?.items?.enum ?? ["none", "standard", "robust", "minmax"];
  if (draft.scalers.length > 4 || draft.scalers.some(value => !availableScalers.includes(value)))
    throw new Error("Une échelle choisie n’est pas prise en charge par le serveur.");
  if (Math.max(1, counts.length) * Math.max(1, draft.scalers.length) * params.length > 16)
    throw new Error("La recherche est limitée à 16 combinaisons de configurations.");
  const options: Options = { ...draft.options, selection_method: draft.method };
  if (draft.method === "percentile") {
    if (!Number.isInteger(draft.percentile) || draft.percentile < 1 || draft.percentile > 100)
      throw new Error("Le pourcentage doit être entre 1 et 100.");
    options.selection_percentile = draft.percentile;
  }
  if (wrapperMethods.has(draft.method)) {
    validFolds(draft.selectorFolds, "inner_folds");
    options.selection_folds = draft.selectorFolds;
  }
  return { session_id: draft.sessionId, options, family: draft.family,
    outer_folds: draft.outerFolds, inner_folds: draft.innerFolds,
    time_column: draft.split === "time" ? draft.splitColumn : null,
    group_column: draft.split === "group" ? draft.splitColumn : null,
    selection_counts: counts, scalers: draft.scalers, model_params: params as NestedRequest["model_params"] };
}

const metricLabel: Record<string, string> = { accuracy: "Accuracy", balanced_accuracy: "Balanced accuracy",
  precision: "Précision", recall: "Rappel", f1: "F1", roc_auc: "ROC-AUC", pr_auc: "PR-AUC",
  f1_macro: "F1 macro", f1_weighted: "F1 pondéré", mae: "MAE", rmse: "RMSE", r2: "R²" };
const score = (value: number | null | undefined) => value == null ? "Indisponible" : value.toLocaleString("fr-FR", { maximumFractionDigits: 4 });
const description = (value: Record<string, unknown>) => Object.keys(value).length ? JSON.stringify(value) : "Configuration courante";

export function NestedResults({ result, selectionMethod, family, request }: { result: NestedResult; selectionMethod: string; family: string; request?: NestedRequest | null }) {
  const summaryRows: Row[] = Object.entries(result.outer_summary).map(([key, item]) => ({
    Métrique: metricLabel[key] ?? key, Moyenne: score(item.mean), "Écart-type": score(item.std), Plis: item.folds,
  }));
  const globalRows: Row[] = Object.entries(result.global_metrics).filter(([key]) => key !== "confusion_matrix")
    .map(([key, value]) => ({ Métrique: metricLabel[key] ?? key, "Valeur sur les prédictions externes": score(value as number) }));
  return <div className="nested-results" aria-label="Résultats Nested CV">
    <p className="notice">Résultat calculé et validation externe réalisée sur les données de cette exécution. Les scores internes choisissent une configuration ; les plis externes évaluent ce choix sur des lignes écartées de cet ajustement. Les moyennes et écarts-types sont descriptifs. Les plis répétés partagent des observations et sont dépendants : aucune supériorité générale ni incertitude indépendante n’est validée ici.</p>
    {request && <p className="muted">Protocole demandé : {request.outer_folds} plis externes ; {request.inner_folds} plis internes{wrapperMethods.has(selectionMethod) ? ` ; ${String(request.options.selection_folds ?? 3)} plis du sélecteur enveloppe` : ""}. Ces trois niveaux ont des rôles distincts.</p>}
    {result.split_mode === "time" && <p className="notice">Résultat temporel descriptif. La disponibilité réelle des variables à l’instant de prédiction n’est pas vérifiée par ce découpage. Sans snapshots historiques ou journal horodaté, aucune performance de prévision opérationnelle n’est validée.</p>}
    <section className="nested-external" aria-labelledby="nested-external-title">
      <h3 id="nested-external-title">Résultats des plis externes</h3>
      <p>{result.outer_folds.length} plis · {result.predicted_row_positions.length} observations évaluées · {result.unscored_row_positions.length} non évaluées · découpage {result.split_mode} · modèle {family} · sélection {selectionMethod}</p>
      {result.unscored_row_positions.length > 0 && <p className="muted">Positions non évaluées : {result.unscored_row_positions.join(", ")}</p>}
      <h4>Moyenne et écart-type descriptifs des plis externes</h4><DataTable rows={summaryRows} />
      <h4>Métriques globales sur les prédictions externes</h4><DataTable rows={globalRows} />
      {Array.isArray(result.global_metrics.confusion_matrix) && <><h4>Matrice de confusion globale</h4>
        <DataTable rows={(result.global_metrics.confusion_matrix as number[][]).map((row, index) => ({
          "Classe réelle": String(result.classes?.[index] ?? index),
          ...Object.fromEntries(row.map((value, column) => [`Prédit ${String(result.classes?.[column] ?? column)}`, value])),
        }))} /></>}
      {result.task === "classification" && result.classes?.length === 2 && !("roc_auc" in result.global_metrics) &&
        <p className="muted">ROC-AUC et PR-AUC indisponibles pour les prédictions externes de ce protocole.</p>}
      {result.outer_folds.map(fold => <details key={fold.fold} className="nested-fold">
        <summary>Pli externe {fold.fold} · {fold.validation_row_positions.length} observations · {fold.selected_features.length} features</summary>
        <p>Temps : {score(fold.elapsed_seconds)} s · fits de pipeline : {fold.pipeline_fit_count ?? "indisponible"}</p>
        {(fold.metric_warnings ?? []).map((warning, index) => <p key={index} className="notice">{warning}</p>)}
        <DataTable rows={Object.entries(fold.metrics).filter(([key]) => key !== "confusion_matrix")
          .map(([key, value]) => ({ Métrique: metricLabel[key] ?? key, Valeur: score(value as number) }))} />
        {Array.isArray(fold.metrics.confusion_matrix) && <><h4>Matrice de confusion du pli</h4>
          <DataTable rows={(fold.metrics.confusion_matrix as number[][]).map((row, index) => ({
            "Classe réelle": String(result.classes?.[index] ?? index),
            ...Object.fromEntries(row.map((value, column) => [`Prédit ${String(result.classes?.[column] ?? column)}`, value])),
          }))} /></>}
        <p>Configuration retenue : sélection {selectionMethod} ; prétraitement {description(fold.selected_config.preprocessing)} ; modèle {description(fold.selected_config.model)}.</p>
        <p>Features retenues : {fold.selected_features.join(", ") || "Aucune"}.</p>
        <section aria-label={`Recherche interne du pli ${fold.fold}`}>
          <h4>Recherche interne · pli externe {fold.fold}</h4>
          <p>Métrique de sélection : {result.scoring_inner}. Ces scores ne sont pas une performance finale.</p>
          <DataTable rows={fold.inner_search.map((item, index) => ({
            Configuration: index + 1, Prétraitement: description(item.preprocessing), Modèle: description(item.model),
            "Score interne moyen": score(item.mean), "Scores des plis internes": item.scores.map(score).join(" ; "),
          }))} />
        </section>
      </details>)}
    </section>
  </div>;
}

export function NestedEvaluationPanel({ sessionId, columns, target, task, options, busy, onTaskChange }: {
  sessionId: string; columns: string[]; target: string; task: string; options: Options; busy: boolean;
  onTaskChange: (task: "classification" | "regression") => void;
}) {
  const [catalog, setCatalog] = useState<Capabilities | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const [retry, setRetry] = useState(0);
  const [family, setFamily] = useState(task === "classification" ? "logistic" : "linear");
  const [outerFolds, setOuterFolds] = useState(3);
  const [innerFolds, setInnerFolds] = useState(3);
  const [split, setSplit] = useState(task === "regression" ? "kfold" : "stratified");
  const [splitColumn, setSplitColumn] = useState("");
  const [method, setMethod] = useState(task === "regression" && options.selection_method === "chi2"
    ? "auto" : String(options.selection_method ?? "auto"));
  const [counts, setCounts] = useState("");
  const [scalers, setScalers] = useState<string[]>([]);
  const [modelParams, setModelParams] = useState("[{}]");
  const [percentile, setPercentile] = useState(Number(options.selection_percentile ?? 50));
  const [selectorFolds, setSelectorFolds] = useState(Number(options.selection_folds ?? 3));
  const [result, setResult] = useState<NestedResult | null>(null);
  const [lastRequest, setLastRequest] = useState<NestedRequest | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const requestVersion = useRef(0);
  useEffect(() => {
    requestVersion.current += 1;
    setResult(null); setLastRequest(null); setError(""); setLoading(false);
  }, [sessionId, target, task, options]);
  useEffect(() => () => { requestVersion.current += 1; }, []);
  useEffect(() => {
    let cancelled = false;
    api<Capabilities>("/capabilities").then(data => {
      if (cancelled) return;
      if (!data.nested_evaluation?.properties.family?.enum?.length) throw new Error("Le serveur ne publie pas la validation imbriquée.");
      setCatalog(data); setCatalogError("");
    }).catch(reason => { if (!cancelled) setCatalogError(reason instanceof Error ? reason.message : "Catalogue indisponible."); });
    return () => { cancelled = true; };
  }, [retry]);
  const fields = catalog?.nested_evaluation?.properties;
  const familyChoices = (fields?.family?.enum ?? []).filter(name => task !== "regression" || name !== "logistic")
    .filter(name => !(options.sparse_output && name === "gradient_boosting"));
  const methods = catalog?.options.properties.selection_method?.enum ?? [];
  const scaleChoices = fields?.scalers?.items?.enum ?? [];
  const invalidate = () => { requestVersion.current += 1; setResult(null); setLastRequest(null); setError(""); setLoading(false); };
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!catalog || loading) return;
    let request: NestedRequest;
    try {
      request = buildNestedRequest({ sessionId, task, options, family, outerFolds, innerFolds, split, splitColumn,
        method, counts, scalers, modelParams, percentile, selectorFolds }, catalog, columns.filter(name => name !== target));
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Configuration invalide."); return; }
    const version = ++requestVersion.current;
    setResult(null); setLastRequest(null); setError(""); setLoading(true);
    try {
      const response = await api<NestedResult>("/evaluate-nested", request);
      if (requestVersion.current === version) { setResult(response); setLastRequest(request); }
    }
    catch (reason) {
      if (requestVersion.current === version)
        setError(reason instanceof Error ? reason.message : "Évaluation indisponible.");
    }
    finally { if (requestVersion.current === version) setLoading(false); }
  }
  function exportResult() {
    if (!result) return;
    const interpretation = { computed: true, validated_on_executed_data: true, summary: "DESCRIPTIVE",
      generalization: "NOT VALIDATED", temporal_operational: result.split_mode === "time" ? "NOT VALIDATED" : "NOT APPLICABLE" };
    const url = URL.createObjectURL(new Blob([JSON.stringify({ request: lastRequest, result, interpretation }, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url; link.download = "nested_cv_result.json"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <section className="panel nested-panel" aria-labelledby="nested-title"><details>
    <summary id="nested-title">Validation croisée imbriquée · résultats calculés</summary>
    <p>Le prétraitement, la sélection et le modèle sont réajustés dans chaque sous-train. Le score interne choisit la configuration ; les métriques externes évaluent ce choix. Entre 20 et 5 000 cibles renseignées, 16 configurations au plus.</p>
    <p className="muted">Cible : {target} · fichier courant · type de tâche configuré sur le serveur : {task}.</p>
    <label className="field">Type de tâche
      <select aria-label="Type de tâche Nested CV" value={task} disabled={busy || loading} onChange={event => onTaskChange(event.target.value as "classification" | "regression")}>
        {task === "auto" && <option value="auto">Choisir explicitement</option>}
        <option value="classification">Classification</option><option value="regression">Régression</option>
      </select>
    </label>
    <p className="muted">Changer la tâche revient à l’import pour la confirmer auprès du serveur.</p>
    {catalogError && <p role="alert" className="notice error">{catalogError}</p>}
    {!catalog && <button type="button" disabled={busy || loading} onClick={() => setRetry(value => value + 1)}>Recharger les capacités du serveur</button>}
    <form onSubmit={event => void submit(event)}><fieldset disabled={busy || loading || !catalog || task === "auto"}>
      <div className="advanced-grid">
        <label className="field">Modèle Nested CV<select aria-label="Modèle Nested CV" value={family} onChange={event => { invalidate(); setFamily(event.target.value); setModelParams("[{}]"); }}>
          {familyChoices.map(name => <option key={name}>{name}</option>)}</select></label>
        <label className="field">Plis externes<input aria-label="Plis externes" type="number" required min={fields?.outer_folds?.minimum ?? 2} max={fields?.outer_folds?.maximum ?? 5} value={outerFolds} onChange={event => { invalidate(); setOuterFolds(event.target.valueAsNumber); }} /></label>
        <label className="field">Plis internes<input aria-label="Plis internes" type="number" required min={fields?.inner_folds?.minimum ?? 2} max={fields?.inner_folds?.maximum ?? 5} value={innerFolds} onChange={event => { invalidate(); setInnerFolds(event.target.valueAsNumber); }} /></label>
        <label className="field">Stratégie de validation<select aria-label="Stratégie de validation" value={split} onChange={event => { invalidate(); setSplit(event.target.value); setSplitColumn(""); }}>
          {task === "classification" && <option value="stratified">Stratifiée</option>}
          {task === "regression" && <option value="kfold">KFold</option>}
          <option value="group">Groupée</option><option value="time">Temporelle</option></select></label>
        {(split === "group" || split === "time") && <label className="field">{split === "group" ? "Colonne de groupe" : "Colonne temporelle"}
          <select aria-label={split === "group" ? "Colonne de groupe" : "Colonne temporelle"} required value={splitColumn} onChange={event => { invalidate(); setSplitColumn(event.target.value); }}>
            <option value="">Choisir une colonne</option>{columns.filter(name => name !== target).map(name => <option key={name}>{name}</option>)}</select></label>}
        <label className="field">Méthode de sélection<select aria-label="Méthode Nested CV" value={method} onChange={event => { invalidate(); setMethod(event.target.value); setCounts(""); }}>
          {methods.filter(name => task !== "regression" || name !== "chi2").map(name => <option key={name}>{name}</option>)}</select></label>
        {countMethods.has(method) && <label className="field">Nombres de features à comparer (1 à 4, séparés par virgule)
          <input aria-label="Nombres de features" value={counts} placeholder="Ex. 2, 4" onChange={event => { invalidate(); setCounts(event.target.value); }} /></label>}
        {method === "percentile" && <label className="field">Pourcentage de features<input aria-label="Pourcentage Nested CV" type="number" min={1} max={100} value={percentile} onChange={event => { invalidate(); setPercentile(event.target.valueAsNumber); }} /></label>}
        {wrapperMethods.has(method) && <label className="field">Plis du sélecteur<input aria-label="Plis du sélecteur Nested CV" type="number" min={2} max={5} value={selectorFolds} onChange={event => { invalidate(); setSelectorFolds(event.target.valueAsNumber); }} /></label>}
      </div>
      <fieldset className="nested-scalers"><legend>Échelles à comparer (facultatif)</legend>
        {scaleChoices.map(name => <label key={name}><input type="checkbox" checked={scalers.includes(name)} onChange={event => { invalidate(); setScalers(current => event.target.checked ? [...current, name] : current.filter(value => value !== name)); }} /> {name}</label>)}
      </fieldset>
      <label className="field">Paramètres du modèle · tableau JSON de 1 à 4 objets<textarea aria-label="Paramètres du modèle Nested CV" rows={3} value={modelParams} onChange={event => { invalidate(); setModelParams(event.target.value); }} /></label>
      {method === "chi2" && <p className="muted">Chi-Square exige des features non négatives ; choisissez MinMax si nécessaire.</p>}
      <p className="muted">Une colonne de groupe est exclue des features ; la colonne temporelle sert uniquement au découpage. Les classes absentes d’un train provoquent une erreur explicite.</p>
      <button className="primary" type="submit">Lancer la Nested CV</button>
    </fieldset></form>
    {loading && <p role="status">Nested CV en cours dans le worker Python… Le calcul peut durer plusieurs minutes.</p>}
    {error && <p role="alert" className="notice error">Calcul non réalisé — protocole rejeté ou configuration invalide : {error}</p>}
    {result && <><button type="button" onClick={exportResult}>Exporter les résultats Nested CV (JSON)</button><NestedResults result={result} selectionMethod={method} family={family} request={lastRequest} /></>}
  </details></section>;
}
