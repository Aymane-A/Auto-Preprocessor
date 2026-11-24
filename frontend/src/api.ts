/** Types and requests shared by the six workflow stages. */
export type Cell = string | number | boolean | null;
export type StepChoice = { action: string; parameters: string };
export type Choice = string | Record<string, StepChoice>;
export type Choices = Record<string, Choice>;
export type Options = Record<string, unknown>;
export type OptionField = { title: string; type?: string; enum?: string[]; items?: { enum?: string[] }; anyOf?: { type?: string; enum?: string[] }[]; minimum?: number; maximum?: number; exclusiveMinimum?: number; exclusiveMaximum?: number; default?: unknown };
export type Capabilities = { defaults: Options; options: { properties: Record<string, OptionField> }; formats: string[];
  evaluation?: { properties: { family: OptionField; folds: OptionField } };
  nested_evaluation?: { properties: Record<string, OptionField> } };
export type EvaluationResult = { family: string; scoring: string; folds: number; results: { scaler: string; mean: number; std: number; scores: number[] }[] };
export type NestedRequest = { session_id: string; options: Options; family: string; outer_folds: number; inner_folds: number;
  time_column: string | null; group_column: string | null; selection_counts: number[]; scalers: string[];
  model_params: Record<string, string | number | boolean | null>[] };
export type NestedFold = { fold: number; train_row_positions: number[]; validation_row_positions: number[];
  predictions: Cell[]; targets: Cell[]; metrics: Record<string, number | number[][]>;
  selected_config: { preprocessing: Record<string, unknown>; model: Record<string, unknown> };
  selected_features: string[]; inner_search: { preprocessing: Record<string, unknown>; model: Record<string, unknown>;
    scores: number[]; mean: number }[]; metric_warnings?: string[]; elapsed_seconds?: number; pipeline_fit_count?: number };
export type NestedResult = { protocol: string; task: string; split_mode: string; scoring_inner: string;
  classes: Cell[] | null; outer_folds: NestedFold[];
  outer_summary: Record<string, { mean: number; std: number | null; folds: number }>;
  global_metrics: Record<string, number | number[][]>;
  predicted_row_positions: number[]; unscored_row_positions: number[] };
export type Row = Record<string, Cell>;
export type Shape = [number, number];
export type Bin = { label: string; count: number; percent?: number; total?: number; low?: number; high?: number; is_other?: boolean };
export type Visualization = {
  kind: "binary_bar" | "frequency_bar" | "ordered_bar" | "top_categories" | "value_counts" | "dominant_share" | "missingness" | "histogram" | "boxplot" | "time_series" | "summary" | "target_boxplot" | "target_distribution" | "target_rate";
  reason: string; data?: Bin[]; box?: Box; horizontal?: boolean; scale?: string; sample_size?: number; outlier_count?: number;
  domain?: { low: number; high: number };
  groups?: { label: string; box?: Box; data?: Bin[]; sample_size: number }[]; cardinality?: number; examples?: string[];
};
export type Matrix = { columns: string[]; values: (number | null)[][]; limit: number; eligible: number; max_categories?: number };
export type Box = { low: number; q1: number; median: number; q3: number; high: number };
export type Bounds = { count: number | null; lower: number | null; upper: number | null };
export type Feature = {
  column: string; dtype: string; type: string; missing_pct: number; missing_count: number;
  unique: number; binary?: boolean; mean?: number | null; std?: number | null; min?: number | null; q1?: number | null; median?: number | null; q3?: number | null;
  max?: number | null; skew?: number | null; kurtosis?: number | null; top: Bin[]; boxplot?: Box;
  text_length?: Record<string, number | null>; iqr?: Bounds; robust_z?: Bounds;
  detected_type?: string; recommended_visualization?: Visualization[]; relationship_note?: string; target_type?: string;
  display_type_override?: string;
};
export type Upload = { session_id: string; filename: string; shape: Shape; columns: string[]; dtypes: Record<string, string>; preview: Row[] };
export type Eda = {
  imbalance?: Imbalance | null;
  shape: Shape; features: Feature[]; missing_pct: number; duplicate_count: number; duplicate_examples: Row[];
  pearson: Matrix; cramers_v: Matrix; relation_sample_size: number; sample_note?: string;
  target_relations: { column: string; method: string | null; effect: number | null; sample_size: number; note: string | null;
    p_value?: number | string | null; q_value?: number | string | null; tests?: Record<string, unknown> }[];
};
export type Imbalance = { counts: Record<string, number>; minority_ratio: number | null; imbalance_ratio: number | null; level: string | null };
export type Recommendation = {
  column: string; type: string; is_target: boolean; anomalies: string[];
  action: string; label: string; justification: { raison: string; preuve: string; explication: string };
  alternatives: { value: string; label: string }[];
  steps: Record<string, { action: string; measure: string | null; threshold: string | null; explanation: string | null; confidence: string | null; help?: string;
    label?: string; choices?: { value: string; label: string; parameters: string | null; help: string }[]; comparison?: Record<string, unknown> }>;
  diagnostics?: Record<string, unknown>;
};
export type Preview = {
  status: string; before_chart: Bin[]; after_charts: { title: string; data: Bin[] }[]; notes: string[]; table: Row[];
  rows?: number; output_columns?: string[]; applied?: Record<string, Cell>;
  before?: Record<string, unknown>; after_scaled?: Record<string, unknown>;
  affected?: { missing_filled: number; outside_bounds: number; rows_changed: number; rows_changed_pct: number };
  samples?: { row: Cell; original: Cell; values: Row }[];
};
export type SelectionDetails = { method: string; selected: string[]; rejected: string[];
  scores: Record<string, number | null> | null; rankings: Record<string, number> | null;
  reasons?: Record<string, string>; elapsed_seconds: number; fit_rows: number; scoring?: string | null };
export type SelectionPreview = { selected: string[]; rejected: string[]; original: string[];
  encoded_count: number; selected_count: number; output_count: number; target_included: boolean;
  expected_rows: number; method: string; warnings: string[]; note: string;
  details?: SelectionDetails | null; feature_sources?: Record<string, string> };
export type Result = {
  imbalance?: { method: string; before: Imbalance; after: Imbalance; test: Imbalance; synthetic_rows: number; warnings: string[]; parameters: Record<string, number> } | null;
  input_shape: Shape; train_shape: Shape; test_shape: Shape; full_shape: Shape;
  output_columns: string[]; warnings: string[]; downloads: string[];
  worker: { elapsed_seconds: number; peak_rss_bytes: number };
};
export type TransformationReport = {
  summary: Record<string, unknown>;
  core?: Record<string, unknown>; correlations?: Record<string, Matrix>;
  features: { column: string; type: string; applied: string; choice: Choice; dropped: boolean;
    steps?: Recommendation["steps"]; diagnostics?: Record<string, unknown>;
    drop_reason: string | null; outputs: string[]; parameters: Record<string, unknown>; leakage: number | null }[];
};
export const apiRoot = import.meta.env.DEV ? (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000") : "";

/** Resolve an unspecified feature transformation to its recommendation. */
export function resolvedChoice(choices: Choices, name: string): Choice {
  return Object.hasOwn(choices, name) ? choices[name] : "recommended";
}
export function effectiveChoice(choices: Choices, name: string) {
  const choice = resolvedChoice(choices, name);
  return typeof choice === "string" ? choice : "custom";
}

/** Reuse existing forced-type choices for EDA; no separate display controls. */
export function featureEdaPath(session: string, column: string, choices: Choices): string {
  const choice = resolvedChoice(choices, column);
  const action = typeof choice === "string" ? (choice.startsWith("type:") ? choice.slice(5) : "AUTO") : choice.type?.action ?? "AUTO";
  const forced = ["FORCE_NUMERIC", "FORCE_CATEGORICAL", "FORCE_DATE", "FORCE_TEXT"].includes(action);
  return `/eda/${session}/feature/${encodeURIComponent(column)}` + (forced ? `?display_type=${action}` : "");
}

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

type Check = (value: unknown) => boolean;
const object: Check = value => value !== null && typeof value === "object" && !Array.isArray(value);
const string: Check = value => typeof value === "string";
const number: Check = value => typeof value === "number" && Number.isFinite(value);
const boolean: Check = value => typeof value === "boolean";
const nullable = (check: Check): Check => value => value === null || check(value);
const optional = (check: Check): Check => value => value === undefined || check(value);
const array = (check: Check): Check => value => Array.isArray(value) && value.every(check);
const record = (check: Check): Check => value => object(value) && Object.values(value as object).every(check);
const fields = (schema: Record<string, Check>): Check => value => object(value) && Object.entries(schema).every(([key, check]) => check((value as Record<string, unknown>)[key]));
const shapeCheck: Check = value => Array.isArray(value) && value.length === 2 && value.every(v => Number.isInteger(v) && v >= 0);
const count: Check = value => Number.isInteger(value) && (value as number) >= 0;
const selectionPreviewCheck: Check = value => {
  if (!fields({ status: string, original: array(string), selected: array(string), rejected: array(string),
    encoded_count: count, selected_count: count, output_count: count, target_included: boolean,
    expected_rows: count, method: string, warnings: array(string), note: string })(value)) return false;
  const preview = value as { status: string; selected: string[]; selected_count: number; output_count: number };
  return preview.status === "estimated_train_fit" && preview.selected_count === preview.selected.length
    && preview.output_count >= preview.selected_count;
};
const cell: Check = value => value === null || string(value) || number(value) || boolean(value);
const rows = array(record(cell));
const bins = array(fields({ label: string, count: number }));
const featureCheck = fields({ column: string, dtype: string, type: string, missing_pct: number, missing_count: number, unique: number, top: bins });
const boxCheck = fields({ low: number, q1: number, median: number, q3: number, high: number });
const chartBins = array(fields({ label: string, count: number, percent: number }));
const chartCheck: Check = value => {
  if (!fields({ kind: string, reason: string })(value)) return false;
  const chart = value as Record<string, unknown>;
  if (["binary_bar", "frequency_bar", "ordered_bar", "top_categories", "value_counts", "dominant_share", "missingness", "histogram", "time_series", "target_rate"].includes(chart.kind as string)) return chartBins(chart.data);
  if (chart.kind === "boxplot") return boxCheck(chart.box);
  if (chart.kind === "summary") return number(chart.cardinality) && array(string)(chart.examples);
  if (chart.kind === "target_boxplot") return fields({ low: number, high: number })(chart.domain) && array(fields({ label: string, box: boxCheck, sample_size: number }))(chart.groups);
  if (chart.kind === "target_distribution") return array(fields({ label: string, data: chartBins, sample_size: number }))(chart.groups);
  return false;
};
const featureEdaCheck: Check = value => {
  if (!fields({ column: string, missing_count: number, missing_pct: number,
    detected_type: string, recommended_visualization: array(chartCheck), relationship_note: string, target_type: string })(value)) return false;
  const feature = value as Record<string, unknown>;
  if (feature.detected_type === "ID_LIKE" && feature.distribution === undefined) {
    return (feature.recommended_visualization as { kind: string }[]).some(chart => chart.kind === "summary");
  }
  return bins(feature.distribution);
};
const matrix = fields({ columns: array(string), values: array(array(nullable(number))), limit: number, eligible: number });
const edaCheck = fields({ shape: shapeCheck, features: array(featureCheck), missing_pct: number, duplicate_count: number,
  duplicate_examples: rows, pearson: matrix, cramers_v: matrix, relation_sample_size: number,
  target_relations: array(fields({ column: string, method: nullable(string), effect: nullable(number), sample_size: number, note: nullable(string) })) });
const uploadCheck = fields({ session_id: string, filename: string, shape: shapeCheck, columns: array(string), dtypes: record(string), preview: rows });
const nestedMetric: Check = value => number(value) || array(array(number))(value);
const nestedCheck: Check = value => fields({ protocol: string, task: string, split_mode: string, scoring_inner: string,
  classes: nullable(array(cell)), predicted_row_positions: array(count), unscored_row_positions: array(count),
  global_metrics: record(nestedMetric), outer_summary: record(fields({ mean: number, std: nullable(number), folds: count })),
  outer_folds: array(fields({ fold: count, train_row_positions: array(count), validation_row_positions: array(count),
    predictions: array(cell), targets: array(cell), metrics: record(nestedMetric),
    selected_config: fields({ preprocessing: object, model: object }), selected_features: array(string),
    inner_search: array(fields({ preprocessing: object, model: object, scores: array(number), mean: number })) })) })(value)
  && (value as NestedResult).protocol === "nested_cross_validation"
  && (value as NestedResult).outer_folds.every(fold => fold.predictions.length === fold.validation_row_positions.length
    && fold.targets.length === fold.validation_row_positions.length)
  && new Set((value as NestedResult).predicted_row_positions).size === (value as NestedResult).predicted_row_positions.length;

/** Reject malformed successful responses before committing them to workflow state. */
export function validateResponse(path: string, value: unknown): void {
  const route = path.split("/").filter(Boolean);
  const checks: Record<string, Check> = {
    config: fields({ mode: string, target: nullable(string), task: string }),
    capabilities: fields({ defaults: object, options: fields({ properties: record(object) }), formats: array(string),
      evaluation: optional(fields({ properties: fields({ family: fields({ enum: array(string), default: string }),
        folds: fields({ minimum: number, maximum: number, default: number }) }) })) }),
    analyze: fields({ features: array(fields({ column: string, type: string, is_target: boolean, anomalies: array(string), action: string, label: string,
      justification: fields({ raison: string, preuve: string, explication: string }), alternatives: array(fields({ value: string, label: string })),
      steps: record(fields({ action: string, measure: nullable(string), threshold: nullable(string), explanation: nullable(string), confidence: nullable(string),
        choices: array(fields({ value: string, label: string, parameters: nullable(string), help: string })) })) })) }),
    preview: fields({ status: string, before_chart: bins, after_charts: array(fields({ title: string, data: bins })), notes: array(string), table: rows }),
    "selection-preview": selectionPreviewCheck,
    preprocess: fields({ input_shape: shapeCheck, train_shape: shapeCheck, test_shape: shapeCheck, full_shape: shapeCheck,
      output_columns: array(string), warnings: array(string), downloads: array(string), worker: fields({ elapsed_seconds: number, peak_rss_bytes: number }) }),
    evaluate: fields({ family: string, scoring: string, folds: number, results: array(fields({ scaler: string, mean: number, std: number, scores: array(number) })) }),
    "evaluate-nested": nestedCheck,
    report: fields({ summary: object, features: array(fields({ column: string, type: string, applied: string, dropped: boolean,
      outputs: array(string), parameters: object, drop_reason: nullable(string), leakage: nullable(number) })) }),
    "results-eda": edaCheck,
    "export-excel": fields({ download: string }),
    session: fields({ closed: boolean }),
    upload: uploadCheck,
    eda: route.length > 2 ? featureEdaCheck : edaCheck,
  };
  if (checks[route[0]] && !checks[route[0]](value)) throw new ApiError("Réponse du serveur incompatible ou incomplète. Réessayez l’opération.", 502);
}

async function checkedFetch(path: string, init?: RequestInit) {
  let response: Response;
  try { response = await fetch(apiRoot + "/api" + path, { ...init, signal: AbortSignal.timeout(660_000) }); }
  catch (reason) {
    throw new ApiError(reason instanceof Error && reason.name === "TimeoutError"
      ? "Le délai de la requête est dépassé. Réessayez l’opération."
      : "Serveur local inaccessible. Lancez python serve.py puis réessayez.", 0);
  }
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new ApiError(typeof error?.detail === "string" ? error.detail : `Le serveur n’a pas pu traiter la demande (HTTP ${response.status}).`, response.status);
  }
  return response;
}

export async function downloadFile(sessionId: string, kind: string): Promise<void> {
  const response = await checkedFetch(`/download/${encodeURIComponent(sessionId)}/${encodeURIComponent(kind)}`);
  const blob = await response.blob();
  if (!blob.size) throw new ApiError("Le fichier exporté est vide.", 502);
  const url = URL.createObjectURL(blob);
  try {
    const link = document.createElement("a");
    link.href = url;
    link.download = response.headers.get("Content-Disposition")?.match(/filename="([^"]+)"/)?.[1] ?? kind;
    link.click();
  } finally { setTimeout(() => URL.revokeObjectURL(url), 1000); }
}

/** Upload a table while reporting actual transmitted bytes, separately from parsing. */
export function uploadTable(file: File, onProgress: (percent: number) => void): Promise<Upload> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", apiRoot + "/api/upload");
    request.responseType = "json";
    request.timeout = 660_000;
    request.upload.onprogress = event => {
      if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100));
    };
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        try { validateResponse("/upload", request.response); resolve(request.response); }
        catch (reason) { reject(reason); }
      } else reject(new ApiError(typeof request.response?.detail === "string" ? request.response.detail : "Import impossible. Vérifiez le format et la taille du fichier.", request.status));
    };
    request.onerror = () => reject(new Error("Serveur local inaccessible. Lancez python serve.py puis réessayez."));
    request.ontimeout = () => reject(new Error("Le délai d’import est dépassé. Vérifiez les limites du serveur."));
    const form = new FormData();
    form.append("file", file);
    request.send(form);
  });
}

/** Fetch one JSON endpoint, preserving a readable error from the API. */
export async function api<T>(path: string, body?: FormData | object, method?: string): Promise<T> {
  const response = await checkedFetch(path, body instanceof FormData
    ? { method: "POST", body }
    : body ? { method: method ?? "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : { method: method ?? "GET" });
  let value: T;
  try { value = await response.json(); }
  catch { throw new ApiError("Réponse JSON du serveur illisible.", 502); }
  validateResponse(path, value);
  return value;
}
