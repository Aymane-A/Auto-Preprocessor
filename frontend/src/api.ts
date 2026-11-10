/** Types and requests shared by the six workflow stages. */
export type Cell = string | number | boolean | null;
export type Row = Record<string, Cell>;
export type Shape = [number, number];
export type Bin = { label: string; count: number };
export type Matrix = { columns: string[]; values: (number | null)[][]; limit: number; eligible: number };
export type Box = { low: number; q1: number; median: number; q3: number; high: number };
export type Bounds = { count: number | null; lower: number | null; upper: number | null };
export type Feature = {
  column: string; dtype: string; type: string; missing_pct: number; missing_count: number;
  unique: number; mean?: number; std?: number; min?: number; q1?: number; median?: number; q3?: number;
  max?: number; skew?: number; kurtosis?: number; top: Bin[]; boxplot?: Box;
  text_length?: Record<string, number | null>; iqr?: Bounds; robust_z?: Bounds;
};
export type Upload = { session_id: string; filename: string; shape: Shape; columns: string[]; dtypes: Record<string, string>; preview: Row[] };
export type Eda = {
  shape: Shape; features: Feature[]; missing_pct: number; duplicate_count: number; duplicate_examples: Row[];
  pearson: Matrix; cramers_v: Matrix; relation_sample_size: number;
  target_relations: { column: string; method: string | null; effect: number | null; sample_size: number; note: string | null }[];
};
export type Recommendation = {
  column: string; type: string; is_target: boolean; anomalies: string[];
  action: string; label: string; justification: { raison: string; preuve: string; explication: string };
  alternatives: { value: string; label: string }[];
  steps: Record<string, { action: string; measure: string; threshold: string; explanation: string; confidence: string }>;
};
export type Preview = {
  status: string; before_chart: Bin[]; after_charts: { title: string; data: Bin[] }[]; notes: string[]; table: Row[];
  rows?: number; output_columns?: string[]; applied?: Record<string, Cell>;
  before?: Record<string, unknown>; after_scaled?: Record<string, unknown>;
  affected?: { missing_filled: number; outside_bounds: number; rows_changed: number; rows_changed_pct: number };
  samples?: { row: Cell; original: Cell; values: Row }[];
};
export type Result = {
  input_shape: Shape; train_shape: Shape; test_shape: Shape; full_shape: Shape;
  output_columns: string[]; warnings: string[]; downloads: string[];
  worker: { elapsed_seconds: number; peak_rss_bytes: number };
};
export type TransformationReport = {
  summary: Record<string, unknown>;
  features: { column: string; type: string; applied: string; choice: string; dropped: boolean;
    drop_reason: string | null; outputs: string[]; parameters: Record<string, unknown>; leakage: number | null }[];
};
export const apiRoot = import.meta.env.DEV ? (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000") : "";

/** Resolve an unspecified feature transformation to its recommendation. */
export function effectiveChoice(choices: Record<string, string>, name: string) {
  return Object.hasOwn(choices, name) ? choices[name] : "recommended";
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
      if (request.status >= 200 && request.status < 300) resolve(request.response);
      else reject(new Error(request.response?.detail ?? "Import impossible. Vérifiez le format et la taille du fichier."));
    };
    request.onerror = () => reject(new Error("Serveur local inaccessible. Lancez python serve.py puis réessayez."));
    request.ontimeout = () => reject(new Error("Le délai d’import est dépassé. Vérifiez les limites du serveur."));
    const form = new FormData();
    form.append("file", file);
    request.send(form);
  });
}

/** Fetch one JSON endpoint, preserving a readable error from the API. */
export async function api<T>(path: string, body?: FormData | object): Promise<T> {
  const response = await fetch(apiRoot + "/api" + path, body instanceof FormData
    ? { method: "POST", body }
    : body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(typeof error?.detail === "string" ? error.detail : "Le serveur n’a pas pu traiter la demande.");
  }
  return response.json();
}
