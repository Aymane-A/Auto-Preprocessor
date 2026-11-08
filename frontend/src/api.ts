/** Types and requests shared by the four workflow steps. */
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
};
export type Result = {
  input_shape: Shape; train_shape: Shape; test_shape: Shape; full_shape: Shape;
  output_columns: string[]; warnings: string[]; downloads: string[];
  worker: { elapsed_seconds: number; peak_rss_bytes: number };
};
export const apiRoot = import.meta.env.DEV ? "http://127.0.0.1:8000" : "";

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
