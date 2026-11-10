/** Presentational tables and accessible, interactive data charts. */
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Bin, Box, Cell, Feature, Matrix, Row, Shape } from "./api";
import { AnimatedNumber, ChartTooltip, motionTiming } from "./animation";

const numericFormat = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 4 });

/** Format numbers and missing values consistently across the workflow. */
export function display(value: Cell | undefined): string {
  return value == null ? "—" : typeof value === "number" ? numericFormat.format(value) :
    typeof value === "boolean" ? (value ? "Oui" : "Non") : value;
}

/** Format a dataset shape as rows by columns. */
export function shape(value: Shape) {
  return value.map(display).join(" × ");
}

/** Show a scrollable data table with the input column order. */
export function DataTable({ rows, columns }: { rows: Row[]; columns?: string[] }) {
  const names = columns ?? [...new Set(rows.flatMap(Object.keys))];
  if (!rows.length) return <p className="muted">Aucune ligne à afficher.</p>;
  return <div className="table-scroll" tabIndex={0} data-motion-key={JSON.stringify([rows.length, rows.slice(0, 12)])}><table>
    <thead><tr>{names.map(name => <th key={name} scope="col">{name}</th>)}</tr></thead>
    <tbody>{rows.map((row, index) => <tr key={index}>{names.map(name =>
      <td key={name} className={typeof row[name] === "number" ? "numeric-cell" : undefined}>{display(row[name])}</td>)}</tr>)}</tbody>
  </table></div>;
}

/** Show key dataset measurements in a consistent card. */
export function Metric({ label, value, suffix = "" }: { label: string; value: number | Shape; suffix?: string }) {
  return <div className="metric"><span>{label}</span><strong>{Array.isArray(value)
    ? <><AnimatedNumber value={value[0]} digits={0} /> × <AnimatedNumber value={value[1]} digits={0} /></>
    : <AnimatedNumber value={value} digits={suffix ? 4 : 0} />}{suffix}</strong></div>;
}

/** Draw a responsive frequency chart with exact values in its tooltip. */
export function Distribution({ data, unit = "Effectif", color = "var(--green)" }: { data: Bin[]; unit?: string; color?: string }) {
  if (!data.length) return <p className="empty-chart">Aucune valeur à tracer.</p>;
  return <div className="chart" data-motion-key={JSON.stringify(data)}>
    <ResponsiveContainer width="100%" height="100%" minWidth={1}>
      <BarChart data={data} margin={{ left: -12, right: 12, top: 12, bottom: 12 }}>
        <CartesianGrid vertical={false} stroke="var(--line)" />
        <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted)" }} tickMargin={8}
          tickFormatter={value => String(value).length > 16 ? String(value).slice(0, 15) + "…" : value} />
        <YAxis tick={{ fontSize: 11, fill: "var(--muted)" }} />
        <Tooltip cursor={{ fill: "var(--pale)" }} animationDuration={motionTiming.interaction * 1000}
          content={({ active, label, payload }) => <ChartTooltip active={active} label={label}>
            {payload.map((item, index) => <p key={index}>{item.name} · {typeof item.value === "number" || typeof item.value === "string" ? display(item.value) : item.value?.join(" – ")}</p>)}
          </ChartTooltip>} />
        <Bar dataKey="count" name={unit} fill={color} radius={[3, 3, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  </div>;
}

/** Draw a boxplot with quartiles, median, and observed Tukey whiskers. */
export function Boxplot({ box }: { box: Box }) {
  const x = (value: number) => box.high === box.low ? 300 : 30 + (value - box.low) / (box.high - box.low) * 540;
  const values = Object.entries(box);
  const label = values.map(([key, value]) => key + " " + display(value)).join(", ");
  return <figure className="boxplot">
    <svg viewBox="0 0 600 95" role="img" aria-label={label}>
      <title>{label}</title>
      <line x1={x(box.low)} x2={x(box.high)} y1={40} y2={40} stroke="var(--control)" strokeWidth={2} />
      {[box.low, box.high].map((value, index) => <line key={index} x1={x(value)} x2={x(value)}
        y1={29} y2={51} stroke="var(--control)" strokeWidth={2} />)}
      <rect x={x(box.q1)} y={21} width={Math.max(1, x(box.q3) - x(box.q1))} height={38}
        fill="var(--pale)" stroke="var(--green)" strokeWidth={2} />
      <line x1={x(box.median)} x2={x(box.median)} y1={21} y2={59} stroke="var(--green)" strokeWidth={3} />
      <text x={30} y={82} fill="var(--muted)">{display(box.low)}</text>
      <text x={570} y={82} textAnchor="end" fill="var(--muted)">{display(box.high)}</text>
    </svg>
    <figcaption>Q1 {display(box.q1)} / Médiane {display(box.median)} / Q3 {display(box.q3)}</figcaption>
  </figure>;
}

/** Show an association matrix with hover labels and accessible values. */
export function Heatmap({ matrix, title }: { matrix: Matrix; title: string }) {
  return <section className="panel">
    <h3>{title}</h3>
    <p className="muted">{matrix.columns.length} / {matrix.eligible} colonnes éligibles · maximum {matrix.limit}, dans l’ordre du fichier</p>
    {!matrix.columns.length ? <p>Aucune colonne éligible.</p> : <div className="table-scroll" tabIndex={0}>
      <table className="heatmap"><thead><tr><th scope="col">Feature</th>{matrix.columns.map(name =>
        <th key={name} scope="col">{name}</th>)}</tr></thead>
        <tbody>{matrix.columns.map((name, i) => <tr key={name}><th scope="row">{name}</th>
          {matrix.values[i].map((value, j) => <td key={j}
            title={name + " × " + matrix.columns[j] + " · " + display(value)}
            style={{ background: value === null ? "var(--surface-quiet)" : value < 0
              ? `color-mix(in srgb, var(--chart-negative) ${Math.abs(value) * 62}%, var(--surface))` : `color-mix(in srgb, var(--chart-positive) ${value * 62}%, var(--surface))` }}>
            {value === null ? "—" : value.toFixed(2)}</td>)}</tr>)}</tbody>
      </table>
    </div>}
  </section>;
}

export type SortKey = "column" | "type" | "missing_pct" | "unique" | "mean" | "std" | "min" | "max" | "skew" | "kurtosis";

/** Filter and sort feature metadata, keeping missing statistics at the end. */
export function visibleFeatures(features: Feature[], query: string, sort: SortKey, descending: boolean) {
  const search = query.trim().toLocaleLowerCase("fr");
  return features.filter(feature => (feature.column + " " + feature.type).toLocaleLowerCase("fr").includes(search))
    .sort((left, right) => {
      const a = left[sort], b = right[sort];
      if (a == null || b == null) return Number(a == null) - Number(b == null);
      const comparison = typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b), "fr");
      return descending ? -comparison : comparison;
    });
}

export { effectiveChoice } from "./api";
