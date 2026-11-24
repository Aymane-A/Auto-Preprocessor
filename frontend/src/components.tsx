/** Presentational tables and accessible, interactive data charts. */
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Bin, Box, Cell, Feature, Matrix, Row, Shape } from "./api";
import { AnimatedNumber, ChartTooltip, motionTiming } from "./animation";

const numericFormat = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 4 });
const percentFormat = new Intl.NumberFormat("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
export const percentDisplay = (value: number | null | undefined) => value == null ? "—" : percentFormat.format(value);

/** Format numbers and missing values consistently across the workflow. */
export function display(value: Cell | undefined): string {
  return value == null ? "—" : typeof value === "number" ? (!Number.isFinite(value) ? "indisponible" :
    value !== 0 && Math.abs(value) < .0001 ? value.toExponential(3) : numericFormat.format(value)) :
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
       <td key={name} className={typeof row[name] === "number" ? "numeric-cell" : undefined}>{typeof row[name] === "number" && (name === "%" || name.includes("(%)")) ? percentDisplay(row[name] as number) : display(row[name])}</td>)}</tr>)}</tbody>
  </table></div>;
}

/** Show key dataset measurements in a consistent card. */
export function Metric({ label, value, suffix = "" }: { label: string; value: number | Shape; suffix?: string }) {
  return <div className="metric"><span>{label}</span><strong>{Array.isArray(value)
    ? <><AnimatedNumber value={value[0]} digits={0} /> × <AnimatedNumber value={value[1]} digits={0} /></>
     : <AnimatedNumber value={value} digits={suffix.includes("%") ? 1 : suffix ? 4 : 0} />}{suffix}</strong></div>;
}

/** Draw a responsive frequency chart with exact values in its tooltip. */
export function Distribution({ data, unit = "Effectif", color = "var(--green)", horizontal = false, rate = false, histogram = false }: { data: Bin[]; unit?: string; color?: string; horizontal?: boolean; rate?: boolean; histogram?: boolean }) {
  if (!data.length) return <p className="empty-chart">Aucune valeur à tracer.</p>;
  return <div className="chart" data-motion-key={JSON.stringify(data)} style={horizontal ? { height: Math.max(280, data.length * 28) } : undefined}>
    <ResponsiveContainer width="100%" height="100%" minWidth={1}>
      <BarChart data={data} layout={horizontal ? "vertical" : "horizontal"} barCategoryGap={histogram ? 0 : undefined} margin={{ left: horizontal ? 0 : -12, right: 12, top: 12, bottom: 12 }}>
        <CartesianGrid vertical={false} stroke="var(--line)" />
        <XAxis dataKey={horizontal ? undefined : "label"} type={horizontal ? "number" : "category"} domain={horizontal && rate ? [0, 100] : undefined}
          tick={{ fontSize: 11, fill: "var(--muted)" }} tickMargin={8}
          tickFormatter={value => String(value).length > 16 ? String(value).slice(0, 15) + "…" : value} />
        <YAxis dataKey={horizontal ? "label" : undefined} type={horizontal ? "category" : "number"} domain={!horizontal && rate ? [0, 100] : undefined}
          width={horizontal ? 145 : 60} tick={{ fontSize: 11, fill: "var(--muted)" }} />
        <Tooltip cursor={{ fill: "var(--pale)" }} animationDuration={motionTiming.interaction * 1000}
          content={({ active, label, payload }) => <ChartTooltip active={active} label={label}>
            {payload.map((item, index) => <p key={index}>{item.name} · {typeof item.value === "number" || typeof item.value === "string" ? display(item.value) : item.value?.join(" – ")}</p>)}
            {typeof payload[0]?.payload?.percent === "number" && <p>{percentDisplay(payload[0].payload.percent)} %
              {typeof payload[0].payload.total === "number" && ` · ${display(payload[0].payload.count)} / ${display(payload[0].payload.total)}`}</p>}
          </ChartTooltip>} />
        <Bar dataKey={rate ? "percent" : "count"} name={unit} fill={color} radius={[3, 3, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  </div>;
}

/** Plot chronological monthly counts selected and aggregated by the backend. */
export function TimeSeries({ data }: { data: Bin[] }) {
  return <div className="chart"><ResponsiveContainer width="100%" height="100%" minWidth={1}>
    <LineChart data={data} margin={{ left: -12, right: 12, top: 12, bottom: 12 }}>
      <CartesianGrid vertical={false} stroke="var(--line)" />
      <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted)" }} />
      <YAxis tick={{ fontSize: 11, fill: "var(--muted)" }} />
      <Tooltip content={({ active, label, payload }) => <ChartTooltip active={active} label={label}>
        {payload.map((item, i) => <p key={i}>{display(item.value as number)} observations · {percentDisplay(item.payload?.percent)} %</p>)}
      </ChartTooltip>} />
      <Line type="linear" dataKey="count" name="Effectif" stroke="var(--green)" strokeWidth={2} dot={false} isAnimationActive={false} />
    </LineChart>
  </ResponsiveContainer></div>;
}

/** Draw a boxplot with quartiles, median, and observed Tukey whiskers. */
export function Boxplot({ box, domain }: { box: Box; domain?: { low: number; high: number } }) {
  const axis = domain ?? box;
  const x = (value: number) => axis.high === axis.low ? 300 : 30 + (value - axis.low) / (axis.high - axis.low) * 540;
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
      <text x={30} y={82} fill="var(--muted)">{display(axis.low)}</text>
      <text x={570} y={82} textAnchor="end" fill="var(--muted)">{display(axis.high)}</text>
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
