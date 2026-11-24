import { useState } from "react";
import { effectiveChoice } from "./api";
import type { Choices, Choice, Recommendation, TransformationReport } from "./api";
import { DataTable } from "./components";
import { ScientificScene } from "./ScientificScene";
import type { SceneNode } from "./ScientificScene";
import { operations } from "./workflow";
import { StepEditor } from "./StepEditor";
import { actionLabel, operationLabel } from "./labels";
import { RecommendationExplanation } from "./RecommendationExplanation";

/** Inspect engine operation groups and edit the same overrides used by execution. */
export function PipelinePanel({ rows, choices, busy, onChoice, workflow, onNavigate }: {
  rows: Recommendation[]; choices: Choices; busy: boolean;
  onChoice: (column: string, value: Choice) => void;
  workflow: SceneNode[]; onNavigate: (index: number) => void;
}) {
  const [operation, setOperation] = useState(0);
  const [scope, setScope] = useState("operations");
  const current = operations[operation];
  const features = rows.filter(row => !row.is_target && current.keys.some(key => row.steps[key]));
  return <>
    <section className="pipeline-stage">
      <div className="panel-head">
        <h2>Architecture du traitement</h2>
        <select aria-label="Vue du pipeline" value={scope} onChange={event => setScope(event.target.value)}>
          <option value="operations">Opérations du moteur</option><option value="workflow">Parcours en six étapes</option>
        </select>
      </div>
      <ScientificScene label={scope === "workflow" ? "Parcours de préparation" : "Opérations Python"}
        nodes={scope === "workflow" ? workflow : operations.map((item, index) => ({
          label: item.label, state: index === operation ? "active" : "pending",
        }))}
        onSelect={scope === "workflow" ? onNavigate : setOperation} />
      <p className="muted">Sélectionnez un nœud. Glissez pour orienter la scène. Les boutons offrent le même accès au clavier.</p>
    </section>
    <section className="panel">
      <h2>{current.label}</h2><p>{current.text}</p>
      <p className="muted">{features.length} variables concernées, dont {features.filter(row => choices[row.column] === "none").length} conservées brutes.</p>
      <p className="muted">Les alternatives rapides et l’éditeur par opération alimentent les mêmes choix. « Ne rien appliquer » conserve la variable brute dans l’export.</p>
      <div className="table-scroll" tabIndex={0}><table>
        <thead><tr><th>Variable</th><th>Opérations recommandées</th><th>Choix pour cette variable</th></tr></thead>
        <tbody data-motion-key={current.label}>{features.map(row => <tr key={row.column}>
          <th scope="row">{row.column}<small>{row.type}</small></th>
          <td>{(row.action === "DROP" ? ["type"] : current.keys.filter(key => row.steps[key])).map(key => <details key={key}>
            <summary title={`${key} / ${row.steps[key]?.action ?? "DROP"}`}>{operationLabel(key)} : {actionLabel(row.action === "DROP" ? "DROP" : row.steps[key].action)}</summary>
            {row.action !== "DROP" && <><p>{row.steps[key].measure}</p><p>{row.steps[key].explanation}</p></>}
          </details>)}<RecommendationExplanation row={row} choice={choices[row.column] ?? "recommended"} /></td>
          <td><select aria-label={`Configuration pour ${row.column}`} disabled={busy}
            value={effectiveChoice(choices, row.column)} onChange={event => onChoice(row.column, event.target.value)}>
            <option value="custom" disabled>Personnalisé par opération</option><option value="recommended">Recommandation</option><option value="none">Ne rien appliquer</option>
            {row.alternatives.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select></td>
        </tr>)}</tbody>
      </table></div>
      {!features.length && <p>Aucune variable éligible à cette opération dans le jeu importé.</p>}
    </section>
    <StepEditor rows={rows} choices={choices} busy={busy} onChoice={onChoice} />
  </>;
}

/** Show learned transformations from the server report, including retained raw features. */
export function AppliedReport({ report }: { report: TransformationReport }) {
  const imbalance = report.summary["Gestion du déséquilibre"] as { method?: string; synthetic_rows?: number; before?: { counts?: Record<string, number> }; test?: { counts?: Record<string, number> } } | null;
  const counts = (value?: Record<string, number>) => value ? `{${Object.entries(value).map(([key, count]) => `${key}:${count}`).join(", ")}}` : "—";
  return <section className="panel">
    <h2>Transformations appliquées</h2>
    {!!imbalance && <DataTable rows={[{
      Variable: "Cible (train)", Type: "Rééquilibrage", Choix: `${imbalance.method === "none" ? "Aucun rééquilibrage" : imbalance.method ?? "Aucun rééquilibrage"} · train ${counts(imbalance.before?.counts)} · test intact ${counts(imbalance.test?.counts)} · ${imbalance.synthetic_rows ?? 0} ligne(s) synthétique(s)`,
      Appliqué: imbalance.method ?? "none",
    }]} />}
    <DataTable rows={report.features.map(feature => ({
      Variable: feature.column, Type: feature.type, Choix: typeof feature.choice === "string" ? feature.choice : JSON.stringify(feature.choice), Appliqué: feature.applied,
      Supprimée: feature.dropped, Motif: feature.drop_reason, "Colonnes de sortie": feature.outputs.join(", "),
      "Fuite suspectée": feature.leakage,
    }))} />
    <details><summary>Paramètres appris sur le train</summary>
      <DataTable rows={report.features.map(feature => ({
        Variable: feature.column, Paramètres: JSON.stringify(feature.parameters),
      }))} />
    </details>
    <details><summary>Résumé complet et diagnostics du moteur</summary>
      <pre className="json-report">{JSON.stringify({ summary: report.summary, core: report.core }, null, 2)}</pre>
    </details>
    <details><summary>Preuves statistiques détaillées par variable</summary>
      <DataTable rows={report.features.map(feature => ({ Variable: feature.column, Diagnostics: JSON.stringify(feature.diagnostics), Étapes: JSON.stringify(feature.steps) }))} />
    </details>
  </section>;
}
