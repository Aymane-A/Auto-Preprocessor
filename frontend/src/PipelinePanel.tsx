import { useState } from "react";
import { effectiveChoice } from "./api";
import type { Recommendation, TransformationReport } from "./api";
import { DataTable } from "./components";
import { ScientificScene } from "./ScientificScene";
import type { SceneNode } from "./ScientificScene";
import { operations } from "./workflow";

/** Inspect engine operation groups and edit the same overrides used by execution. */
export function PipelinePanel({ rows, choices, busy, onChoice, workflow, onNavigate }: {
  rows: Recommendation[]; choices: Record<string, string>; busy: boolean;
  onChoice: (column: string, value: string) => void;
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
      <p className="muted">Le choix alternatif remplace une opération pour la variable. Les autres étapes suivent le moteur.
        « Ne rien appliquer » conserve la variable brute dans l’export.</p>
      <div className="table-scroll" tabIndex={0}><table>
        <thead><tr><th>Variable</th><th>Opérations recommandées</th><th>Choix pour cette variable</th></tr></thead>
        <tbody data-motion-key={current.label}>{features.map(row => <tr key={row.column}>
          <th scope="row">{row.column}<small>{row.type}</small></th>
          <td>{current.keys.filter(key => row.steps[key]).map(key => <details key={key}>
            <summary>{key} / {row.steps[key].action}</summary>
            <p>{row.steps[key].measure}</p><p>{row.steps[key].explanation}</p>
          </details>)}</td>
          <td><select aria-label={`Configuration pour ${row.column}`} disabled={busy}
            value={effectiveChoice(choices, row.column)} onChange={event => onChoice(row.column, event.target.value)}>
            <option value="recommended">Recommandation</option><option value="none">Ne rien appliquer</option>
            {row.alternatives.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select></td>
        </tr>)}</tbody>
      </table></div>
      {!features.length && <p>Aucune variable éligible à cette opération dans le jeu importé.</p>}
    </section>
  </>;
}

/** Show learned transformations from the server report, including retained raw features. */
export function AppliedReport({ report }: { report: TransformationReport }) {
  return <section className="panel">
    <h2>Transformations appliquées</h2>
    <DataTable rows={report.features.map(feature => ({
      Variable: feature.column, Type: feature.type, Choix: feature.choice, Appliqué: feature.applied,
      Supprimée: feature.dropped, Motif: feature.drop_reason, "Colonnes de sortie": feature.outputs.join(", "),
      "Fuite suspectée": feature.leakage,
    }))} />
    <details><summary>Paramètres appris sur le train</summary>
      <DataTable rows={report.features.map(feature => ({
        Variable: feature.column, Paramètres: JSON.stringify(feature.parameters),
      }))} />
    </details>
  </section>;
}
