import { useEffect, useState } from "react";
import type { Choice, Choices, Recommendation, StepChoice } from "./api";
import { resolvedChoice } from "./api";
import { actionLabel, operationLabel } from "./labels";

function expanded(row: Recommendation, choice: Choice): Record<string, StepChoice> {
  if (typeof choice !== "string") return choice;
  if (choice === "none" || choice === "recommended") return {};
  const [step, action] = choice.split(":");
  if (action) return { [step]: { action, parameters: "" } };
  const aliases: Record<string, [string, string]> = {
    SCALE_STANDARD: ["scaling", "STANDARD"], SCALE_ROBUST: ["scaling", "ROBUST"], SCALE_MINMAX: ["scaling", "MINMAX"],
    NO_SCALE: ["scaling", "NONE"], DROP: ["type", "DROP"], IMPUTE_MEDIAN: ["missing", "MEDIAN"],
  };
  const found = aliases[choice] ?? [Object.keys(row.steps).find(key => row.steps[key].choices?.some(item => item.value === choice)), choice];
  return found[0] ? { [found[0]]: { action: found[1]!, parameters: "" } } : {};
}

function StepControl({ name, detail, value, busy, onSave }: {
  name: string; detail: Recommendation["steps"][string]; value?: StepChoice; busy: boolean;
  onSave: (value: StepChoice | null) => void;
}) {
  const [action, setAction] = useState(value?.action ?? "recommended");
  const [parameters, setParameters] = useState(value?.parameters ?? "");
  useEffect(() => { setAction(value?.action ?? "recommended"); setParameters(value?.parameters ?? ""); }, [value?.action, value?.parameters]);
  const selected = detail.choices?.find(item => item.value === action);
  return <form onSubmit={event => { event.preventDefault(); onSave(action === "recommended" ? null : { action, parameters }); }}>
     <fieldset disabled={busy}><div className="step-control-head"><strong title={name}>{operationLabel(name)}</strong><span title={detail.action}>Recommandation · {actionLabel(detail.action)}</span></div>
       <div className="advanced-grid"><label className="field">Action pour {operationLabel(name)}<select aria-label={`Action pour ${name}`} value={action} onChange={event => { setAction(event.target.value); setParameters(""); }}>
        <option value="recommended">Suivre la recommandation</option>
         {detail.choices?.map(item => <option key={item.value} value={item.value}>{actionLabel(item.value) === item.value ? item.label : actionLabel(item.value)}</option>)}
      </select></label>
        {selected?.parameters && <label className="field">Paramètres pour {name}<input aria-label={`Paramètres pour ${name}`} value={parameters}
          placeholder={selected.parameters} onChange={event => setParameters(event.target.value)} /><small>{selected.parameters}</small></label>}
      </div><p className="muted">{selected?.help ?? detail.explanation}</p><small>{detail.measure} · {detail.threshold} · confiance {detail.confidence}</small>
       <button type="submit">Appliquer l’étape {operationLabel(name)}</button>
    </fieldset>
  </form>;
}

export function StepEditor({ rows, choices, busy, onChoice }: {
  rows: Recommendation[]; choices: Choices; busy: boolean; onChoice: (column: string, choice: Choice) => void;
}) {
  const [column, setColumn] = useState("");
  const available = rows.filter(row => !row.is_target);
  const row = available.find(item => item.column === column) ?? available[0];
  if (!row) return null;
  const current = expanded(row, resolvedChoice(choices, row.column));
  return <section className="panel"><details><summary>Configurer plusieurs opérations par variable</summary>
    <p>Chaque étape peut suivre la recommandation ou recevoir un choix explicite. La règle des doublons est globale et doit être identique entre les variables.</p>
    <label className="field">Variable à configurer<select value={row.column} disabled={busy} onChange={event => setColumn(event.target.value)}>
      {available.map(item => <option key={item.column}>{item.column}</option>)}</select></label>
     <div className="step-controls">{Object.entries(row.steps).filter(([name]) => !["nettoyage", "target"].includes(name) && (row.action !== "DROP" || name === "type")).map(([name, detail]) =>
       <details key={row.column + name}><summary title={`${name} / ${current[name]?.action ?? detail.action}`}>{operationLabel(name)} · {actionLabel(current[name]?.action ?? detail.action)}</summary>
        <StepControl name={name} detail={detail} value={current[name]} busy={busy} onSave={value => {
          const next = { ...current };
          if (value) next[name] = value;
          else delete next[name];
          onChoice(row.column, Object.keys(next).length ? next : "recommended");
        }} />
      </details>)}</div>
  </details></section>;
}
