import type { Choice, Preview, Recommendation } from "./api";
import { actionLabel, operationLabel } from "./labels";

const definitions: Record<string, string> = {
  DROP: "Retirer une variable qui ne doit pas alimenter le modèle.",
  MEDIAN: "Remplacer les valeurs manquantes par la médiane calculée sur le train.",
  MEAN: "Remplacer les valeurs manquantes par la moyenne calculée sur le train.",
  MODE: "Remplacer les valeurs manquantes par la valeur la plus fréquente du train.",
  KNN: "Estimer les valeurs manquantes à partir de lignes voisines du train.",
  WINSORIZE: "Ramener les valeurs extrêmes à des bornes apprises sur le train.",
  IQR: "Repérer les valeurs au-delà des bornes définies par l’écart interquartile.",
  ONEHOT: "Créer une colonne indicatrice pour chaque modalité retenue.",
  ORDINAL: "Remplacer les modalités par des codes qui respectent un ordre connu.",
  LABEL: "Associer un code numérique à chaque modalité.",
  BINARY: "Représenter les deux modalités d’une variable binaire par des codes numériques.",
  STANDARD: "Centrer puis réduire une variable à partir du train.",
  MINMAX: "Ramener une variable dans un intervalle appris sur le train.",
  ROBUST: "Mettre à l’échelle avec la médiane et l’écart interquartile du train.",
  LOG1P: "Appliquer le logarithme de un plus la valeur lorsque son domaine le permet.",
  NONE: "Conserver les valeurs sans cette transformation.",
};

function text(value: unknown): string | null {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : null;
  return typeof value === "string" || typeof value === "boolean" ? String(value) : null;
}

function evidence(row: Recommendation): string[] {
  const diagnostics = row.diagnostics ?? {};
  const keys: [string, string][] = [
    ["manquants_pct", "Valeurs manquantes (%)"], ["uniques", "Valeurs distinctes"],
    ["ratio_uniques_lignes", "Part des valeurs uniques"], ["seuil_identifiant_uniques", "Seuil identifiant"],
    ["outliers_pct", "Valeurs extrêmes (%)"], ["skew", "Asymétrie"],
    ["score_fuite", "Score de fuite CV"], ["cramer_cible", "V de Cramér"],
    ["information_mutuelle_cible", "Information mutuelle"],
  ];
  return keys.flatMap(([key, label]) => {
    const value = text(diagnostics[key]);
    return value === null ? [] : [`${label} : ${value}`];
  });
}

function alternativeEvidence(row: Recommendation, value: string): string | null {
  const action = value.includes(":") ? value.split(":").at(-1)! : value;
  for (const detail of Object.values(row.steps)) {
    const comparison = detail.comparison?.[action];
    if (!comparison || typeof comparison !== "object") continue;
    const data = comparison as Record<string, unknown>;
    const notes = [data.avertissement, data.avantage].map(text).filter((item): item is string => item !== null);
    if (data.applicable === false) notes.unshift("Non applicable aux données observées");
    if (typeof data.largeur_estimee === "number") notes.push(`${data.largeur_estimee} colonne(s) estimée(s)`);
    if (text(data.inconnu) !== null) notes.push(`Modalité inconnue : ${text(data.inconnu)}`);
    if (text(data.valeur) !== null) notes.push(`Valeur calculée : ${text(data.valeur)}`);
    return notes.length ? notes.join(" · ") : null;
  }
  return null;
}

/** Explain the recommendation from the engine's actual measures and available examples. */
export function RecommendationExplanation({ row, choice, preview }: {
  row: Recommendation; choice: Choice; preview?: Preview | null;
}) {
  const steps = row.action === "DROP" ? Object.entries(row.steps).filter(([name]) => name === "type" || name === "selection") : Object.entries(row.steps);
  const primary = steps.find(([, detail]) => detail.action === row.action) ?? steps[0];
  const definition = (row.action === "DROP" && row.type === "DROP_ID" ? "Écarter un identifiant dont les valeurs servent à reconnaître les lignes plutôt qu'à prédire." :
    primary && definitions[primary[1].action]) ||
    (
      `${actionLabel(row.action)} : opération proposée par le moteur pour cette variable.`);
  const examples = Array.isArray(row.diagnostics?.exemples) ? row.diagnostics.exemples.map(text).filter((value): value is string => value !== null).slice(0, 3) : [];
  const chosen = typeof choice === "string" ? choice : "Choix personnalisé par opération";
  const isRecommended = chosen === "recommended";
  return <details className="recommendation-explanation">
    <summary>Pourquoi cette recommandation ?</summary>
    <div className="explanation-content">
      <section><h4>Qu'est-ce que c'est ?</h4><p>{definition}</p></section>
      <section><h4>Pourquoi cette recommandation ?</h4><p>{row.justification.raison}</p></section>
      <section><h4>Comment ça fonctionne ?</h4>{steps.length ? <ol>{steps.map(([name, detail]) => <li key={name}>
        <strong>{operationLabel(name)} · {actionLabel(detail.action)}</strong> — {detail.help || detail.explanation || "Le moteur n'a pas fourni de détail supplémentaire pour cette étape."}
      </li>)}</ol> : <p>Aucune étape détaillée fournie par le moteur.</p>}</section>
      <section><h4>Exemple concret</h4>{examples.length ? <p>Valeurs observées pour {row.column} : {examples.join(" · ")}.</p> : <p>Aucune valeur d'exemple disponible dans le diagnostic.</p>}
        {preview?.samples?.length ? <p>Exemple après exécution de l’aperçu : {JSON.stringify(preview.samples[0].values)}.</p> :
          <p>La valeur après transformation est disponible dans l’aperçu, lorsqu’il est calculé.</p>}</section>
      <section><h4>Impact sur les données</h4><p>{row.action === "DROP" ?
        "La variable est recommandée pour retrait de l'entrée du modèle. Les autres variables ne sont pas retirées par cette recommandation." :
        "Cette opération concerne cette variable. Les valeurs et colonnes finales seront déterminées lors du fit sur le train."}</p></section>
      <section><h4>Avantages et risques</h4><p>{primary?.[1].help || row.justification.explication || "Le diagnostic ne fournit pas d'évaluation supplémentaire des effets secondaires."} {row.action === "DROP" ?
        "Le retrait peut réduire un signal parasite, mais une information utile peut être perdue ; vérifiez la disponibilité de la variable au moment de la prédiction." :
        "Le bénéfice dépend du modèle et de la distribution future ; comparez avec une évaluation sur des données tenues à l’écart."}</p></section>
      <section><h4>Pourquoi les autres méthodes ne sont pas choisies ?</h4>{row.alternatives.length ? <ul>
        {row.alternatives.slice(0, 5).map(item => <li key={item.value}><strong>{item.label}</strong> : {alternativeEvidence(row, item.value) ||
          "Alternative disponible ; le moteur ne fournit pas de comparaison chiffrée pour cette action."}</li>)}
      </ul> : <p>Aucune alternative applicable n'a été fournie pour cette variable.</p>}</section>
      <section><h4>Preuves statistiques</h4><p>{row.justification.preuve || "Aucune preuve chiffrée fournie."}</p>
        <ul>{steps.map(([name, detail]) => <li key={name}>{operationLabel(name)} : {detail.measure || "mesure indisponible"} · seuil {detail.threshold || "indisponible"}</li>)}
          {evidence(row).map(item => <li key={item}>{item}</li>)}</ul></section>
      <p className="explanation-status"><strong>Votre choix :</strong> {isRecommended ? "Suivre la recommandation" : chosen === "none" ? "Conserver la colonne brute" : chosen}.
        <br /><strong>Exécution :</strong> recommandation calculée ; traitement non encore exécuté.</p>
    </div>
  </details>;
}
