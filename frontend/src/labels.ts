export const operationLabels: Record<string, string> = {
  clean_missing_tokens: "Faux manquants", clean_variants: "Variantes de modalités",
  clean_numeric_text: "Nombres écrits en texte", clean_dates: "Formats de dates",
  clean_impossible: "Valeurs impossibles", duplicates: "Lignes dupliquées",
  type: "Type / suppression",
};

export const actionLabels: Record<string, string> = {
  TO_NAN: "Convertir en valeur manquante", MERGE: "Fusionner", KEEP: "Garder tel quel",
  IGNORE: "Ignorer", DROP: "Supprimer",
};

export const operationLabel = (value: string) => operationLabels[value] ?? value;
export const actionLabel = (value: string) => actionLabels[value] ?? value;
