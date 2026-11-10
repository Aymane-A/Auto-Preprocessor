export const stages = [
  { name: "Import des données", title: "Préparez vos données.", text: "Importez un jeu de données et définissez la tâche d’apprentissage.", short: "Import" },
  { name: "Analyse statistique", title: "Explorez vos données.", text: "Inspectez les distributions, les valeurs manquantes et les corrélations.", short: "Analyse" },
  { name: "Recommandations", title: "Examinez les recommandations.", text: "Appuyez vos choix sur les diagnostics et les preuves statistiques du moteur.", short: "Conseils" },
  { name: "Configuration", title: "Configurez le pipeline.", text: "Inspectez les opérations et choisissez les transformations par variable.", short: "Pipeline" },
  { name: "Aperçu et traitement", title: "Vérifiez puis exécutez.", text: "Comparez une transformation et lancez le traitement sur votre jeu de données.", short: "Traitement" },
  { name: "Export des résultats", title: "Récupérez vos résultats.", text: "Consultez les opérations appliquées et exportez les données préparées.", short: "Export" },
] as const;

export const operations = [
  { label: "Nettoyage", keys: ["nettoyage", "clean_missing_tokens", "clean_variants", "clean_numeric_text", "clean_dates", "clean_impossible", "duplicates", "type"], text: "Le moteur nettoie les entrées, puis identifie les types avant de préparer les transformateurs." },
  { label: "Valeurs extrêmes", keys: ["outliers"], text: "Pour les colonnes numériques, le traitement des extrêmes précède l’imputation." },
  { label: "Imputation", keys: ["missing"], text: "Les valeurs de remplacement sont apprises sur le train. Les colonnes catégorielles suivent leur propre branche." },
  { label: "Transformation", keys: ["distribution", "scaling"], text: "Les transformations de distribution précèdent la mise à l’échelle dans la branche numérique." },
  { label: "Extraction et encodage", keys: ["encoding", "dates", "text"], text: "Les branches catégorielles, dates et texte produisent des variables selon le type détecté." },
  { label: "Sélection finale", keys: ["selection"], text: "Après assemblage des branches, le moteur applique ses règles de sélection de variables." },
];
