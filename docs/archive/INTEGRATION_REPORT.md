# Intégration React et FastAPI

Rapport du 11 novembre 2026.

## Audit et périmètre

L’application conserve React 19, TypeScript, FastAPI, Recharts, Three.js, React Three Fiber, Drei et Framer Motion. Le parcours reste composé de six étapes. Les fichiers de ce travail ajoutent des contrôles et des contrats aux composants existants. Aucun algorithme du moteur, écran du parcours, composant 3D ou mécanisme d’animation n’a été retiré.

| Module | Capacités auditées | Intégration |
|---|---|---|
| `auto_preprocessor.py` | Détection de types, nettoyage, imputation, distributions, outliers, encodages, dates, texte, sélection, cible, sauvegarde | Moteur conservé, catalogue d’actions réutilisé, préprocesseur entraîné téléchargeable |
| `app_preprocess.py` | Lecture des formats, options, validations des choix, aperçu, découpage et exports desktop | Logique réutilisée par les workers web, options étendues sans modifier les défauts desktop |
| `statistical_profile.py` | Normalité, associations, effets, mécanismes de manquants, comparaison des méthodes et texte | Diagnostics avancés exposés, p-values et corrections BH affichées, Pearson, Spearman, Kendall et effets selon les types |
| `model_evaluation.py` | Validation croisée et temporelle, modèles et comparaison des échelles | Endpoint et panneau de comparaison ajoutés |
| `isolated_processing.py` | Temps et mémoire bornés, processus isolés | Toutes les nouvelles opérations coûteuses utilisent le même mécanisme |
| `api_server.py` | Imports bornés, sessions, verrouillage, erreurs, publication atomique et téléchargements | Endpoints existants conservés et étendus, fermeture de session ajoutée |
| `web_processing.py` | EDA, recommandations, aperçus, traitement et rapports | Options partagées, choix composés, paramètres appris, exports et EDA des sorties |
| Scripts de benchmark et de contrôle | Benchmarks multi-datasets, signal, permutations et ablations | Scripts CLI conservés. La comparaison interactive réutilise le module de validation croisée |

L’inventaire complet des **52 réglages publics** et des **91 couples étape/action** figure dans [INTEGRATION_OPTIONS.md](INTEGRATION_OPTIONS.md). Les paramètres de journalisation, le budget d’allocation dense, la représentation interne des overrides et `preserve_rows` sont pilotés par le serveur ou les routines d’évaluation. Ils conservent leur API Python.

## Exigences TP et résultats accessibles

| Exigence | Interface et données utilisées |
|---|---|
| Import CSV et Excel | Import natif et glisser-déposer, aperçu, types, validation et progression d’envoi XHR |
| Supervisé et sans cible | Choix du mode, sélection de cible, tâche automatique, classification ou régression |
| EDA complète des variables | Descriptifs, distributions, quartiles, manquants, doublons, IQR, MAD, asymétrie, kurtosis et relations |
| Corrélations et effets | Pearson, V de Cramér, tests adaptés au type, Kruskal–Wallis et ε², Spearman, Kendall, p-values et BH disponibles dans les panneaux avancés |
| Détection et recommandations | Anomalies, identifiants, colinéarité, fuites et diagnostics du moteur, preuves et explications |
| Traitements | Imputation, outliers, distributions, échelles, encodages, dates, texte, nettoyage et suppression |
| Accepter, modifier ou conserver brut | Sélecteurs rapides existants et éditeur de plusieurs opérations par variable |
| Avant/après | Graphiques, statistiques, opérations appliquées, effets et douze observations alignées avec six sorties au maximum |
| Prévention des fuites | Paramètres appris sur le train, transformation du test, encodages supervisés et ajustement dans chaque pli de validation |
| Exports | CSV train/test/complet, dataset XLSX, rapport XLSX, rapport JSON et configuration JSON |
| Préprocesseur | Téléchargement du modèle ajusté des variables traitées lorsque celui-ci a été ajusté |
| Démonstrations | Fraude bancaire synthétique et churn télécom fournis dans `data/` |

Les bornes, ordres et variables protégées sont résolus sur les noms normalisés du moteur. Les choix par variable sont également associés aux noms normalisés uniques. Les valeurs de fichier et les colonnes laissées brutes sont restituées par identifiant de ligne.

## Synchronisation des réglages

`web_contracts.py` contient les schémas Pydantic et leurs défauts, bornes, énumérations et validations croisées. `GET /api/capabilities` publie ce schéma. Le formulaire React est généré à partir de ce catalogue et organise les réglages en sections avancées.

Les requêtes d’analyse, d’aperçu et d’exécution utilisent les mêmes options. Une modification appliquée invalide les aperçus et résultats devenus obsolètes. Une variation de la part du test invalide également les exports de l’interface. Les paramètres d’exécution et choix figurent dans le rapport et le JSON de configuration.

Le comportement adaptatif historique reste le défaut du moteur. Il peut préciser les règles globales selon chaque variable. Un choix explicite par opération fixe la méthode concernée. Les paramètres numériques métier, les marqueurs manquants, les ordres, les mappings, les n-grammes et les dimensions exposées sont validés avant leur utilisation.

Les alternatives sous forme de chaînes restent acceptées pour les clients existants. Les choix composés utilisent une structure par étape.

```json
{
  "session_id": "identifiant-de-session",
  "options": {"test_size_pct": 25, "outlier_method": "none"},
  "choices": {
    "montant": {
      "missing": {"action": "CONSTANT", "parameters": "7"},
      "distribution": {"action": "NONE", "parameters": ""},
      "scaling": {"action": "MINMAX", "parameters": ""}
    }
  }
}
```

Les noms d’étapes, les actions disponibles, les paramètres attendus et l’aide proviennent du backend. Un type forcé relance le calcul des opérations applicables. Le choix global de suppression des doublons doit être cohérent entre les variables. Les variables KNN partagent un nombre de voisins cohérent.

La configuration exportée peut être réimportée sur un fichier dont les variables, le mode, la cible et la tâche correspondent. L’import est validé avant modification de l’état React. Les requêtes échouées conservent les choix précédemment validés.

## Endpoints

| Route | Utilisation |
|---|---|
| `POST /api/upload` | Lecture bornée et création de session |
| `POST /api/config` | Mode, cible et tâche |
| `GET /api/eda/{id}` | EDA et associations |
| `GET /api/eda/{id}/feature/{column}` | Distribution et statistiques d’une variable |
| `GET /api/capabilities` | Schéma et défauts des réglages publics, formats et limites |
| `POST /api/analyze` | Recommandations avec options et validation des choix éventuels |
| `POST /api/preview` | Aperçu avec options et choix simple ou composé |
| `POST /api/preprocess` | Ajustement sur train et publication d’une nouvelle génération d’exports |
| `POST /api/evaluate` | Comparaison de mises à l’échelle dans des plis indépendants |
| `GET /api/results-eda/{id}` | EDA du CSV transformé publié |
| `POST /api/export-excel` | Préparation du dataset XLSX dans le worker |
| `GET /api/report/{id}` | Rapport détaillé complet |
| `GET /api/download/{id}/{kind}` | CSV, dataset XLSX, rapport XLSX/JSON ou préprocesseur |
| `DELETE /api/session/{id}` | Fermeture explicite et suppression des fichiers temporaires de la session |

Les requêtes inconnues ou incohérentes sont rejetées. Les réponses sont converties en JSON strict avec les non-finis représentés par `null`, et leurs champs, formes et contenus sont vérifiés dans les tests. Le client conserve des contrats TypeScript pour les réponses métier.

## Rapports et exports

Le classeur conserve ses quatre premières feuilles et ajoute deux feuilles de détail.

1. `Résumé` contient les dimensions, le mode, la tâche, les choix, les options, les avertissements et les informations de nettoyage.
2. `Features` contient une ligne par variable d’origine, y compris la cible, les variables supprimées et les variables brutes.
3. `Corrélations` conserve les matrices d’entrée calculées dans les limites de largeur documentées.
4. `Colonnes de sortie` conserve le schéma ordonné des exports.
5. `Opérations` détaille les preuves, seuils, explications, confiance, alternatives et comparaisons par variable et par étape.
6. `Diagnostics` contient les diagnostics complets par champ. Les longues valeurs sont réparties en parties de 30 000 caractères.

Les chaînes présentant un préfixe de formule sont protégées dans Excel par une apostrophe. Les lecteurs programmatiques des fragments JSON doivent retirer cette apostrophe de protection avant reconstitution. Le rapport JSON conserve les valeurs complètes sans limite de taille de cellule Excel.

Les mappings d’encodage affichés dans les paramètres appris sont accompagnés de leur cardinalité et limités aux vingt premières entrées. Le modèle entraîné conserve son état complet. Les références de dates, paramètres d’imputation multivariée, paramètres d’étapes et informations de vocabulaire/SVD sont aussi exposés lorsqu’ils existent.

Le dataset XLSX est préparé à la demande, dans les limites natives de 1 048 575 lignes de données et 16 384 colonnes. Les valeurs textuelles sont protégées contre l’interprétation en formules. Une erreur de préparation conserve les CSV et le rapport déjà publiés.

Le préprocesseur téléchargeable concerne les variables traitées. La restitution des colonnes laissées brutes et l’alignement de la cible font partie de l’export web. Les sessions entièrement brutes ne proposent aucun modèle non ajusté. Le serveur conserve un import exclusivement tabulaire. Le chargement Python de modèles joblib reste réservé aux fichiers de confiance.

## Sessions et échecs

Le verrou serveur sérialise les opérations d’une session, y compris les téléchargements. Les conflits renvoient `409`. Les opérations lourdes utilisent les limites mémoire et temps du processus isolé. Une génération de résultats n’est publiée qu’après succès. Un échec conserve la génération serveur précédente.

L’expiration renvoie `404` avec une invitation à réimporter le fichier. L’interface efface son état de session expirée. La fermeture explicite refuse les sessions occupées et retire uniquement les fichiers temporaires de la session sélectionnée. Les chemins locaux fournis par le navigateur ne servent jamais de chemins de travail au serveur.

## Validation

Les preuves détaillées de cette livraison sont conservées hors du dépôt et ne sont pas publiées.

La suite historique comporte 221 tests. Les nouveaux tests couvrent les 91 couples étape/action, les paramètres supplémentaires, les types forcés, les contrats, les formats, les exports, les modèles et les sessions. Un test supplémentaire vérifie la régression, la cible originale et la validation temporelle. Les résultats finaux d’exécution sont indiqués à la fin de ce rapport.

Les contrôles navigateur utilisent le Playwright JavaScript et Edge Chromium déjà installés, avec SwiftShader pour WebGL. Le module Python Playwright est absent. Les captures sont inspectées, les erreurs console et réseau enregistrées, et les téléchargements ouverts pour vérifier leur contenu.

## Limites et fonctions restantes

- Les jeux de fraude bancaire et de churn fournis sont synthétiques. La démonstration ne constitue pas une validation sur des transactions réelles.
- XGBoost et LightGBM nécessitent leurs bibliothèques optionnelles, absentes de cet environnement. Leur indisponibilité renvoie une erreur explicite et a été testée. Les huit autres familles de comparaison ont produit des scores.
- Les imports CSV, TSV, TXT, XLSX, JSON et Parquet ont été exercés. Le lecteur XLS existant dépend de xlrd. Aucun fichier XLS binaire historique n’a été validé pendant cette livraison.
- La validation croisée accepte 20 à 5 000 cibles renseignées et 2 à 5 plis. Le découpage temporel exige des dates valides et distinctes. Les règles globales sont fixées pendant cet essai pour que les noms des scalers correspondent à leur exécution. Une évaluation indépendante reste nécessaire après sélection.
- L’analyse des résultats utilise le CSV publié. Les types inférés de ce CSV peuvent différer de ceux d’une colonne brute conservée en mémoire.
- L’EDA décrit toutes les variables. Les matrices restent bornées à 40 colonnes numériques et 12 catégorielles, les relations à 20 000 lignes et les catégories des associations à leurs limites statistiques. Le moteur signale l’échantillonnage de son analyse au-delà de 200 000 lignes.
- L’aperçu est descriptif et peut être échantillonné. Ses paramètres sont ajustés pour ce diagnostic. Le traitement final et les validations apprennent leurs paramètres sur leurs ensembles d’entraînement respectifs.
- L’API ne diffuse pas de pourcentage de calcul, journal ou étape interne en direct. Le statut indique la requête active et les mesures finales du worker.
- Les budgets par défaut et protections des formats restent applicables. Cette livraison n’ajoute pas de nouveau test de charge de 285 000 lignes. Le test de charge de l’itération précédente est documenté séparément.
- La validation visuelle utilise Chromium headless. Firefox, Safari, les appareils physiques et un profil mémoire prolongé restent à vérifier.
- Les avertissements de dépréciation des dépendances et de taille du chunk GPU différé demeurent visibles dans le build.

## Reproduction

```powershell
python -m pip install -r requirements-web.txt
npm --prefix frontend install
npm --prefix frontend run build
python serve.py
```

Ouvrir [AutoPrépa local](http://127.0.0.1:8000). Un processus Python déjà lancé doit être redémarré pour charger les nouveaux endpoints. Les tests utilisent les commandes suivantes.

```powershell
python -m pytest -ra
npm --prefix frontend test
npm --prefix frontend exec -- tsc --noEmit --noUnusedLocals --noUnusedParameters -p frontend/tsconfig.json
```

## Résultats exécutés le 11 novembre 2026

| Vérification | Résultat constaté |
|---|---|
| Suite Python complète | 340 tests réussis en 578,48 s, 25 avertissements de dépendances |
| Régression, cible originale et validation temporelle ajoutées ensuite | 1 test réussi en 27,63 s |
| Total Python exécuté | 341 tests réussis, dont 221 historiques et 120 tests de parité |
| Tests frontend | 4 tests réussis |
| TypeScript strict et build production | Réussis ; avertissement de taille du chunk GPU différé conservé |
| Parcours navigateur fraude CSV supervisée | Six étapes, options globales, opérations combinées, aperçu, validation croisée, traitement, exports et réimport de configuration réussis |
| Parcours navigateur churn XLSX sans cible | Six étapes, encodage alternatif, aperçu, traitement et préparation Excel réussis |
| Console et réseau des deux parcours finaux | Aucune erreur JavaScript, erreur console ou requête HTTP en échec ; avertissement THREE.Clock existant |
| Responsive et accessibilité du mouvement | Largeurs 320, 390, 768 et 1024 px contrôlées ; contexte à mouvement réduit sans animation active ; repli sans WebGL contrôlé |
| Captures | Configuration, aperçu et résultats des deux datasets inspectés visuellement |

Le scénario fraude produit un train de 750 × 50, un test de 250 × 50 et un dataset complet de 1 000 × 50. Les CSV téléchargés ont été ouverts et leur concaténation vérifiée. Le dataset XLSX contient le même schéma et les mêmes valeurs numériques. Le rapport Excel téléchargé comprend les six feuilles attendues, dont 14 variables, 221 opérations et 569 champs de diagnostic hors en-têtes. Le JSON contient les sections `summary`, `features`, `core` et `correlations`. La configuration téléchargée conserve l’imputation constante personnalisée à 7 et le fichier joblib contient un modèle ajusté. Les tests API vérifient séparément sa réutilisation et la reconstitution des diagnostics Excel.

Le scénario churn part de 3 015 × 12 et produit 3 005 × 33, après nettoyage, répartis en 2 402 lignes train et 603 lignes test. Le téléchargement du dataset Excel a été vérifié dans le scénario fraude ; sa préparation a été vérifiée dans le scénario churn. Les contrôles de fuite, les erreurs et les formats supplémentaires sont couverts par les tests Python, sans prétendre avoir parcouru chaque combinaison de réglages dans le navigateur.

Les preuves locales comprennent les journaux navigateur et d’animation, les captures PNG et les exports de validation ; elles restent hors du dépôt. L’audit de mouvement sur Chromium headless avec rendu logiciel mesure une médiane de trame de 16,7 ms et un 95e percentile de 50 ms. Il ne démontre pas un maintien constant à 60 FPS sur tous les appareils. Les animations 3D automatiques sont absentes dans le contexte à mouvement réduit. Aucun paquet supplémentaire n’a été installé pour cette intégration.
