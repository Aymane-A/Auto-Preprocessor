# Réalisation de l’atelier scientifique 3D

Validation du 10 novembre 2026, sous Windows avec Python 3.13 et Node.js 24.

## Système d’animation automatique

`frontend/src/animation.tsx` centralise les variantes, les durées et les composants d’animation. Framer Motion pilote les entrées, les interactions, les compteurs, les notifications, la navigation et les tooltips. Les dépendances déjà installées ont été réutilisées.

`AnimationSystem` observe les panneaux, métriques, graphiques, listes et tableaux du conteneur partagé. Les nouveaux éléments sont enregistrés automatiquement et animés à leur entrée dans le viewport. Le décalage entre éléments vaut 35 ms, plafonné à 245 ms. Les lignes et listes sont limitées aux 12 premiers éléments de chaque groupe. Les changements de données des graphiques, tableaux et vues de configuration relancent une entrée lorsqu’ils sont visibles. Les observateurs sont retirés lorsque leurs éléments quittent le DOM.

`WorkflowStage` et `AnimatePresence` gèrent le montage et le démontage des six étapes. Les anciennes étapes deviennent immédiatement inertes et masquées pour les technologies d’assistance pendant leur sortie de 120 ms. Le contenu entrant reste interactif. Le repère actif de navigation se déplace avec une transition spring sans rebond. Les composants existants conservent leurs tailles et leur disposition.

`AnimatedNumber` anime les valeurs pendant 650 ms, à leur première apparition et lors de leurs mises à jour. Les valeurs accessibles correspondent immédiatement au résultat exact. Un emplacement invisible réserve leur largeur finale. Les mesures détaillées des tableaux conservent leurs valeurs exactes. Les tooltips Recharts utilisent le composant partagé `ChartTooltip`. Les notifications d’erreur et de chargement disposent de transitions de présence, et la barre d’import suit la progression XHR réelle.

Les scènes 3D démarrent automatiquement à leur entrée et lors des changements d’état. Une séquence illustrative dure 1,4 seconde, ou 1,8 seconde pour la confirmation de résultat. Le flux de traitement continue pendant la requête réelle. Le système revient ensuite au rendu à la demande. La pause reste disponible comme contrôle d’accessibilité. Les scènes hors écran et les onglets masqués arrêtent le rendu.

La préférence de réduction des mouvements est suivie avec `useSyncExternalStore` et `matchMedia`, y compris pendant l’utilisation. Le hook de la version installée de Framer Motion lit la préférence initiale sans mettre son état React à jour lorsque celle-ci change. Le système partagé transmet donc explicitement la préférence courante à MotionConfig, aux compteurs et aux scènes GPU. En mode réduit, les entrées, les compteurs et les transitions deviennent immédiats.

Les menus déroulants conservent leurs contrôles natifs, avec retour visuel sur le focus et les changements de données. Le popup dessiné par le système d’exploitation conserve son comportement natif. L’application ne comporte actuellement aucun écran modal. Les surfaces DOM portant les rôles dialog et listbox dans le conteneur partagé bénéficient d’une entrée automatique.

### Validation des animations

Le build de production, TypeScript avec contrôle des déclarations inutilisées et les 4 tests frontend passent. Le parcours navigateur couvre les six étapes en mode supervisé fraude et sans cible avec le jeu churn, une transformation catégorielle, les aperçus alignés et les exports. Les CSV et le classeur du parcours supervisé sont téléchargés par le navigateur puis ouverts et vérifiés. Les dimensions sont 800 × 50 pour le train, 200 × 50 pour le test et 1 000 × 50 pour le complet. Le classeur contient les quatre feuilles attendues et le JSON conserve les choix bruts et Min-Max.

Les contrôles supplémentaires vérifient l’animation automatique d’un panneau ajouté au DOM, les entrées dans le viewport, le survol, le focus clavier, les compteurs après changement de fichier, les tooltips, les notifications d’erreur et le passage à la réduction des mouvements pendant la session. Les navigations rapides laissent une seule étape et un seul canvas montés. Les éléments en sortie deviennent inertes et les observateurs des éléments retirés sont nettoyés.

Le dernier contrôle headless Edge/SwiftShader a observé une médiane d’intervalle entre frames de 16,7 ms et un 95e percentile de 30,7 ms pendant un échantillon initial de 2,2 secondes. Cette mesure ne garantit pas une cadence constante sur tous les appareils. La préférence réduite laisse zéro animation active après chargement. Les captures, scripts et journaux sont conservés dans le dossier `automatic-animation`, à côté des preuves des itérations précédentes. Les avertissements de dépréciation et de taille du chunk GPU restent présents.

## Deuxième itération visuelle

Le réseau conserve ses 220 nœuds instanciés avec plusieurs profondeurs, un noyau éclairé, des structures satellites et un éclairage émeraude et beige. La caméra propose l’orbite à la souris, le zoom à la molette, un rapprochement, un recentrage et des transitions interruptibles. Les nœuds du pipeline réagissent au survol, à la sélection et aux états réels du parcours. Le rendu reste suspendu au repos et hors écran.

Les espacements du bandeau, de la navigation et des panneaux ont été resserrés. Les tableaux ont des lignes alternées, des nombres alignés et une hiérarchie plus lisible. La comparaison sépare les valeurs originales en beige des sorties transformées en émeraude. Elle affiche les différences de moyenne, d’écart-type, de manquants et d’asymétrie disponibles, les opérations appliquées et les compteurs d’effets renvoyés par Python.

L’API d’aperçu expose désormais jusqu’à 12 observations alignées et 6 sorties par observation, avec les valeurs originales provenant du tableau source. Les champs ont été ajoutés aux réponses existantes. Les calculs de transformation conservent leurs algorithmes. L’alignement des valeurs numériques et catégorielles est vérifié dans les tests API, ainsi que la conservation brute.

Les validations de cette itération comprennent 6 tests API, 7 tests ciblés sur les aperçus du moteur et 4 tests frontend, ainsi que TypeScript et le build de production. Les 2 tests API des démos et les 7 tests d’aperçu ont aussi été rejoués après l’ajout du nom réel du traitement catégoriel. Le parcours navigateur supervisé fraude couvre les six étapes, les statistiques, les 12 lignes comparées, les exports et l’invalidation des résultats. Le survol et la sélection directe des objets 3D ont été vérifiés par raycasting. Le contrôle des caméras vérifie une image différente après rapprochement, rotation et zoom. Les compteurs de rendu vérifient l’arrêt au repos, hors écran et en réduction des mouvements, puis la libération de 5 contextes sur 4 retours import/analyse.

La comparaison mobile a été vérifiée après stabilisation du redimensionnement du canvas. À 390 px, le document mesure 390 px et la scène mesure environ 321 px. Les fichiers téléchargés ont été ouverts et contrôlés, avec 800 lignes de train, 200 de test, 1 000 au total et les quatre feuilles attendues du rapport XLSX. Le journal du parcours ne contient aucune erreur JavaScript, console ou HTTP. Les avertissements de dépréciation de Three.js et de taille du chunk GPU demeurent.

Les preuves détaillées de cette itération sont conservées hors du dépôt. Les mêmes limites de diffusion du statut par l’API, de validation sur GPU physique et les avertissements des dépendances restent applicables.

## Audit et choix d’architecture

Le projet utilise React 19, TypeScript 5.9, Vite 7, Recharts 3 et Vitest 4. Le style repose sur CSS et des composants natifs. Les appels HTTP, les panneaux statistiques et les tableaux existaient déjà. Le backend FastAPI délègue les lectures et transformations à `run_isolated`, puis au moteur Python `AutoPreprocessor`.

L’interface réutilise ces points d’entrée et les modèles existants. Les modules du moteur, les algorithmes et les routes FastAPI conservent leur implémentation. `web_smoke.py` accepte désormais `--base-url` pour vérifier un serveur local sur un autre port.

La navigation comporte six étapes dans une application continue.

1. Import des données avec validation du format, glisser-déposer, progression des octets envoyés, aperçu et types détectés.
2. Analyse statistique avec les tableaux, histogrammes, boîtes à moustaches, corrélations et diagnostics fournis par Python.
3. Recommandations avec les preuves, explications, niveaux de confiance disponibles et alternatives par variable.
4. Configuration avec une scène interactive des opérations réelles et une vue distincte du parcours en six étapes.
5. Aperçu par variable et exécution du pipeline dans le worker Python.
6. Résultats, avertissements, transformations appliquées, paramètres appris et exports CSV, XLSX et configuration JSON.

Les choix, la cible et les options invalident les résultats devenus obsolètes. Un verrou synchrone sérialise les opérations déclenchées depuis l’interface. Une erreur de chargement du rapport conserve les fichiers déjà produits et permet de réessayer.

## Bibliothèques réutilisées

| Bibliothèque | Usage |
|---|---|
| React et React DOM | État, composants, chargement différé |
| TypeScript | Contrats des données et des composants |
| Recharts | Graphiques scientifiques existants |
| Three.js | Géométries, matériaux, éclairage, textures locales de numérotation |
| React Three Fiber | Canvas React, instancing et cycle de rendu |
| Drei | Contrôles orbitaux avec amortissement |
| Framer Motion | Apparition des étapes et respect de la réduction des mouvements |
| Vite et Vitest | Build et tests existants |

Three.js, Fiber, Drei, Framer Motion et les types Three avaient été installés à la demande précédente. Cette réalisation réutilise cette installation. Le diff Git contient donc aussi les modifications de dépendances issues de cette demande. Aucune bibliothèque concurrente de composants ou de graphiques n’a été ajoutée.

## Fichiers principaux

| Fichier | Responsabilité |
|---|---|
| `frontend/src/main.tsx` | Structure, navigation et composition des six étapes |
| `frontend/src/useWorkflow.ts` | Appels API, état, erreurs et invalidation des résultats |
| `frontend/src/workflow.ts` | Parcours et groupes d’opérations Python |
| `frontend/src/ScientificScene.tsx` | Visibilité, contrôles clavier, pause et repli WebGL |
| `frontend/src/ScientificCanvas.tsx` | Réseau de 220 nœuds, pipeline, éclairage et flux de particules |
| `frontend/src/PipelinePanel.tsx` | Configuration et rapport des transformations appliquées |
| `frontend/src/panels.tsx` | Panneaux métier réutilisés et adaptés |
| `frontend/src/api.ts` | Contrats API, upload XHR et client JSON |
| `frontend/src/style.css` | Palette forêt, émeraude et beige, composants et responsive |

L’ancienne landing page et ses illustrations ont été retirées. La scène réseau est explicitement présentée comme une illustration du moteur. Les données scientifiques restent dans les graphiques alimentés par l’API.

## Rendu et accessibilité

La scène GPU est chargée à sa première visibilité. Le réseau utilise un maillage instancié pour ses 220 nœuds et le flux utilise 18 particules instanciées. La résolution est plafonnée à 1,5 fois la résolution CSS. Les géométries restent simples, sans textures distantes ni post-traitement.

Le rendu fonctionne à la demande au repos. Les scènes hors écran et les onglets masqués cessent de dessiner. Les animations du flux suivent une requête de traitement active ou une action explicite de lecture. La confirmation d’export déclenche une animation de 1,6 seconde. La préférence de réduction des mouvements supprime les flux animés.

La souris contrôle l’orientation avec OrbitControls. Un curseur natif propose une orientation au clavier et les boutons numérotés donnent accès aux mêmes nœuds que la sélection 3D. Les indicateurs textuels complètent la couleur. Une vue schématique prend le relais lors d’une absence ou perte de WebGL. Les textures locales des numéros sont libérées explicitement et les ressources déclaratives suivent le démontage Fiber.

## Validation

| Vérification | Résultat |
|---|---|
| `python -m pytest -ra` | 221 tests réussis, 8 avertissements de dépréciation des dépendances |
| `npm --prefix frontend test` | 4 tests réussis |
| TypeScript et options `noUnusedLocals`, `noUnusedParameters` | Réussite |
| `npm --prefix frontend run build` | Réussite |
| Parcours navigateur | Import démo fraude, EDA, recommandations, choix bruts et Min-Max, configuration, aperçu réel, traitement, rapport et téléchargements, en développement et dans le build de production |
| Contenu téléchargé | Train 800 lignes, test 200 lignes, complet 1 000 lignes, 50 colonnes avec cible et variable conservée brute |
| Classeur téléchargé | Feuilles Résumé, Features, Corrélations et Colonnes de sortie ouvertes avec openpyxl |
| Configuration téléchargée | Choix `transaction_id=none` et `montant=scaling:MINMAX` vérifiés |
| Résultats obsolètes | Modification de la part du test désactive les exports jusqu’à une nouvelle exécution |
| Responsive | Absence de débordement de document à 320, 390, 768 et 1 024 px |
| WebGL indisponible | Repli visible et import du fichier churn réussi, avec réduction des mouvements |
| Console navigateur | Aucune erreur JavaScript, console ou requête HTTP dans le parcours validé |
| Arrêt du rendu | Compteurs d’appels GPU stabilisés au repos, après pause, hors écran et avec réduction des mouvements |
| Cycle des contextes | Après quatre retours import/analyse, 5 contextes créés et 5 libérés, aucun canvas restant dans l’analyse |

Le contrôle navigateur utilise le Playwright JavaScript déjà installé et Edge Chromium headless, avec SwiftShader pour le WebGL. Le module Python Playwright était absent. Aucun navigateur ou paquet Playwright supplémentaire n’a été installé. Les captures ont été inspectées et le cadrage du pipeline corrigé. Un problème de démontage des étiquettes HTML Drei a été corrigé en utilisant des numéros dans la scène et des libellés natifs accessibles.

Le mode Vite a été vérifié sur `localhost:5173` avec l’API dédiée sur 8001. Le build de production a été servi par un second processus de test sur 8002, avec cette origine ajoutée à sa liste autorisée uniquement en mémoire. Le serveur existant de l’autre copie du projet sur 8000 a été conservé. Le lancement normal du projet utilise 8000, déjà autorisé par le backend.

Les contrôles de contexte constituent une vérification bornée du montage et du démontage. Un profil mémoire de longue durée et des mesures sur GPU physique restent à réaliser.

### Charge sur grand tableau

Le test HTTP utilise `web_smoke.py --dataset large --base-url http://127.0.0.1:8001` et un fichier synthétique de 159 029 805 octets, 285 000 lignes et 31 colonnes. Les limites du serveur de validation étaient 200 Mio par fichier, 2 048 Mio par worker et 600 secondes par opération.

| Mesure | Résultat |
|---|---:|
| Import HTTP | 6,88 s |
| Configuration HTTP | 4,83 s |
| Prétraitement HTTP | 194,07 s |
| Durée du worker de traitement | 193,932 s |
| Pic RSS du worker | 821 489 664 octets, environ 783 Mio |
| Train | 228 000 × 31 |
| Test | 57 000 × 31 |
| Complet | 285 000 × 31 |
| Rapport et téléchargement | HTTP 200 |

Le parcours navigateur complet a utilisé le jeu de fraude de 1 000 lignes. La charge de 285 000 lignes a été vérifiée par HTTP. Les limites de validation supérieures aux valeurs par défaut sont requises pour reproduire cet essai.

### Outils de contrôle et avertissements

Le dépôt ne configure aucun script ESLint. TypeScript et le contrôle des déclarations inutilisées ont été exécutés. Sloplint Hallmark a également été exécuté et a permis de corriger le surtitre redondant, le titre bicolore et la largeur de prose. Son signal sur deux éléments sticky concerne la navigation et l’en-tête du tableau, qui appartiennent à des conteneurs de défilement distincts. Les libellés longs peuvent revenir à la ligne sur mobile.

Le build signale des directives `use client` ignorées dans Framer Motion et un chunk GPU différé supérieur à 500 Ko. Ce chunk mesure environ 950 Ko minifié et 255 Ko gzip. L’avertissement de taille reste visible. Three.js émet également un avertissement de dépréciation de `Clock` utilisé par l’écosystème Fiber.

## Limites fonctionnelles documentées

L’API existante ne diffuse aucun pourcentage de calcul, journal d’exécution ou étape interne en temps réel. L’écran indique donc une requête active et affiche les mesures du worker une fois reçues. La progression de l’import mesure uniquement les octets envoyés.

L’aperçu est calculé par variable sur un échantillon. L’API ne fournit pas d’EDA complète du tableau final ni de graphe calculé de dépendances entre features. Le résultat affiche les dimensions et les transformations du rapport. Le réseau 3D n’exprime aucune projection statistique des observations.

Chaque variable accepte l’alternative prise en charge par le contrat d’override existant. Les autres étapes suivent le moteur. Le JSON de configuration permet l’archivage des options et des choix, avec le rapport et les données. L’interface ne comporte pas encore d’import de ce JSON.

Le mode sans cible et les deux jeux de démonstration sont couverts par les tests API existants. La nouvelle interface a été exercée de bout en bout en mode supervisé fraude. Le parcours sans cible complet dans le navigateur, Safari, Firefox et les appareils mobiles physiques restent à vérifier.

## Preuves locales

Les captures, journaux et exports de validation locale sont conservés hors du dépôt et ne sont pas inclus dans cette publication.

## Lancement local

Depuis la racine du dépôt après installation des dépendances Python et npm, exécutez les commandes suivantes.

```powershell
npm --prefix frontend run build
python serve.py
```

Ouvrez ensuite [AutoPrépa local](http://127.0.0.1:8000). Le mode développement utilise `npm --prefix frontend run dev` avec l’API démarrée dans un second terminal. `VITE_API_URL` permet de choisir une autre URL d’API de développement.
