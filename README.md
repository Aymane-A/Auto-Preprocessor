# AutoPreprocessor : nettoyage et transformation automatiques

Prend n'importe quel jeu de données tabulaire supervisé (classification ou
régression) et le transforme en matrice 100 % numérique, sans fuite de données
(le préprocesseur est ajusté sur le train uniquement).

## Installation (Windows / PowerShell)

```powershell
py -m pip install -r requirements.txt
```

`requirements.txt` regroupe tout. `openpyxl` n'est nécessaire que pour lire les fichiers Excel, `matplotlib`
que pour le graphique du benchmark.

## Fichiers

| Fichier | Rôle |
|---|---|
| `auto_preprocessor.py` | Le module (classe `AutoPreprocessor`, compatible scikit-learn) |
| `app_preprocess.py` | Application Tkinter (interface graphique) |
| `test_auto_preprocessor.py` | 16 tests automatiques |
| `signal_check.py` | Vérifie si un dataset contient du signal et si AutoPreprocessor aide (AUC, PR-AUC, cible mélangée) |
| `benchmark_multi.py` | Benchmark sur plusieurs datasets (naïf, sklearn standard, AutoPreprocessor) |
| `credit_g_check.py` | Analyse des variantes d'options sur le dataset credit_g |
| `build_exe.bat` | Construit `dist\AutoPreprocessor.exe` (PyInstaller, Windows) |
| `data/` | Datasets de démonstration (synthétiques, voir plus bas) |
| `requirements.txt` | Dépendances |
| `.gitignore` | Exclut `__pycache__`, `build/`, `dist/`, `resultats/`, `*.joblib`, `*.spec` |

## Utilisation

### 1. Application graphique

```powershell
py app_preprocess.py
```

1. **Ouvrir un fichier** (CSV, Excel, JSON).
2. Onglet **Paramètres** : choisir la colonne cible, la part du test (20 % par défaut), puis **Lancer le prétraitement**.
3. Onglet **Rapport** : colonnes supprimées, types détectés, avertissements (fuites, déséquilibre de classes).
4. Onglet **Résultat** : exporter `train_transforme.csv`, `test_transforme.csv`, `detail_colonnes.csv`, `rapport_pretraitement.txt` et `preprocesseur.joblib`.

### 2. En Python

```python
import pandas as pd
from auto_preprocessor import AutoPreprocessor

df = pd.read_csv("mon_fichier.csv")
y = df.pop("ma_cible")

prep = AutoPreprocessor(task="classification")   # ou "regression", ou "auto"
X_train_num = prep.fit_transform(df_train, y_train)  # valeurs out-of-fold pour le target encoding
X_test_num = prep.transform(df_test)

print(prep.report())
```

Toujours utiliser `fit_transform` sur le train et `transform` sur le test.

## Ce que fait le module

- **Nettoyage** : valeurs manquantes déguisées, nombres écrits en texte, doublons.
- **Détection des types** : identifiant, constante, binaire, ordinale, numérique, catégorielle (faible / forte cardinalité), date/heure, texte.
- **Transformations** : winsorisation, Yeo-Johnson si asymétrie, mise à l'échelle, indicateurs de valeurs manquantes, regroupement des catégories rares, One-Hot, Target Encoding (out-of-fold), extraction de features temporelles (cycliques), TF-IDF + SVD pour le texte.
- **Filtres** : variance nulle (colonnes constantes), colinéarité.
- **Détection de fuites** : chaque colonne est testée seule contre la cible ; alerte (ou suppression si `drop_leaky_columns=True`) si une colonne prédit la cible bien mieux que toutes les autres.

## Tests

```powershell
py -m pytest
```

Résultat attendu : `16 passed`.

## Vérifier si un dataset contient du signal

```powershell
py signal_check.py data\churn_telecom_demo.csv churn --sep ";"
```

Compare en validation croisée (5 plis, préprocesseur refit à chaque pli) un pipeline sklearn standard et AutoPreprocessor, avec RandomForest et régression logistique, plus une ligne « cible mélangée » :

- AUC réel proche de 0.50 : le dataset n'a pas de signal exploitable (cas de `Dataset_Fraude_Bancaire_complete`).
- Ligne « cible mélangée » au-dessus de ~0.55 : fuite à investiguer.

## Datasets de démonstration (`data/`)

Tous synthétiques, générés pour tester le projet ; ils ne représentent pas de vraies données.

| Fichier | Tâche | Cible | Particularités |
|---|---|---|---|
| `churn_telecom_demo.csv` (`;`) | classification | `churn` | ID, date, montants en texte (`"67,50 MAD"`), `?`, doublons, NaN |
| `immobilier_demo.csv` | régression | `prix_dh` | cible asymétrique (skew ~7.8), outliers, NaN |
| `fraud_signal_demo.csv` | classification | `is_fraud` | 5 % de fraudes, signal faible injecté (`montant`, `canal`, `score_risque_client`) |

## Créer l'exécutable Windows

`build_exe.bat` construit `dist\AutoPreprocessor.exe` (PyInstaller, un seul fichier, sans console). Testé sous Windows 11 avec Python 3.13 : la construction prend 5 à 10 minutes et produit un fichier d'environ 105 Mo qui s'ouvre et fonctionne.

1. **Copier le projet hors d'OneDrive** (par exemple `C:\AutoPre`). OneDrive synchronise les gros fichiers pendant la construction et peut produire un `.exe` tronqué (symptôme observé : fichier de ~590 octets et message « This app can't run on your PC »).
2. Double-clic sur `build_exe.bat`, ou en PowerShell :

```powershell
cd C:\AutoPre
py -m pip install --upgrade pyinstaller openpyxl
py -m PyInstaller --noconfirm --clean --onefile --windowed --name AutoPreprocessor --hidden-import sklearn.utils._typedefs --hidden-import sklearn.neighbors._partition_nodes --collect-submodules sklearn --collect-submodules scipy app_preprocess.py
"{0:N0} Mo" -f ((Get-Item dist\AutoPreprocessor.exe).Length/1MB)
```

3. Vérifier la taille : **moins de 20 Mo = construction incomplète**, recommencer. Le script `build_exe.bat` fait cette vérification et s'arrête sur erreur.

**Si l'exécutable ne s'ouvre pas** : construire une version avec console, qui affiche les erreurs.

```powershell
py -m PyInstaller --noconfirm --onefile --console --name AutoPreprocessor_debug --hidden-import sklearn.utils._typedefs --hidden-import sklearn.neighbors._partition_nodes --collect-submodules sklearn --collect-submodules scipy --exclude-module torch --exclude-module IPython --exclude-module matplotlib app_preprocess.py
$t = Get-Date; .\dist\AutoPreprocessor_debug.exe; "code: $LASTEXITCODE | durée: $([int]((Get-Date)-$t).TotalSeconds) s"
```

Un code `0` avec une durée correspondant au temps d'utilisation de la fenêtre signifie que tout fonctionne. Les avertissements `torch` et `scipy.special._cdflib` pendant la construction sont sans conséquence.

L'exécutable n'est pas versionné dans Git (voir `.gitignore`) : le publier dans les « Releases » du dépôt GitHub si besoin.

## Limites à connaître

- Une colonne qui « fuit » légitimement la cible (ex. `boat` dans Titanic) n'est supprimée automatiquement que s'il y a au moins 10 colonnes candidates.
- Le prétraitement n'invente pas de signal : sur un dataset sans lien entre features et cible, le score restera proche du hasard (AUC ≈ 0.5), quel que soit le nettoyage.
- Sur un petit dataset avec peu de positifs (ex. 43 fraudes), les colonnes de type date, identifiant ou haute cardinalité sans lien avec la cible ajoutent des features de bruit et peuvent faire baisser l'AUC d'un RandomForest par rapport à un pipeline simple (observé : 0.56 contre 0.63). Retirer ces colonnes avant le prétraitement si elles n'ont pas de sens métier.
- Une tâche « regression » demandée sur une cible textuelle (« Oui »/« Non ») est automatiquement convertie en classification, avec avertissement.
- Sur les modèles à base d'arbres (Random Forest), le gain par rapport à un pipeline sklearn standard est faible ; il est net pour les modèles linéaires avec outliers et asymétrie.

## Journal des changements

| Étape | Changement |
|---|---|
| Détection de fuite | Chaque colonne brute est testée seule (arbre de profondeur 4, validation croisée) ; alerte si le score dépasse 0.95 **et** dépasse la meilleure autre colonne de 0.10 (ou ≥ 0.995). Suppression automatique seulement si `drop_leaky_columns` et au moins 10 colonnes. |
| Target Encoding | `fit_transform()` remplace les colonnes encodées du train par des valeurs out-of-fold (pas de fuite de la cible dans le train). |
| credit_g | `"none"` n'est plus un faux manquant ; les entiers discrets restent numériques (`encode_discrete_as_categorical=False`). |
| Colonnes date/heure | Une colonne horodatage quasi unique (`heure`) n'est plus classée identifiant : elle produit des features cycliques. |
| Variance | Les colonnes constantes ou de variance < 1e-12 sont supprimées (bug de comparaison flottante). |
| Nombres en texte | `"67,50 MAD"`, `"12 kg"`, `"MAD 67"` deviennent numériques ; `"C1004"` et `"2024-01-05"` restent non numériques. |
| Tâche régression | Une cible textuelle (`Oui`/`Non`) avec tâche « regression » passe en classification avec avertissement ; l'application remet « Type de tâche » sur `auto` à chaque fichier chargé. |
| Outils | `signal_check.py`, datasets de démonstration, `build_exe.bat` avec vérification de taille, exécutable Windows testé. |
| Tests | 16 tests (`py -m pytest`). |

## Historique des prompts utilisés pour construire ce projet

Le projet a été construit en plusieurs étapes, chacune guidée par un prompt. Ils sont reproduits ci-dessous, dans l'ordre, pour pouvoir refaire le travail ou l'adapter à un autre projet. Chaque prompt est à coller tel quel dans une conversation avec Claude (ou un autre LLM), en attachant les fichiers mentionnés.

### Étape 0 : demande initiale (en darija)

> Je veux la transformation d'un dataset en tenant compte des différents types de features dans le cas d'un projet ML qui consiste à prendre n'importe quel dataset structuré supervisé et faire la transformation en numérique.
>
> Je ne veux pas une transformation pour un seul dataset, je veux la faire sur plusieurs datasets.

### Étape 1a : prompt de génération du module (première version)

À coller dans une conversation vide. Il produit `auto_preprocessor.py`, les tests et une démo.

````text
# RÔLE
Tu es un Data Scientist / ML Engineer senior, expert en prétraitement de données tabulaires, en scikit-learn et en code Python de production.

# CONTEXTE
Je construis un projet de Machine Learning supervisé (classification ou régression). Je veux un module réutilisable qui prend n'importe quel dataset structuré (CSV/DataFrame pandas) et le transforme automatiquement en matrice 100 % numérique, prête pour l'entraînement, quel que soit le mélange de types de colonnes.

# OBJECTIF
Écris une classe Python `AutoPreprocessor` qui :
1. Détecte automatiquement le type de chaque feature.
2. Applique la transformation adaptée à chaque type.
3. Retourne X (numérique, sans NaN, sans colonne non numérique) et y (encodée si besoin).
4. Évite toute fuite de données : fit uniquement sur le train, transform sur le test.

# ENTRÉES
- Un DataFrame pandas `df` et le nom de la colonne cible `target`.
- Le type de tâche : "classification", "regression" ou "auto" (déduit de la cible).
- Paramètres optionnels : seuil de cardinalité, stratégie d'imputation, scaler, etc.

# DÉTECTION DES TYPES (règles explicites)
- Colonnes à supprimer : identifiants (cardinalité ≈ nb de lignes), constantes, > 90 % de valeurs manquantes (seuil configurable).
- Booléennes/binaires : 2 valeurs uniques.
- Numériques : int/float. Un entier avec très peu de valeurs uniques peut être traité comme catégoriel (seuil configurable).
- Catégorielles faible cardinalité (≤ 10 par défaut) : object/category.
- Catégorielles haute cardinalité (> 10) : object/category.
- Dates/temps : datetime ou chaînes parsables en date.
- Texte libre : chaînes longues avec forte cardinalité (traitement simple type TF-IDF limité, ou suppression avec avertissement).

# TRANSFORMATIONS
- Numériques : imputation par médiane, puis StandardScaler (ou RobustScaler si outliers forts détectés par IQR).
- Binaires : mapping 0/1.
- Catégorielles faible cardinalité : imputation par mode, puis OneHotEncoder(handle_unknown="ignore").
- Haute cardinalité : Frequency Encoding ou Target Encoding avec validation croisée interne (pour éviter la fuite).
- Dates : extraction année, mois, jour, jour de semaine, trimestre, est_weekend, puis traitement comme numériques.
- Cible : LabelEncoder si classification avec classes textuelles ; aucune transformation en régression.
- Lignes où la cible est manquante : supprimées avec un message.

# CONTRAINTES TECHNIQUES
- Utiliser `sklearn.pipeline.Pipeline` et `ColumnTransformer`.
- API compatible sklearn : `fit`, `transform`, `fit_transform`, `get_feature_names_out`.
- Gérer les catégories inconnues au moment du transform sans erreur.
- Sauvegarde/chargement avec `joblib` (`save()` / `load()`).
- Aucune dépendance exotique : pandas, numpy, scikit-learn uniquement.
- Type hints, docstrings, code PEP8, commentaires utiles.

# SORTIES ATTENDUES
1. Le code complet de `AutoPreprocessor` dans un fichier `auto_preprocessor.py`.
2. Un rapport (`report()`) listant, pour chaque colonne : type détecté, transformation appliquée, colonnes supprimées et pourquoi.
3. Un script de test sur un dataset synthétique contenant : numériques avec NaN, catégorielles, binaires, dates, haute cardinalité, un ID et une constante. Vérifier avec des assertions que X final est numérique, sans NaN ni inf, et que train/test ont les mêmes colonnes.
4. Un court exemple d'utilisation (train_test_split, fit sur train, transform sur test, entraînement d'un RandomForest).

# CRITÈRES DE QUALITÉ
- Fonctionne sans modification sur un dataset inconnu.
- Aucune fuite de données entre train et test.
- Code lisible, modulaire, testé, exécuté avant livraison.

# FORMAT DE RÉPONSE
Réponds en français. Commence par un court plan, puis le code, puis les résultats du test, puis les limites connues et les pistes d'amélioration.
````

### Étape 1b : prompt de génération du module (version améliorée, en français)

Version finale utilisée : ajoute la gestion des cas limites, de la fuite de cible, du déséquilibre de classes, des dates, etc.

````text
# RÔLE
Tu es un Data Scientist / ML Engineer senior (10+ ans d'expérience), expert en prétraitement de données tabulaires, en scikit-learn et en code Python de production. Tu privilégies la robustesse, l'absence de fuite de données et la reproductibilité.

# CONTEXTE
Je construis un projet de Machine Learning supervisé (classification binaire, multiclasse ou régression). Je veux un module réutilisable qui prend n'importe quel dataset structuré (CSV/DataFrame pandas), quel que soit le mélange de types de colonnes, et le transforme automatiquement en matrice 100 % numérique, propre et prête pour l'entraînement.

# OBJECTIF
Écris une classe Python `AutoPreprocessor` (compatible scikit-learn) qui :
1. Nettoie les données brutes et détecte automatiquement le type réel de chaque feature.
2. Applique la transformation optimale pour chaque type.
3. Retourne X (numérique, sans NaN, sans inf, sans colonne non numérique) et y (encodée si besoin).
4. Évite toute fuite de données : fit uniquement sur le train, transform sur le test.
5. Produit un rapport détaillé et explicable de toutes ses décisions.

# ENTRÉES
- `df` (DataFrame pandas) et `target` (nom de la colonne cible).
- `task` : "classification", "regression" ou "auto" (déduit : cible numérique à > 15 valeurs uniques = régression, sinon classification).
- Un objet de configuration `PreprocessConfig` (dataclass) avec valeurs par défaut sensées : seuil de cardinalité (10), seuil de valeurs manquantes (0.90), seuil de corrélation (0.95), seuil de catégories rares (1 %), seuil d'asymétrie (|skew| > 1), stratégie d'imputation, type de scaler, `random_state=42`, `verbose`.

# ÉTAPE 0 : NETTOYAGE PRÉALABLE
- Normaliser les noms de colonnes (espaces, casse, caractères spéciaux), sans modifier la cible.
- Nettoyer les chaînes : strip, casse homogène, remplacement des faux manquants ("?", "N/A", "null", "none", "", "-", "unknown") par NaN.
- Convertir les nombres stockés en texte ("1 200", "12,5", "45%", "$300") en vrais numériques lorsque > 90 % des valeurs sont convertibles.
- Supprimer les doublons exacts et les lignes dont la cible est manquante (avec message d'information).
- Conversion de dtypes pour réduire la mémoire (downcast) sur les gros datasets.

# DÉTECTION DES TYPES (règles explicites et configurables)
- À supprimer : identifiants (cardinalité ≈ nb de lignes), constantes ou quasi-constantes (> 99 % même valeur), > 90 % de manquants, colonnes dupliquées.
- Binaires : 2 valeurs uniques (y compris Oui/Non, True/False, M/F).
- Numériques continues : int/float à forte cardinalité.
- Numériques discrètes : entiers à faible cardinalité, traités comme catégoriels ordinaux ou nominaux selon le contexte.
- Ordinales : détecter les motifs évidents (low/medium/high, S/M/L/XL, niveaux d'études) et appliquer un OrdinalEncoder avec ordre déduit ; permettre à l'utilisateur de fournir l'ordre via config.
- Catégorielles faible cardinalité (≤ seuil) et haute cardinalité (> seuil).
- Dates/temps : datetime ou chaînes parsables en date (plusieurs formats).
- Texte libre : chaînes longues avec forte cardinalité.
- Chaque colonne reçoit un type final et une justification enregistrée.

# TRANSFORMATIONS
Numériques :
- Indicateur de valeur manquante (`MissingIndicator`) si le taux de NaN dépasse 5 %, car l'absence est parfois informative.
- Imputation par médiane (ou KNN/itérative en option).
- Traitement des outliers par winsorisation (percentiles 1 % / 99 %) ou clipping IQR, appris sur le train uniquement.
- Correction de l'asymétrie : PowerTransformer (Yeo-Johnson) ou log1p si |skew| > seuil.
- Scaling : StandardScaler par défaut, RobustScaler si outliers marqués, MinMaxScaler en option.

Binaires : mapping 0/1 avec mémorisation du mapping.

Catégorielles faible cardinalité :
- Regroupement des catégories rares (< 1 %) dans "__RARE__".
- Imputation par mode ou catégorie "__MISSING__".
- OneHotEncoder(handle_unknown="ignore", drop="if_binary").

Catégorielles haute cardinalité :
- Target Encoding avec validation croisée interne (K-fold, lissage bayésien) pour éviter la fuite, ou Frequency Encoding si la cible n'est pas adaptée.
- Gestion des catégories inconnues au transform (moyenne globale).

Dates :
- Extraction : année, mois, jour, jour de semaine, trimestre, semaine de l'année, heure si présente, est_weekend, est_fin_de_mois, ancienneté en jours par rapport à la date de référence apprise sur le train.
- Encodage cyclique (sin/cos) pour mois, jour de semaine et heure.

Texte libre : TF-IDF limité (max_features configurable) suivi de TruncatedSVD, ou suppression avec avertissement si désactivé.

Cible :
- Classification : LabelEncoder, avec conservation du mapping des classes ; détection du déséquilibre (ratio de la classe minoritaire) et recommandation de `class_weight` ou SMOTE (sans l'appliquer automatiquement).
- Régression : option de transformation log1p / PowerTransformer si la cible est très asymétrique, avec méthode `inverse_transform_target`.

# POST-TRAITEMENT
- Suppression des features à variance nulle ou quasi nulle (`VarianceThreshold`).
- Suppression des features fortement corrélées entre elles (|r| > 0.95), en gardant la plus corrélée à la cible.
- Détection de fuite de cible : alerte si une feature seule a une corrélation / information mutuelle quasi parfaite avec la cible.
- Vérification finale : tout est numérique, sans NaN, sans inf, sans doublon de nom de colonne.

# CONTRAINTES TECHNIQUES
- Utiliser `Pipeline` et `ColumnTransformer` de scikit-learn ; sortie en DataFrame pandas via `set_output(transform="pandas")` pour conserver les noms de colonnes.
- API sklearn : `fit`, `transform`, `fit_transform`, `get_feature_names_out`, `get_params`.
- Méthode `fit_transform_xy(df)` qui sépare automatiquement X et y.
- `save(path)` / `load(path)` via joblib ; versionnage de la config sauvegardée.
- Gestion des datasets volumineux : matrices creuses si le One-Hot dépasse un seuil de colonnes.
- Reproductibilité : `random_state` partout.
- Dépendances : pandas, numpy, scikit-learn, joblib uniquement (SMOTE, category_encoders ou autres uniquement en option, avec import protégé).
- Qualité : type hints, docstrings, PEP8, logging (pas de print), erreurs explicites et messages clairs, fonctions courtes et modulaires.

# SORTIES ATTENDUES
1. Le code complet dans `auto_preprocessor.py`.
2. `report()` : tableau par colonne (type détecté, justification, transformation, colonnes créées, % de manquants, colonnes supprimées et raison), plus un résumé global (nb de features avant/après, avertissements, déséquilibre de classes, fuites suspectes).
3. Des tests `pytest` (`test_auto_preprocessor.py`) sur un dataset synthétique contenant : numériques avec NaN et outliers, catégorielles, ordinales, binaires, dates, haute cardinalité, texte, un ID, une constante, des faux manquants ("?") et des nombres stockés en texte. Vérifier par assertions : X numérique, sans NaN ni inf ; mêmes colonnes en train et test ; catégories inconnues gérées ; sauvegarde/chargement identiques ; aucun fuite (les statistiques viennent du train seulement).
4. Un cas limite testé : dataset minuscule, une seule colonne, cible avec classes rares, colonne entièrement vide.
5. Un exemple complet : `train_test_split` stratifié, fit sur train, transform sur test, entraînement d'un RandomForest, puis comparaison des scores (baseline naïve vs pipeline) pour prouver le gain.
6. Une démonstration sur au moins un dataset réel (ex. Titanic, Adult Income ou California Housing via sklearn) pour valider la généricité.

# CRITÈRES DE QUALITÉ
- Fonctionne sans modification sur un dataset inconnu.
- Zéro fuite de données entre train et test.
- Résultats explicables : chaque décision est tracée dans le rapport.
- Code exécuté et testé avant livraison ; si un test échoue, corrige-le avant de répondre.

# FORMAT DE RÉPONSE
Réponds en français. Structure : (1) bref plan, (2) code complet, (3) résultats réels des tests et de l'exemple, (4) limites connues, (5) pistes d'amélioration (ex. sélection de features, AutoML, détection d'anomalies, support Spark/Polars).
````

### Étape 2 : prompt de benchmark multi-datasets

À utiliser avec `auto_preprocessor.py` attaché. Il produit `benchmark_multi.py` (comparaison naïf / sklearn standard / AutoPreprocessor, test de la cible mélangée).

````text
# RÔLE
Tu es un Data Scientist / ML Engineer senior, expert en benchmarking rigoureux, en prétraitement de données tabulaires et en scikit-learn. Tu privilégies l'honnêteté des comparaisons et la détection des fuites de données.

# CONTEXTE
J'ai un module Python `auto_preprocessor.py` (classe `AutoPreprocessor` + `PreprocessConfig`) qui transforme automatiquement n'importe quel dataset tabulaire supervisé en matrice 100 % numérique. API : `fit_transform_xy(df, target)` pour le train, `transform_xy(df, target)` pour le test, `report()`, `summary()`, `inverse_transform_target()`, `save()/load()`.
Il a déjà passé 10 tests unitaires sur un dataset synthétique. Je veux maintenant le VALIDER sur PLUSIEURS datasets réels et variés, pour prouver qu'il est générique, sans fuite, et qu'il ne dégrade jamais les performances.

# OBJECTIF
Écris un script `benchmark_multi.py` qui exécute le préprocesseur sur au moins 8 datasets réels et produit un rapport comparatif fiable.

# DATASETS (mélange obligatoire)
- Classification binaire, multiclasse et régression.
- Types variés : numériques purs, catégorielles mixtes, valeurs manquantes, dates, haute cardinalité, texte court, classes déséquilibrées.
- Sources sans compte ni clé API : `sklearn.datasets` (california_housing, breast_cancer, wine, diabetes) et `fetch_openml` (titanic, adult, credit-g, house_prices, bank-marketing, diabetes, etc.).
- Chaque dataset est chargé dans un bloc try/except : un échec de téléchargement ne doit pas arrêter le benchmark, il est simplement noté "SKIPPED" dans le rapport.
- Un dictionnaire de configuration par dataset : nom, identifiant OpenML, nom de la cible, tâche, colonnes à exclure.

# PROTOCOLE DE COMPARAISON (équitable)
Pour chaque dataset, comparer 3 approches avec le MÊME modèle et les mêmes hyperparamètres :
A. Baseline naïve : imputation simple + OneHot/ordinal basique, sur les colonnes brutes identiques.
B. AutoPreprocessor avec configuration par défaut.
C. AutoPreprocessor avec `drop_leaky_columns=True`.
Règles impératives :
- Même split pour les 3 approches : validation croisée stratifiée (5 plis) en classification, KFold en régression, avec `random_state=42`.
- Le préprocesseur est RÉAJUSTÉ à l'intérieur de chaque pli (fit sur le pli train uniquement, transform sur le pli validation). Aucune statistique du pli de validation ne doit être utilisée.
- Modèles : RandomForest ET LogisticRegression/Ridge (un modèle à arbres et un modèle linéaire, car le scaling n'aide que le linéaire).
- Métriques : classification binaire = ROC-AUC, accuracy, F1 ; multiclasse = accuracy, F1 macro ; régression = RMSE, R². Toujours moyenne ± écart-type sur les plis.
- Mesurer aussi : temps de fit, temps de transform, nb de features avant/après.

# CONTRÔLES DE FUITE (point critique)
- Avant chaque benchmark, lister les colonnes suspectes d'être des fuites de la cible (ex. sur Titanic : `boat`, `body`) et vérifier que le module les signale dans `report()["fuites_suspectees"]`.
- Si le score AUC > 0.95 ou R² > 0.95 sur un dataset connu pour être difficile, afficher une alerte "RÉSULTAT SUSPECT : vérifier la fuite".
- Faire aussi un test "cible mélangée" (labels permutés au hasard) : le score doit retomber au niveau du hasard (AUC ≈ 0.5, R² ≈ 0). Sinon il y a une fuite dans le pipeline.

# TESTS DE ROBUSTESSE
Sur chaque dataset, vérifier par assertions :
- X transformé 100 % numérique, sans NaN, sans inf, mêmes colonnes en train et test.
- `transform` ne supprime jamais de lignes.
- Catégories inconnues gérées : remplacer 10 % des valeurs catégorielles du test par une modalité inventée, vérifier qu'aucune exception n'est levée.
- Sauvegarde/chargement joblib : sortie identique avant/après.
- Reproductibilité : deux exécutions avec le même `random_state` donnent exactement le même X.

# CONTRAINTES TECHNIQUES
- Environnement : Windows (PowerShell), Python 3.13, pandas, numpy, scikit-learn >= 1.3, scipy, joblib. Aucune autre dépendance obligatoire (matplotlib optionnel, import protégé).
- Compatible avec `py benchmark_multi.py` (pas de chemins Unix, pas de commandes shell).
- Mise en cache des datasets téléchargés (`data_home` local) pour ne pas re-télécharger.
- Ligne de commande : `--datasets titanic,adult` pour en choisir, `--quick` pour un benchmark rapide (3 plis, RandomForest de 50 arbres), `--out resultats` pour le dossier de sortie.
- Logging clair, barre de progression simple par dataset, gestion des warnings sklearn.
- Code PEP8, type hints, docstrings, fonctions courtes.

# SORTIES ATTENDUES
1. `benchmark_multi.py` complet.
2. `resultats/benchmark.csv` : une ligne par (dataset, approche, modèle) avec toutes les métriques (moyenne, écart-type), temps et nb de features.
3. `resultats/rapport.md` : tableau récapitulatif, gain moyen du préprocesseur, victoires/égalités/défaites par rapport à la baseline, liste des fuites détectées, liste des datasets SKIPPED.
4. Graphique `resultats/gains.png` (barres de gain par dataset) si matplotlib est disponible.
5. Une section "Verdict" : sur quels types de datasets le préprocesseur aide, sur lesquels il est neutre, sur lesquels il dégrade, avec une explication.
6. Une section "Bugs trouvés" : toute exception ou comportement anormal rencontré, avec la correction proposée dans `auto_preprocessor.py`.

# CRITÈRES DE QUALITÉ
- Comparaison strictement équitable (mêmes colonnes brutes, même split, même modèle).
- Aucune fuite : préprocesseur réajusté dans chaque pli.
- Ne jamais présenter un gain sans écart-type ni sans vérifier la fuite.
- Si une conclusion est incertaine (différence inférieure à l'écart-type), le dire explicitement au lieu de la présenter comme un gain.
- Code exécuté avant livraison ; si un dataset échoue, corriger ou le marquer SKIPPED avec la raison.

# FORMAT DE RÉPONSE
Réponds en français. Structure : (1) bref plan, (2) code complet, (3) résultats réels de l'exécution, (4) verdict, (5) bugs trouvés et corrections, (6) limites et pistes d'amélioration.
````

### Étape 3 : prompt d'analyse approfondie (volets A, B, C)

À utiliser avec `auto_preprocessor.py` et `benchmark_multi.py` attachés. A : ablation sur credit_g. B : tests statistiques appariés et modèles à régularisation CV. C : verdict tenant compte des fuites. Il produit `ablation_credit_g.py`.

````text
# RÔLE
Tu es un Data Scientist / ML Engineer senior, expert en benchmarking rigoureux, scikit-learn et statistiques appliquées au ML. Tu privilégies l'honnêteté des conclusions : tu ne présentes jamais un écart comme un gain s'il est dans le bruit.

# CONTEXTE
J'ai un module Python `auto_preprocessor.py` (classe `AutoPreprocessor` + `PreprocessConfig`) qui transforme n'importe quel dataset tabulaire supervisé en matrice 100 % numérique. Je te joins aussi `benchmark_multi.py` qui le compare, en validation croisée (réajusté dans chaque pli), à 3 autres approches sur 10 datasets (titanic, breast_cancer, wine, california_housing, diabetes_reg, credit_g, car, pima_diabetes, adult, iris) :
- A : baseline naïve (imputation + OneHot, sans scaling)
- D : pipeline sklearn standard (médiane + StandardScaler + OneHot)
- B : AutoPreprocessor par défaut
- C : AutoPreprocessor avec drop_leaky_columns=True
Modèles : RandomForest et modèle linéaire (Ridge / LogisticRegression, hyperparamètres fixes).
Environnement : Windows (PowerShell), Python 3.13, pandas, numpy, scikit-learn >= 1.3, scipy, joblib, pytest, matplotlib (optionnel). Lancement : `py benchmark_multi.py`. Aucune dépendance supplémentaire.

# RÉSULTATS ACTUELS (à ne pas contredire sans preuve)
- RandomForest : égalité sur 9 datasets sur 10 (le "1 victoire" sur Titanic est faux : B garde la colonne `boat`, fuite de cible).
- Titanic, mode C (`boat` supprimée) : AUC RF = 0.8551.
- Test "cible mélangée" : au niveau du hasard partout, aucune fuite dans le pipeline.
- Modèles linéaires, B moins D : gain réel sur california_housing (RMSE 0.8714 -> 0.7187, la baseline est instable : écart-type 0.29). Écarts dans le bruit sur diabetes, breast_cancer, pima, wine. Légèrement négatifs sur adult (-0.007, OneHot meilleur que Target Encoding pour un modèle linéaire) et credit_g (-0.010 en linéaire, -0.024 en RandomForest : cause inconnue).

# TÂCHES (3 volets, dans cet ordre)

## Volet A : diagnostiquer pourquoi AutoPreprocessor perd sur credit_g
Écris `ablation_credit_g.py` qui charge credit-g (OpenML, version 1) et mesure, avec la même validation croisée (StratifiedKFold 5 plis, random_state=42, RandomForest 100 arbres ET LogisticRegression), le ROC-AUC (moyenne ± écart-type) de AutoPreprocessor avec UNE fonctionnalité désactivée à la fois :
- sans winsorisation (outlier_method="none")
- sans correction d'asymétrie (skew_threshold très grand)
- sans scaling (scaler_type="none")
- sans regroupement des catégories rares (rare_category_threshold=0)
- sans MissingIndicator (missing_indicator_threshold=1.1)
- sans filtre de colinéarité (correlation_threshold=1.01)
- entiers discrets traités comme numériques au lieu d'être One-Hot (adapter la détection de types via une option de config, ou un contournement documenté)
- toutes les options désactivées (doit retrouver le niveau de D)
Compare chaque variante à D (pipeline sklearn standard) et à la configuration par défaut. Conclus : quelle option explique la perte ? Si une option est en cause, propose le correctif minimal dans `auto_preprocessor.py` (diff précis, sans casser les 15 tests pytest existants), puis re-mesure.

## Volet B : conclusions statistiquement solides sur les écarts
Modifie `benchmark_multi.py` pour :
1. Remplacer les modèles linéaires à hyperparamètres fixes par RidgeCV (alphas logspace(-3, 3, 13)) en régression et LogisticRegressionCV (Cs=10, max_iter=2000) en classification, pour que le scaling ne soit pas confondu avec une mauvaise régularisation.
2. Conserver les scores PAR PLI (pas seulement la moyenne) pour chaque approche et modèle.
3. Calculer pour chaque dataset un test apparié B contre D (et B contre A) : différence par pli, moyenne, écart-type, intervalle de confiance à 95 % par bootstrap sur les plis (ou test de Wilcoxon si 5 plis ou plus, en précisant les limites), et un verdict "gain", "perte" ou "indistinguable" selon que l'intervalle exclut 0.
4. Utiliser 5 plis répétés 3 fois (RepeatedStratifiedKFold / RepeatedKFold) quand le temps d'exécution reste raisonnable, sinon expliquer pourquoi non.
5. Écrire ces résultats dans `resultats/benchmark.csv` (nouvelles colonnes) et dans une nouvelle section de `rapport.md`.

## Volet C : bilan correct vis-à-vis des fuites
Dans `benchmark_multi.py` :
1. Pour chaque dataset où le détecteur signale une fuite (`fuites_suspectees` non vide), le bilan Victoire / Égalité / Défaite et le tableau principal doivent utiliser l'approche C (colonnes fuyantes supprimées) et non B. Affiche explicitement une note "B exclu : fuite détectée sur <colonnes>".
2. Le tableau "Comparaison avec drop_leaky_columns=True" doit montrer, pour chaque dataset, le score de A, D, B et C côte à côte.
3. Ajoute au rapport une phrase de verdict automatique par dataset ("AutoPreprocessor aide / neutre / dégrade" avec la raison), fondée sur les tests appariés du volet B, et non sur un simple seuil.

# CONTRAINTES
- Ne modifie pas le comportement de `AutoPreprocessor` sauf pour le correctif minimal du volet A, justifié par les mesures. Les 15 tests de `test_auto_preprocessor.py` doivent rester verts.
- Reproductibilité : random_state=42 partout. Aucun sous-échantillonnage caché.
- Comparaison strictement équitable : même split, mêmes colonnes brutes, préprocesseur réajusté dans chaque pli, `fit_transform` sur le train et `transform` sur la validation.
- Chaque dataset OpenML est dans un try/except : un échec réseau le marque SKIPPED sans arrêter le benchmark.
- Code PEP8, type hints, docstrings, fonctions courtes, logging propre (pas de spam : un avertissement de déséquilibre de classes ne doit s'afficher qu'une fois par dataset).
- Si un dataset OpenML est inaccessible depuis ton environnement, dis-le clairement, teste sur les datasets sklearn et un jeu synthétique, et indique ce qui reste à vérifier chez moi.

# CRITÈRES DE QUALITÉ
- Ne jamais annoncer un gain sans intervalle de confiance. Si l'intervalle contient 0, écris "indistinguable".
- Distinguer clairement "gain dû au scaling" (B égale D) et "gain dû au module" (B meilleur que D).
- Code exécuté avant livraison ; si un test échoue, corrige avant de répondre.

# FORMAT DE RÉPONSE
Réponds en français. Structure : (1) plan en quelques lignes, (2) pour chaque volet : fichiers livrés complets, résultats réels de l'exécution, conclusion, (3) tableau récapitulatif final "où AutoPreprocessor aide, est neutre ou dégrade", (4) limites connues et commandes PowerShell exactes pour tout relancer chez moi.
````

### Étape 4 : prompt de l'application graphique (Tkinter)

À utiliser avec `auto_preprocessor.py` attaché. Il produit `app_preprocess.py`.

````text
# RÔLE
Tu es un développeur Python senior, expert en interfaces Tkinter/ttk, en pandas et en scikit-learn. Tu écris du code robuste, lisible et testé, qui ne fige jamais l'interface.

# CONTEXTE
J'ai un module `auto_preprocessor.py` (classe `AutoPreprocessor`, dataclass `PreprocessConfig`, méthodes `fit_transform`, `transform`, `report()`, `summary()`, `save()`/`load()`). Il nettoie et transforme n'importe quel dataset tabulaire supervisé en matrice numérique, sans fuite de données. Je suis sur Windows 10/11, Python 3.13 lancé avec `py`, et je veux une application de bureau pour l'utiliser sans écrire de code.

# OBJECTIF
Écris `app_preprocess.py`, une application Tkinter (ttk uniquement, aucune dépendance autre que pandas, numpy, scikit-learn, joblib et `auto_preprocessor`) qui, pour un dataset quelconque, fait automatiquement le nettoyage et les transformations nécessaires, puis exporte le résultat.

# FONCTIONNALITÉS
1. Chargement de fichiers : CSV (détection automatique du séparateur et de l'encodage : utf-8, utf-8-sig, latin-1, cp1252), Excel (.xlsx/.xls, via openpyxl, message clair s'il manque), JSON, Parquet si disponible.
2. Onglet « 1. Données » : aperçu du tableau (Treeview avec barres de défilement, limité aux premières lignes), dimensions, nombre de valeurs manquantes par colonne.
3. Onglet « 2. Paramètres » :
   - colonne cible (liste déroulante, option « aucune : sans cible »),
   - type de tâche (auto, classification, régression),
   - part du test en % (0 = aucun ; découpage stratifié en classification),
   - mise à l'échelle (standard, robust, minmax, none), valeurs extrêmes (winsorize, iqr, none),
   - seuils : haute cardinalité, manquants, colinéarité, catégories rares, asymétrie,
   - cases à cocher : transformer la cible asymétrique (régression), supprimer automatiquement les colonnes qui fuient la cible, encoder les entiers à faible cardinalité en One-Hot,
   - bouton « Lancer le prétraitement ».
4. Onglet « 3. Rapport » : rapport texte complet (colonnes supprimées et pourquoi, types détectés, avertissements, fuites suspectées, déséquilibre de classes) et tableau des transformations par colonne.
5. Onglet « 4. Résultat » : aperçu de X_train et X_test transformés, et boutons d'export : `train_transforme.csv`, `test_transforme.csv` (la cible en dernière colonne), `detail_colonnes.csv`, `rapport_pretraitement.txt`, `preprocesseur.joblib`, dans un dossier choisi par l'utilisateur.
6. Application à de nouvelles données : charger un `preprocesseur.joblib` existant et transformer un autre fichier (sans cible) avec les mêmes règles.

# CONTRAINTES TECHNIQUES
- Séparer strictement la logique de l'interface : fonctions pures testables sans Tkinter (`read_table`, `build_config`, `run_preprocessing`, `apply_to_new_data`, `report_text`, `export_result`), dataclasses `Options` et `RunResult`, puis une classe `App`.
- Le traitement tourne dans un thread séparé ; communication avec l'interface par `queue.Queue` consultée avec `after()`. Barre de progression, bouton désactivé pendant le calcul, jamais de gel de la fenêtre.
- Aucune fuite de données : `fit_transform` sur le train uniquement, `transform` sur le test.
- Toute erreur est affichée dans une boîte de dialogue claire (messagebox) avec la cause probable, jamais une trace brute ; le détail va dans un journal.
- Interface en français, redimensionnable, utilisable en 1280x720 ; les noms de fichiers avec accents et espaces doivent fonctionner sous Windows.
- Type hints, docstrings, PEP8. Lancement : `py app_preprocess.py`.

# TESTS À EXÉCUTER AVANT LIVRAISON
- Tester les fonctions de logique sur au moins : un CSV avec `;` et accents, un CSV avec `,`, un fichier sans cible, un dataset de classification déséquilibré, un dataset de régression.
- Vérifier que les fichiers exportés existent, qu'ils sont numériques, sans NaN ni inf, et que train/test ont les mêmes colonnes.
- Vérifier que le joblib rechargé donne exactement les mêmes valeurs sur de nouvelles données.
- Si Tkinter n'est pas disponible dans ton environnement, teste le câblage avec des stubs et dis-le clairement ; ne prétends pas avoir vu la fenêtre.

# FORMAT DE RÉPONSE
Réponds en français. Plan en quelques lignes, puis le fichier complet `app_preprocess.py`, puis les résultats réels des tests, puis les limites connues et la commande PowerShell pour lancer l'application.
````

### Étape 5 : prompts de correction et d'extension

Ces prompts ont été utilisés (en dialogue, avec le résultat réel du rapport collé dans la conversation) pour corriger le module après les premiers essais. Chacun se colle dans une conversation avec `auto_preprocessor.py` attaché. Ils montrent la méthode : donner le symptôme observé, demander la cause avant le correctif, puis un test de non-régression.

**5a. Colonne `heure` supprimée à tort**

````text
Voici le rapport d'AutoPreprocessor sur un dataset de transactions bancaires : la colonne `heure` (valeurs « 22:38:02 », 798 valeurs uniques sur 800 lignes) est supprimée comme DROP_ID. C'est faux : c'est une heure, pas un identifiant. Corrige la détection pour qu'une colonne date/heure quasi unique ne soit jamais classée ID, puis vérifie que `heure` produit des features d'heure (sinus/cosinus). Ajoute ou ajuste un test, relance les tests existants et donne le diff exact.
````

**5b. Colonnes constantes non supprimées**

````text
Dans le fichier exporté, trois colonnes (`heure_month_sin`, `heure_dow_sin`, `heure_dow_cos`) ont un écart-type de 0 mais n'ont pas été supprimées par le filtre de variance. Trouve pourquoi (probablement une comparaison flottante) et corrige : une colonne constante, ou de variance inférieure à 1e-12, doit être retirée. Relance les tests et mesure le nombre de features avant/après sur ce dataset.
````

**5c. Nombres avec unité détectés comme catégories**

````text
Dans `churn_telecom_demo.csv`, la colonne `facture_mensuelle` (« 67,50 MAD ») est détectée comme catégorielle à haute cardinalité (1865 valeurs) au lieu de numérique. Étends le parseur de nombres écrits en texte pour gérer une unité en suffixe (MAD, DH, kg...) et un code devise en préfixe (« MAD 67 »), sans jamais transformer un identifiant comme « C1004 » ou « T000012 » en nombre. Ajoute un test avec ces cas (y compris une date « 2024-01-05 » qui doit rester non numérique) et vérifie sur le dataset que la colonne devient NUMERIC_CONTINUOUS.
````

**5d. Tâche « régression » sur une cible textuelle**

````text
Dans l'application, j'ai laissé « Type de tâche » sur « regression » après avoir chargé un autre fichier, puis ouvert un dataset dont la cible contient « Oui »/« Non ». Erreur : « could not convert string to float: 'Non' ». Corrige à deux endroits : (1) dans le module, si la tâche demandée est « regression » et que la cible contient du texte non convertible, repasser en classification avec un avertissement dans le rapport ; (2) dans l'application, remettre « Type de tâche » sur « auto » à chaque chargement de fichier. Vérifie avec le dataset, puis relance les tests.
````

**5e. Script de vérification du signal**

````text
Écris `signal_check.py <fichier.csv> <cible> [--sep ;]` qui dit si un dataset contient du signal et si AutoPreprocessor aide. Validation croisée stratifiée à 5 plis, préprocesseur refit à chaque pli ; comparer un pipeline sklearn standard (imputation + One-Hot + scaler) et AutoPreprocessor, avec RandomForest (class_weight balanced) et LogisticRegression ; ajouter une ligne « cible mélangée » avec AutoPreprocessor. Afficher AUC (± écart-type) et PR-AUC, plus la valeur du hasard. Termine par une phrase d'interprétation. Teste-le sur un dataset sans signal et sur un dataset avec signal injecté, et dis honnêtement si AutoPreprocessor est meilleur, égal ou moins bon.
````

**5f. Exécutable Windows**

````text
Écris `build_exe.bat` qui installe PyInstaller et construit `dist\AutoPreprocessor.exe` à partir de `app_preprocess.py` (un seul fichier, sans console, avec les sous-modules scikit-learn et scipy nécessaires). Le script doit : vérifier que `app_preprocess.py` et `auto_preprocessor.py` sont présents, supprimer `build` et `dist` avant de construire, s'arrêter avec un message clair si une étape échoue, et vérifier la taille finale (moins de 20 Mo = construction incomplète). Précise les pannes probables : fichier de quelques centaines d'octets (construction dans OneDrive : recommencer dans `C:\AutoPre`), fenêtre qui ne s'ouvre pas (construire une version `--console` et afficher le code de sortie et la durée), antivirus, taille de l'exécutable.
````

### Remarque

Les corrections plus anciennes (détection de fuite par colonne avec règle d'écart, target encoding out-of-fold, désactivation de l'encodage des entiers discrets et de `"none"` comme faux manquant suite au cas credit_g) ont été faites en dialogue à partir des résultats réels ; elles sont déjà intégrées dans le code livré. `ablation_credit_g.py` est produit par le prompt de l'étape 3 et n'est pas inclus dans ce dépôt.