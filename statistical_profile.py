"""Descriptive evidence shared by analysis and numerical execution."""

from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import (KBinsDiscretizer, MaxAbsScaler, MinMaxScaler, PowerTransformer,
                                   QuantileTransformer, RobustScaler, StandardScaler)


ELEMENTWISE = {"LOG": np.log, "LOG1P": np.log1p, "SQRT": np.sqrt, "CBRT": np.cbrt}
FITTED = {"BOX_COX", "YEO_JOHNSON", "QUANTILE_NORMAL", "QUANTILE_UNIFORM", "RANK", "DISCRETIZE"}
DISTRIBUTIONS = ["AUTO", "NONE", *ELEMENTWISE, "BOX_COX", "YEO_JOHNSON",
                 "QUANTILE_NORMAL", "QUANTILE_UNIFORM", "RANK", "DISCRETIZE"]
SCALERS = {"STANDARD": StandardScaler, "ROBUST": RobustScaler,
           "MINMAX": MinMaxScaler, "MAXABS": MaxAbsScaler}
OUTLIER_ACTIONS = ["NONE", "WINSORIZE", "IQR", "Z_SCORE", "MODIFIED_Z", "MEDIAN_REPLACE", "TO_MISSING"]


def validate_distribution(values, action, column, *, fitting=True):
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{action} impossible sur {column!r} : valeurs non finies.")
    minimum = values.min()
    if ((action in {"LOG", "BOX_COX"} and minimum <= 0)
            or (action == "LOG1P" and minimum <= -1)
            or (action == "SQRT" and minimum < 0)):
        raise ValueError(f"{action} impossible sur {column!r} : minimum {minimum:g} hors domaine.")
    if fitting and action in FITTED and values.nunique() < 2:
        raise ValueError(f"{action} impossible sur {column!r} : deux valeurs distinctes requises.")


class EmpiricalRank:
    """Midrank of training ties, interpolated for unseen values."""

    def fit(self, frame):
        self.values_, counts = np.unique(frame.to_numpy(dtype=float), return_counts=True)
        self.ranks_ = (np.cumsum(counts) - counts / 2) / counts.sum()
        return self

    def transform(self, frame):
        return np.interp(frame.to_numpy(dtype=float), self.values_, self.ranks_, left=0, right=1)


def fit_distribution(values, action, random_state):
    validate_distribution(values, action, values.name)
    if action in ELEMENTWISE:
        return action
    if action == "RANK":
        return EmpiricalRank().fit(values.to_frame())
    if action == "DISCRETIZE":
        return KBinsDiscretizer(n_bins=min(5, values.nunique()), encode="ordinal", strategy="uniform",
                                subsample=None).fit(values.to_frame())
    if action.startswith("QUANTILE_"):
        return QuantileTransformer(
            n_quantiles=min(1000, len(values)),
            output_distribution=action.split("_")[1].lower(),
            random_state=random_state,
        ).fit(values.to_frame())
    return PowerTransformer(method="box-cox" if action == "BOX_COX" else "yeo-johnson").fit(values.to_frame())


def apply_distribution(values, transformer, action):
    validate_distribution(values, action if action in ELEMENTWISE or action == "BOX_COX" else "NONE",
                          values.name, fitting=False)
    if isinstance(transformer, str):
        result = ELEMENTWISE[transformer](values).to_numpy()
    else:
        result = transformer.transform(values.to_frame()).ravel()
    if not np.isfinite(result).all():
        raise ValueError(f"{action} produit des valeurs non finies sur {values.name!r}.")
    return result


def numerical_comparison(values, random_state):
    values = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if values.empty:
        return {}
    sample = values.sample(5000, random_state=random_state) if len(values) > 5000 else values
    results = {}
    for action in DISTRIBUTIONS[1:]:
        try:
            validate_distribution(values, action, values.name)
            if action == "NONE":
                transformed = sample
            else:
                fitted = fit_distribution(sample, action, random_state)
                transformed = pd.Series(apply_distribution(sample, fitted, action))
            results[action] = {
                "applicable": True, "effectif": len(sample), "asymetrie_avant": float(sample.skew()),
                "asymetrie_apres": float(transformed.skew()), "variance_avant": float(sample.var()),
                "variance_apres": float(transformed.var()),
                "perte_valeurs_distinctes": max(0, sample.nunique() - transformed.nunique()),
                "resultat_constant": transformed.nunique() == 1,
            }
        except (ValueError, FloatingPointError, OverflowError) as exc:
            results[action] = {"applicable": False, "avertissement": str(exc)}
    return results


def outlier_plan(values, action, lower, upper, multiplier):
    if values.empty or action == "NONE":
        return -np.inf, np.inf
    if action == "WINSORIZE":
        return tuple(values.quantile([lower, upper]))
    if action in {"Z_SCORE", "MODIFIED_Z"}:
        center = values.mean() if action == "Z_SCORE" else values.median()
        spread = (3 * values.std(ddof=0) if action == "Z_SCORE"
                  else 3.5 * 1.4826 * (values - center).abs().median())
        return (center - spread, center + spread) if spread > 0 else (-np.inf, np.inf)
    q1, q3 = values.quantile([.25, .75])
    spread = multiplier * (q3 - q1)
    return q1 - spread, q3 + spread


def apply_outlier_plan(values, action, bounds, median):
    if action in {"MEDIAN_REPLACE", "TO_MISSING"}:
        return values.mask(values.lt(bounds[0]) | values.gt(bounds[1]),
                           median if action == "MEDIAN_REPLACE" else np.nan)
    return values.clip(*bounds)


def numerical_cleaning_comparison(values, lower, upper, multiplier):
    if values.empty:
        return {}, {}
    outliers = {}
    for action in OUTLIER_ACTIONS:
        bounds = outlier_plan(values, action, lower, upper, multiplier)
        transformed = apply_outlier_plan(values, action, bounds, values.median())
        outliers[action] = {
            "bornes": bounds, "pct_modifie": float((values != transformed).mean() * 100),
            "variance_avant": float(values.var()), "variance_apres": float(transformed.var()),
            "asymetrie_avant": float(values.skew()), "asymetrie_apres": float(transformed.skew()),
            "perte_valeurs_distinctes": max(0, values.nunique() - transformed.nunique()),
        }
    scalers = {}
    for action, scaler in SCALERS.items():
        transformed = pd.Series(scaler().fit_transform(values.to_frame()).ravel())
        scalers[action] = {
            "min_apres": float(transformed.min()), "max_apres": float(transformed.max()),
            "variance_avant": float(values.var()), "variance_apres": float(transformed.var()),
        }
    return outliers, scalers


def adjusted_p_values(p_values):
    """Benjamini–Hochberg over the tests with finite p-values."""
    finite = sorted((p, name) for name, p in p_values.items() if np.isfinite(p))
    corrected = {}
    previous = 1.0
    for rank in range(len(finite), 0, -1):
        p, name = finite[rank - 1]
        previous = min(previous, p * len(finite) / rank)
        corrected[name] = previous
    return corrected


def association(x, y, x_numeric, y_numeric, *, kendall=True):
    pairs = pd.DataFrame({"x": x.reset_index(drop=True), "y": y.reset_index(drop=True)}).dropna()
    if x_numeric:
        pairs["x"] = pd.to_numeric(pairs["x"], errors="coerce")
    if y_numeric:
        pairs["y"] = pd.to_numeric(pairs["y"], errors="coerce")
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    x, y = pairs["x"], pairs["y"]
    n = len(pairs)
    result = {"effectif": n, "tests": {}, "p_primaire": np.nan}
    if n < 8 or min(x.nunique(), y.nunique()) < 2:
        return result
    tests = result["tests"]

    def add(name, test):
        tests[name] = {"effet": float(test.statistic), "p": float(test.pvalue)}

    if x_numeric and y_numeric:
        add("spearman", stats.spearmanr(x, y))
        if kendall:
            pearson = stats.pearsonr(x, y)
            add("pearson", pearson)
            tests["pearson"]["ic_95"] = tuple(pearson.confidence_interval())
            add("kendall", stats.kendalltau(x, y))
        primary = "spearman"
    elif x_numeric or y_numeric:
        values, labels = (x, y) if x_numeric else (y, x)
        if labels.nunique() > 40:
            return {**result, "limite": "Plus de 40 groupes pour une comparaison robuste."}
        groups = [group.to_numpy() for _, group in values.groupby(labels, observed=True)]
        if min(map(len, groups)) < 5:
            return {**result, "limite": "Au moins un groupe compte moins de cinq observations."}
        statistic, p = stats.kruskal(*groups)
        tests["kruskal"] = {"effet": float(max(0, (statistic - len(groups) + 1) / (n - len(groups)))),
                            "statistique": float(statistic), "p": float(p)}
        if len(groups) == 2:
            add("point_biserial", stats.pointbiserialr(pd.factorize(labels, sort=True)[0], values))
        primary = "kruskal"
    else:
        if x.nunique() * y.nunique() > 100_000:
            return {**result, "limite": "Table de contingence supérieure à 100 000 cellules."}
        table = pd.crosstab(x.astype(str), y.astype(str))
        if min(table.shape) < 2:
            return result
        statistic, p, _, expected = stats.chi2_contingency(table, correction=False)
        if (expected < 1).any() or (expected < 5).mean() > .2:
            if table.shape != (2, 2):
                return {**result, "limite": "Effectifs théoriques insuffisants pour le khi-deux."}
            p = stats.fisher_exact(table.to_numpy()).pvalue
            primary = "fisher"
        else:
            primary = "khi_deux"
        tests[primary] = {"effet": float(np.sqrt(statistic / (n * (min(table.shape) - 1)))),
                          "statistique": float(statistic), "p": float(p)}
    result["p_primaire"] = tests[primary]["p"]
    result["effet_primaire"] = tests[primary]["effet"]
    return result


def missingness_evidence(frame, random_state):
    sampled = frame.sample(5000, random_state=random_state) if len(frame) > 5000 else frame
    results = {}
    for column in sampled.columns:
        missing = sampled[column].isna()
        if missing.nunique() < 2:
            continue
        tests = {}
        for other in sampled.columns:
            if other == column:
                continue
            values = sampled[other]
            numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
            tests[str(other)] = association(missing, values, False, numeric, kendall=False)
        corrected = adjusted_p_values({name: result["p_primaire"] for name, result in tests.items()})
        supported = {name: {"q": q, "effet": tests[name]["effet_primaire"]}
                     for name, q in corrected.items() if q <= .05}
        results[column] = {
            "statut": "MAR plausible" if supported else "indéterminé",
            "associations": supported, "effectif": len(sampled), "tests_valides": len(corrected),
            "colonnes_examinees": len(tests),
            "avertissement": "MCAR peut être compatible en l’absence d’association détectée. "
                             "Le mécanisme MNAR n’est pas identifiable à partir des seules données observées.",
        }
    return results


def semantic_type(series, kind, column_profile):
    if kind == "BINARY":
        if pd.api.types.is_bool_dtype(series):
            return "BOOLEAN"
        return "NUMERIC_BINARY" if pd.api.types.is_numeric_dtype(series) else "CATEGORICAL_BINARY"
    if kind == "NUMERIC_CONTINUOUS":
        values = series.replace([np.inf, -np.inf], np.nan).dropna()
        name = str(series.name).casefold()
        if len(values) and name in {"lat", "latitude", "lon", "lng", "longitude"}:
            limit = 90 if name in {"lat", "latitude"} else 180
            if values.between(-limit, limit).all():
                return "GEOGRAPHIC_COORDINATE"
        if len(values) and values.mod(1).eq(0).all():
            return "NUMERIC_DISCRETE"
    if kind.startswith("CATEGORICAL"):
        sample = series.dropna().head(2000)
        parsed = pd.to_numeric(sample, errors="coerce").notna()
        if .1 <= parsed.mean() <= .9:
            return "MIXED"
    if column_profile.get("quasi_constante"):
        return "NEAR_CONSTANT"
    return kind


def text_evidence(series, random_state):
    sample = series.dropna()
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=random_state)
    strings = sample.astype(str)
    lengths = strings.str.len()
    documents = strings.str.casefold().str.findall(r"[^\W\d_]+")
    counts = Counter(token for document in documents for token in document)
    bigrams = Counter(pair for document in documents for pair in zip(document, document[1:]))
    token_counts = documents.str.len()
    return {
        "effectif_texte": len(sample), "longueur_moyenne": float(lengths.mean()),
        "longueur_mediane": float(lengths.median()), "longueur_min": float(lengths.min()),
        "longueur_max": float(lengths.max()), "mots_moyens": float(token_counts.mean()),
        "mots_mediane": float(token_counts.median()), "vocabulaire": len(counts),
        "diversite_lexicale": len(counts) / max(sum(counts.values()), 1),
        "termes_frequents": dict(counts.most_common(10)),
        "bigrammes_frequents": {" ".join(pair): count for pair, count in bigrams.most_common(10)},
        "part_phrases": float(token_counts.ge(4).mean()),
        "part_ponctuee": float(strings.str.contains(r"[.!?](?:\s|$)", regex=True).mean()),
        "part_codes": float(strings.str.fullmatch(r"[\w-]+").mean()),
    }


def parse_dates(series):
    """Parse DMY dates and anchor clock-only observations to a fixed date."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
    strings = series.astype("string").str.strip()
    clock = strings.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?", na=False)
    strings = strings.mask(clock, "1970-01-01 " + strings)
    iso = strings.str.match(r"\d{4}[-/]", na=False)
    parsed = pd.to_datetime(strings.mask(iso), errors="coerce", format="mixed", dayfirst=True, utc=True)
    parsed.loc[iso] = pd.to_datetime(strings.loc[iso], errors="coerce", format="mixed", utc=True)
    return parsed.dt.tz_localize(None)


def profile(raw, clean, kind, rare_threshold, random_state):
    observed = clean.dropna()
    counts = observed.value_counts()
    unique = len(counts)
    n = len(observed)
    concentration = float(counts.iloc[0] / n) if n else 0.0
    result = {
        "effectif": n, "manquants_bruts": int(raw.isna().sum()),
        "manquants": int(clean.isna().sum()), "cardinalite": unique,
        "ratio_cardinalite": unique / max(n, 1),
        "doublons_valeurs": n - unique, "concentration": concentration,
        "quasi_constante": unique > 1 and n >= 30 and concentration >= .98,
    }
    if kind == "DATETIME":
        dates = parse_dates(observed).dropna()
        gaps = dates.sort_values().drop_duplicates().diff().dropna()
        result.update(dates_invalides=n - len(dates), taux_conversion=len(dates) / max(n, 1),
                      date_min=dates.min(), date_max=dates.max(), duree=dates.max() - dates.min(),
                      dates_dupliquees=int(dates.duplicated().sum()), intervalle_median=gaps.median(),
                      annees=dates.dt.year.value_counts().head(100).to_dict(),
                      mois=dates.dt.month.value_counts().to_dict(),
                      jours=dates.dt.day.value_counts().to_dict(),
                      jours_semaine=dates.dt.dayofweek.value_counts().to_dict())
    elif pd.api.types.is_numeric_dtype(clean) and not pd.api.types.is_bool_dtype(clean):
        infinite = int(observed.isin([np.inf, -np.inf]).sum())
        values = observed.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        q1, q2, q3 = values.quantile([.25, .5, .75])
        iqr = q3 - q1
        outlier = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
        skew, kurtosis = float(values.skew()), float(values.kurt())
        shapes = []
        if values.nunique() <= 1:
            shapes.append("constante" if len(values) else "sans valeurs finies")
        elif len(values) >= 30:
            if abs(skew) >= 1:
                shapes.append("asymétrie à droite" if skew > 0 else "asymétrie à gauche")
            if kurtosis >= 3:
                shapes.append("queues lourdes")
            if values.nunique() >= 8 and abs(skew) <= .3 and abs(kurtosis) <= .5:
                shapes.append("moments compatibles avec une forme normale")
            if values.nunique() >= 8 and abs(skew) <= .3 and abs(kurtosis + 1.2) <= .35:
                shapes.append("moments compatibles avec une forme uniforme")
            if values.min() >= 0 and abs(skew - 2) <= .5 and abs(kurtosis - 6) <= 2:
                shapes.append("moments compatibles avec une forme exponentielle")
        if (values == 0).mean() >= .5 and values.nunique() > 1:
            shapes.append("forte masse à zéro")
        if result["quasi_constante"]:
            shapes.append("forte concentration")
        extreme = (values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)
        result.update(moyenne=float(values.mean()), mediane=float(q2), q1=float(q1), q3=float(q3),
                      mode=counts.index[0] if n else None, variance=float(values.var()),
                      ecart_type=float(values.std()), min=float(values.min()), max=float(values.max()),
                      etendue=float(values.max() - values.min()), iqr=float(iqr),
                      coefficient_variation=float(values.std() / abs(values.mean())) if values.mean() else np.nan,
                      asymetrie=skew, kurtosis=kurtosis, infinis=infinite,
                      pct_zero=float((values == 0).mean() * 100), pct_negatif=float(values.lt(0).mean() * 100),
                      pct_positif=float(values.gt(0).mean() * 100),
                      pct_atypique_iqr=float(outlier.mean() * 100),
                      pct_atypique_extreme=float(extreme.mean() * 100), formes=shapes)
    else:
        proportions = counts / max(n, 1)
        result.update(frequences_principales=counts.head(20).to_dict(),
                      modalite_plus_frequente=counts.index[0] if n else None,
                      modalite_moins_frequente=counts.index[-1] if n else None,
                      entropie=float(-(proportions * np.log2(proportions)).sum()),
                      ratio_desequilibre=float(counts.iloc[0] / counts.iloc[-1]) if n else np.nan,
                      part_observations_rares=float(proportions[proportions < rare_threshold].sum()))
        if kind == "TEXT":
            result.update(text_evidence(clean, random_state))
            result["vides_bruts"] = int(raw.dropna().astype(str).str.strip().eq("").sum())
    return result
