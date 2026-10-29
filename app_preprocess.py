"""
AutoPreprocessor GUI : nettoyage et transformation automatiques de n'importe quel dataset (Tkinter).

Utilisation :  py app_preprocess.py
Prérequis   :  auto_preprocessor.py dans le même dossier ; pandas, numpy, scikit-learn, scipy, joblib.
               (Excel : py -m pip install openpyxl)

Flux : ouvrir un fichier -> choisir la cible et les options -> lancer -> lire le rapport -> exporter.
La logique (sans interface) est regroupée dans la première moitié du fichier ; l'interface Tkinter
est dans la classe App, en seconde moitié.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import os
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from auto_preprocessor import AutoPreprocessor, PreprocessConfig

NO_TARGET_LABEL = "(aucune : sans cible)"


# =============================================================================
# 1. LOGIQUE (SANS INTERFACE)
# =============================================================================

def _sniff_separator(sample: str) -> str:
    """Devine le séparateur d'un fichier texte (',', ';', tabulation, '|')."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_table(path: str) -> pd.DataFrame:
    """Lit un fichier tabulaire : CSV/TXT/TSV (séparateur et encodage détectés), Excel, JSON, Parquet."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise RuntimeError("Lecture Excel impossible : installez openpyxl (py -m pip install openpyxl).") from exc
    if ext == ".json":
        return pd.read_json(path)
    if ext == ".parquet":
        return pd.read_parquet(path)

    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                sample = handle.read(50000)
            sep = _sniff_separator(sample)
            return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Impossible de lire le fichier (encodage) : {last_error}")


@dataclass
class Options:
    """Options choisies dans l'interface (valeurs par défaut = celles d'AutoPreprocessor)."""
    task: str = "auto"                     # auto | classification | regression
    test_size_pct: int = 20                # 0 = pas de découpage train/test
    scaler_type: str = "standard"          # standard | robust | minmax | none
    outlier_method: str = "winsorize"      # winsorize | iqr | none
    cardinality_threshold: int = 10
    missing_threshold: float = 0.90
    correlation_threshold: float = 0.95
    rare_category_threshold: float = 0.01
    skew_threshold: float = 1.0
    transform_target: bool = True
    drop_leaky_columns: bool = False
    encode_discrete_as_categorical: bool = False
    random_state: int = 42


def build_config(opts: Options) -> PreprocessConfig:
    """Construit un PreprocessConfig en ignorant les options inconnues de la version du module."""
    wanted = {
        "scaler_type": opts.scaler_type,
        "outlier_method": opts.outlier_method,
        "cardinality_threshold": opts.cardinality_threshold,
        "missing_threshold": opts.missing_threshold,
        "correlation_threshold": opts.correlation_threshold,
        "rare_category_threshold": opts.rare_category_threshold,
        "skew_threshold": opts.skew_threshold,
        "transform_target": opts.transform_target,
        "drop_leaky_columns": opts.drop_leaky_columns,
        "encode_discrete_as_categorical": opts.encode_discrete_as_categorical,
        "random_state": opts.random_state,
        "verbose": 0,
    }
    valid = {f.name for f in dataclasses.fields(PreprocessConfig)}
    return PreprocessConfig(**{k: v for k, v in wanted.items() if k in valid})


@dataclass
class RunResult:
    """Résultat d'un prétraitement."""
    prep: AutoPreprocessor
    target: Optional[str]
    X_train: pd.DataFrame
    y_train: Optional[np.ndarray]
    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[np.ndarray] = None
    y_train_original: Optional[pd.Series] = None
    y_test_original: Optional[pd.Series] = None
    notes: List[str] = field(default_factory=list)
    n_rows_in: int = 0
    n_cols_in: int = 0


def _looks_like_classification(y: pd.Series, task: str) -> bool:
    if task == "classification":
        return True
    if task == "regression":
        return False
    return (not pd.api.types.is_numeric_dtype(y)) or y.nunique(dropna=True) <= 15


def _original_target(df: pd.DataFrame, X: pd.DataFrame, target: str) -> Optional[pd.Series]:
    """Cible d'origine alignée sur les lignes de X (si l'index est retrouvable)."""
    try:
        if X.index.isin(df.index).all():
            return df.loc[X.index, target]
    except Exception:  # noqa: BLE001 - alignement facultatif
        pass
    return None


def run_preprocessing(df: pd.DataFrame, target: Optional[str], opts: Options) -> RunResult:
    """Ajuste AutoPreprocessor sur le train uniquement, puis transforme le test (aucune fuite)."""
    notes: List[str] = []
    config = build_config(opts)
    prep = AutoPreprocessor(config=config, task=opts.task)
    n_rows_in, n_cols_in = df.shape

    # Cas non supervisé : pas de cible
    if not target:
        X = prep.fit_transform(df)
        notes.append("Mode sans cible : aucun découpage train/test, encodages non supervisés.")
        return RunResult(prep, None, X, None, notes=notes, n_rows_in=n_rows_in, n_cols_in=n_cols_in)

    if target not in df.columns:
        raise ValueError(f"La colonne cible '{target}' est absente du fichier.")

    n_missing = int(df[target].isna().sum())
    data = df.dropna(subset=[target]).copy()
    if n_missing:
        notes.append(f"{n_missing} ligne(s) sans valeur de cible supprimée(s) avant le traitement.")
    if len(data) < 10:
        raise ValueError("Moins de 10 lignes avec une cible renseignée : traitement impossible.")

    train_df, test_df = data, None
    if opts.test_size_pct > 0:
        stratify = data[target] if _looks_like_classification(data[target], opts.task) else None
        try:
            train_df, test_df = train_test_split(
                data, test_size=opts.test_size_pct / 100.0, random_state=opts.random_state, stratify=stratify
            )
        except ValueError:
            train_df, test_df = train_test_split(
                data, test_size=opts.test_size_pct / 100.0, random_state=opts.random_state
            )
            notes.append("Découpage non stratifié (une classe est trop rare pour être stratifiée).")

    X_train, y_train = prep.fit_transform_xy(train_df, target=target)
    y_train_original = None
    if prep.target_stats_.get("transformed", False):
        y_train_original = _original_target(train_df, X_train, target)
        notes.append("La cible a été transformée (asymétrie) : utilisez inverse_transform_target() pour revenir à l'échelle d'origine.")

    X_test = y_test = y_test_original = None
    if test_df is not None:
        X_test, y_test = prep.transform_xy(test_df, target=target)
        if prep.task_resolved_ == "classification" and np.any(np.asarray(y_test) == -1):
            notes.append("Le test contient des classes absentes du train : elles sont codées -1.")
        if prep.target_stats_.get("transformed", False):
            y_test_original = _original_target(test_df, X_test, target)

    return RunResult(
        prep, target, X_train, np.asarray(y_train), X_test,
        None if y_test is None else np.asarray(y_test),
        y_train_original, y_test_original, notes, n_rows_in, n_cols_in,
    )


def apply_to_new_data(prep: AutoPreprocessor, df: pd.DataFrame) -> pd.DataFrame:
    """Applique un préprocesseur déjà ajusté à de nouvelles données (la colonne cible est ignorée si présente)."""
    target = getattr(prep, "target_col_", None)
    if target and target in df.columns:
        X, y = prep.transform_xy(df, target=target)
        out = X.copy()
        out[target if target not in out.columns else f"{target}_cible"] = np.asarray(y)
        return out
    return prep.transform(df)


def report_text(result: RunResult) -> str:
    """Rapport lisible : résumé, décisions colonne par colonne, fuites, classes de la cible."""
    prep = result.prep
    rep = prep.report()
    lines = [prep.summary(), ""]
    lines.append(f"Fichier : {result.n_rows_in} lignes x {result.n_cols_in} colonnes")
    lines.append(f"Train   : {result.X_train.shape[0]} lignes x {result.X_train.shape[1]} features numériques")
    if result.X_test is not None:
        lines.append(f"Test    : {result.X_test.shape[0]} lignes x {result.X_test.shape[1]} features numériques")
    if result.target:
        lines.append(f"Cible   : {result.target} ({rep['tache_detectee']})")
    classes = rep["analyse_cible"].get("classes")
    if classes:
        mapping = ", ".join(f"{i} = {c}" for i, c in enumerate(classes))
        lines.append(f"Classes encodées : {mapping}")
    for note in result.notes:
        lines.append(f"Note : {note}")
    scores = rep.get("scores_fuite_par_colonne") or {}
    if scores:
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append("Test de fuite (score d'une colonne seule, 5 plus hauts) : "
                     + ", ".join(f"{c}={s:.3f}" for c, s in top))
    lines += ["", "Détail des transformations par colonne :",
              rep["colonnes_detaillees"].to_string(index=False)]
    return "\n".join(lines)


def _with_target(X: pd.DataFrame, y: Optional[np.ndarray], target: Optional[str],
                 y_original: Optional[pd.Series]) -> pd.DataFrame:
    out = X.copy()
    if y is not None and target:
        name = target if target not in out.columns else f"{target}_cible"
        out[name] = np.asarray(y)
        if y_original is not None:
            out[f"{target}_original"] = np.asarray(y_original)
    return out


def export_result(result: RunResult, folder: str) -> List[str]:
    """Écrit les CSV transformés, le rapport, le détail par colonne et le préprocesseur. Retourne les chemins."""
    os.makedirs(folder, exist_ok=True)
    written: List[str] = []

    train_path = os.path.join(folder, "train_transforme.csv")
    _with_target(result.X_train, result.y_train, result.target, result.y_train_original).to_csv(
        train_path, index=False, encoding="utf-8-sig")
    written.append(train_path)

    if result.X_test is not None:
        test_path = os.path.join(folder, "test_transforme.csv")
        _with_target(result.X_test, result.y_test, result.target, result.y_test_original).to_csv(
            test_path, index=False, encoding="utf-8-sig")
        written.append(test_path)

    detail_path = os.path.join(folder, "detail_colonnes.csv")
    result.prep.report()["colonnes_detaillees"].to_csv(detail_path, index=False, encoding="utf-8-sig")
    written.append(detail_path)

    report_path = os.path.join(folder, "rapport_pretraitement.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"Généré le {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n{report_text(result)}\n")
    written.append(report_path)

    prep_path = os.path.join(folder, "preprocesseur.joblib")
    result.prep.save(prep_path)
    written.append(prep_path)
    return written


# =============================================================================
# 2. INTERFACE TKINTER
# =============================================================================

def _import_tk():
    """Import différé : la logique ci-dessus reste utilisable (et testable) sans Tkinter."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    return tk, ttk, filedialog, messagebox


class App:
    """Fenêtre principale."""

    PREVIEW_ROWS = 200
    PREVIEW_COLS = 60

    def __init__(self) -> None:
        tk, ttk, filedialog, messagebox = _import_tk()
        self.tk, self.ttk, self.filedialog, self.messagebox = tk, ttk, filedialog, messagebox

        self.root = tk.Tk()
        self.root.title("AutoPreprocessor : nettoyage et transformation automatiques")
        self.root.geometry("1150x720")
        self.root.minsize(900, 600)

        self.df: Optional[pd.DataFrame] = None
        self.path: Optional[str] = None
        self.result: Optional[RunResult] = None
        self.prep: Optional[AutoPreprocessor] = None   # préprocesseur actif (ajusté ou chargé)
        self._queue: "queue.Queue[tuple]" = queue.Queue()

        self._build_variables()
        self._build_layout()

    # ------------------------------------------------------------------ variables
    def _build_variables(self) -> None:
        tk = self.tk
        self.v_file = tk.StringVar(value="Aucun fichier chargé")
        self.v_target = tk.StringVar(value=NO_TARGET_LABEL)
        self.v_task = tk.StringVar(value="auto")
        self.v_test = tk.StringVar(value="20")
        self.v_scaler = tk.StringVar(value="standard")
        self.v_outlier = tk.StringVar(value="winsorize")
        self.v_card = tk.StringVar(value="10")
        self.v_missing = tk.StringVar(value="0.90")
        self.v_corr = tk.StringVar(value="0.95")
        self.v_rare = tk.StringVar(value="0.01")
        self.v_skew = tk.StringVar(value="1.0")
        self.v_transform_target = tk.BooleanVar(value=True)
        self.v_drop_leaky = tk.BooleanVar(value=False)
        self.v_discrete_cat = tk.BooleanVar(value=False)
        self.v_status = tk.StringVar(value="Ouvrez un fichier pour commencer.")

    # ------------------------------------------------------------------ mise en page
    def _build_layout(self) -> None:
        ttk = self.ttk
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="Ouvrir un fichier (CSV, Excel, JSON...)", command=self._open_file).pack(side="left")
        ttk.Label(top, textvariable=self.v_file).pack(side="left", padx=10)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_params = ttk.Frame(self.notebook, padding=10)
        self.tab_report = ttk.Frame(self.notebook)
        self.tab_result = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="1. Données")
        self.notebook.add(self.tab_params, text="2. Paramètres")
        self.notebook.add(self.tab_report, text="3. Rapport")
        self.notebook.add(self.tab_result, text="4. Résultat")

        self._build_tab_data()
        self._build_tab_params()
        self._build_tab_report()
        self._build_tab_result()

        bottom = ttk.Frame(self.root, padding=(8, 2))
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, mode="indeterminate", length=160)
        self.progress.pack(side="left")
        ttk.Label(bottom, textvariable=self.v_status).pack(side="left", padx=10)

    def _make_tree(self, parent):
        ttk = self.ttk
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, show="headings")
        ys = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        return tree

    def _build_tab_data(self) -> None:
        self.lbl_info = self.ttk.Label(self.tab_data, text="", padding=6)
        self.lbl_info.pack(fill="x")
        self.tree_data = self._make_tree(self.tab_data)

    def _build_tab_params(self) -> None:
        ttk = self.ttk
        box = ttk.LabelFrame(self.tab_params, text="Cible et découpage", padding=8)
        box.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        ttk.Label(box, text="Colonne cible :").grid(row=0, column=0, sticky="w")
        self.cb_target = ttk.Combobox(box, textvariable=self.v_target, state="readonly", width=32,
                                      values=[NO_TARGET_LABEL])
        self.cb_target.grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(box, text="Type de tâche :").grid(row=1, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.v_task, state="readonly", width=16,
                     values=["auto", "classification", "regression"]).grid(row=1, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(box, text="Part du test (%), 0 = aucun :").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(box, textvariable=self.v_test, from_=0, to=50, increment=5, width=6).grid(
            row=2, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(box, text="Le préprocesseur est ajusté sur le train seulement (aucune fuite vers le test).",
                  foreground="#555").grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        opt = ttk.LabelFrame(self.tab_params, text="Nettoyage et transformations", padding=8)
        opt.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        rows = [
            ("Mise à l'échelle :", ttk.Combobox(opt, textvariable=self.v_scaler, state="readonly", width=12,
                                                 values=["standard", "robust", "minmax", "none"])),
            ("Valeurs extrêmes :", ttk.Combobox(opt, textvariable=self.v_outlier, state="readonly", width=12,
                                                 values=["winsorize", "iqr", "none"])),
            ("Seuil haute cardinalité :", ttk.Spinbox(opt, textvariable=self.v_card, from_=2, to=200, width=8)),
            ("Seuil de manquants (0 à 1) :", ttk.Spinbox(opt, textvariable=self.v_missing, from_=0.1, to=1.0,
                                                          increment=0.05, width=8)),
            ("Seuil de colinéarité (|r|) :", ttk.Spinbox(opt, textvariable=self.v_corr, from_=0.5, to=1.0,
                                                          increment=0.01, width=8)),
            ("Seuil catégories rares :", ttk.Spinbox(opt, textvariable=self.v_rare, from_=0.0, to=0.2,
                                                      increment=0.005, width=8)),
            ("Seuil d'asymétrie :", ttk.Spinbox(opt, textvariable=self.v_skew, from_=0.1, to=5.0,
                                                 increment=0.1, width=8)),
        ]
        for i, (label, widget) in enumerate(rows):
            ttk.Label(opt, text=label).grid(row=i, column=0, sticky="w")
            widget.grid(row=i, column=1, sticky="w", padx=6, pady=2)
        checks = [
            (self.v_transform_target, "Transformer la cible si elle est très asymétrique (régression)"),
            (self.v_drop_leaky, "Supprimer automatiquement les colonnes qui fuient la cible"),
            (self.v_discrete_cat, "Entiers à faible cardinalité : encodage One-Hot (sinon numériques)"),
        ]
        for j, (var, text) in enumerate(checks, start=len(rows)):
            ttk.Checkbutton(opt, text=text, variable=var).grid(row=j, column=0, columnspan=2, sticky="w", pady=2)

        self.btn_run = ttk.Button(self.tab_params, text="▶  Lancer le prétraitement", command=self._start_run)
        self.btn_run.grid(row=1, column=0, columnspan=2, pady=14)

    def _build_tab_report(self) -> None:
        ttk, tk = self.ttk, self.tk
        frame = ttk.Frame(self.tab_report)
        frame.pack(fill="both", expand=True)
        self.txt_report = tk.Text(frame, wrap="none", font=("Consolas", 9))
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.txt_report.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=self.txt_report.xview)
        self.txt_report.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        self.txt_report.pack(fill="both", expand=True)

    def _build_tab_result(self) -> None:
        ttk = self.ttk
        bar = ttk.Frame(self.tab_result, padding=6)
        bar.pack(fill="x")
        self.btn_export = ttk.Button(bar, text="Exporter tout (CSV + rapport + préprocesseur)", command=self._export)
        self.btn_export.pack(side="left")
        ttk.Button(bar, text="Charger un préprocesseur (.joblib)", command=self._load_preprocessor).pack(
            side="left", padx=8)
        ttk.Button(bar, text="Appliquer à un nouveau fichier", command=self._apply_new_file).pack(side="left")
        self.lbl_result = ttk.Label(self.tab_result, text="", padding=6)
        self.lbl_result.pack(fill="x")
        self.tree_result = self._make_tree(self.tab_result)

    # ------------------------------------------------------------------ utilitaires
    def _fill_tree(self, tree, df: pd.DataFrame) -> None:
        view = df.iloc[: self.PREVIEW_ROWS, : self.PREVIEW_COLS]
        cols = [str(c) for c in view.columns]
        tree["columns"] = cols
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=100, minwidth=60, stretch=False)
        for item in tree.get_children():
            tree.delete(item)
        for row in view.itertuples(index=False):
            tree.insert("", "end", values=[f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in row])

    def _set_status(self, text: str) -> None:
        self.v_status.set(text)

    def _read_options(self) -> Options:
        def num(var, cast, default):
            try:
                return cast(var.get())
            except (ValueError, self.tk.TclError):
                return default

        return Options(
            task=self.v_task.get(),
            test_size_pct=max(0, min(50, num(self.v_test, int, 20))),
            scaler_type=self.v_scaler.get(),
            outlier_method=self.v_outlier.get(),
            cardinality_threshold=num(self.v_card, int, 10),
            missing_threshold=num(self.v_missing, float, 0.90),
            correlation_threshold=num(self.v_corr, float, 0.95),
            rare_category_threshold=num(self.v_rare, float, 0.01),
            skew_threshold=num(self.v_skew, float, 1.0),
            transform_target=bool(self.v_transform_target.get()),
            drop_leaky_columns=bool(self.v_drop_leaky.get()),
            encode_discrete_as_categorical=bool(self.v_discrete_cat.get()),
        )

    # ------------------------------------------------------------------ actions
    def _open_file(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Choisir un dataset",
            filetypes=[("Tableaux", "*.csv *.tsv *.txt *.xlsx *.xls *.json *.parquet"), ("Tous les fichiers", "*.*")],
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            df = read_table(path)
        except Exception as exc:  # noqa: BLE001 - message affiché à l'utilisateur
            self.messagebox.showerror("Lecture impossible", str(exc))
            return
        self.df, self.path, self.result = df, path, None
        self.v_file.set(os.path.basename(path))
        self.cb_target["values"] = [NO_TARGET_LABEL] + [str(c) for c in df.columns]
        self.v_target.set(NO_TARGET_LABEL)
        self.v_task.set("auto")  # évite de garder « regression » d'un fichier précédent
        n_missing = int(df.isna().sum().sum())
        self.lbl_info.configure(
            text=f"{len(df)} lignes x {df.shape[1]} colonnes | {n_missing} valeurs manquantes | "
                 f"aperçu : {min(len(df), self.PREVIEW_ROWS)} premières lignes")
        self._fill_tree(self.tree_data, df)
        self._set_status("Fichier chargé. Choisissez la colonne cible dans « Paramètres ».")
        self.notebook.select(self.tab_params)

    def _start_run(self) -> None:
        if self.df is None:
            self.messagebox.showinfo("Aucun fichier", "Ouvrez d'abord un fichier.")
            return
        target = self.v_target.get()
        target = None if target == NO_TARGET_LABEL else target
        opts = self._read_options()
        self.btn_run.state(["disabled"])
        self.progress.start(12)
        self._set_status("Traitement en cours...")
        df = self.df

        def work() -> None:
            try:
                self._queue.put(("ok", run_preprocessing(df, target, opts)))
            except Exception as exc:  # noqa: BLE001 - remonté à l'interface
                self._queue.put(("err", f"{exc}\n\n{traceback.format_exc()}"))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(150, self._poll)

    def _poll(self) -> None:
        try:
            kind, payload = self._queue.get_nowait()
        except queue.Empty:
            self.root.after(150, self._poll)
            return
        self.progress.stop()
        self.btn_run.state(["!disabled"])
        if kind == "err":
            self._set_status("Erreur pendant le traitement.")
            self.txt_report.delete("1.0", "end")
            self.txt_report.insert("1.0", payload)
            self.notebook.select(self.tab_report)
            self.messagebox.showerror("Erreur", payload.split("\n\n")[0])
            return
        self._show_result(payload)

    def _show_result(self, result: RunResult) -> None:
        self.result, self.prep = result, result.prep
        self.txt_report.delete("1.0", "end")
        self.txt_report.insert("1.0", report_text(result))
        shown = _with_target(result.X_train, result.y_train, result.target, None)
        self._fill_tree(self.tree_result, shown)
        extra = f" | test : {result.X_test.shape[0]} lignes" if result.X_test is not None else ""
        self.lbl_result.configure(
            text=f"Train : {result.X_train.shape[0]} lignes x {result.X_train.shape[1]} features numériques{extra}"
                 f" (aperçu : {self.PREVIEW_ROWS} premières lignes)")
        leaks = result.prep.report().get("fuites_suspectees", [])
        warn = f" | ALERTE FUITE : {', '.join(leaks)}" if leaks else ""
        self._set_status(f"Terminé : {result.X_train.shape[1]} features numériques{warn}")
        self.notebook.select(self.tab_report)

    def _export(self) -> None:
        if self.result is None:
            self.messagebox.showinfo("Rien à exporter", "Lancez d'abord le prétraitement.")
            return
        folder = self.filedialog.askdirectory(title="Dossier de destination")
        if folder:
            self._do_export(folder)

    def _do_export(self, folder: str) -> None:
        try:
            written = export_result(self.result, folder)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Export impossible", str(exc))
            return
        self._set_status(f"Export terminé dans {folder}")
        self.messagebox.showinfo("Export terminé", "Fichiers écrits :\n" + "\n".join(os.path.basename(p) for p in written))

    def _load_preprocessor(self) -> None:
        path = self.filedialog.askopenfilename(title="Préprocesseur", filetypes=[("joblib", "*.joblib")])
        if path:
            self._do_load_preprocessor(path)

    def _do_load_preprocessor(self, path: str) -> None:
        try:
            self.prep = AutoPreprocessor.load(path)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Chargement impossible", str(exc))
            return
        self._set_status(f"Préprocesseur chargé : {os.path.basename(path)}")

    def _apply_new_file(self) -> None:
        if self.prep is None:
            self.messagebox.showinfo("Aucun préprocesseur", "Lancez un prétraitement ou chargez un fichier .joblib.")
            return
        path = self.filedialog.askopenfilename(title="Nouveau fichier à transformer")
        if not path:
            return
        out_path = self.filedialog.asksaveasfilename(title="Enregistrer le résultat", defaultextension=".csv",
                                                     filetypes=[("CSV", "*.csv")])
        if out_path:
            self._do_apply_new_file(path, out_path)

    def _do_apply_new_file(self, path: str, out_path: str) -> None:
        try:
            out = apply_to_new_data(self.prep, read_table(path))
            out.to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Transformation impossible", str(exc))
            return
        self._fill_tree(self.tree_result, out)
        self.lbl_result.configure(text=f"Nouveau fichier transformé : {out.shape[0]} lignes x {out.shape[1]} colonnes")
        self._set_status(f"Résultat enregistré : {out_path}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    try:
        app = App()
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Tkinter est introuvable ({exc}). Réinstallez Python depuis python.org avec l'option tcl/tk.")
    app.run()


if __name__ == "__main__":
    main()