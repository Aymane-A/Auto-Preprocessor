/** Coordinate the local session and its four workflow steps. */
import { lazy, StrictMode, Suspense, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, effectiveChoice } from "./api";
import type { Bin, Eda, Feature, Preview, Recommendation, Result, Upload } from "./api";
import type { Configuration } from "./panels";
import { Landing } from "./Landing";
import demoUrl from "../../data/fraud_signal_demo.csv?url";
import "./style.css";

const ImportPanel = lazy(() => import("./panels").then(module => ({ default: module.ImportPanel })));
const EdaPanel = lazy(() => import("./panels").then(module => ({ default: module.EdaPanel })));
const CleaningPanel = lazy(() => import("./panels").then(module => ({ default: module.CleaningPanel })));
const ExecutionPanel = lazy(() => import("./panels").then(module => ({ default: module.ExecutionPanel })));

const steps = [
  { label: "Import & configuration", title: "Préparer un jeu de données", description: "Importez un tableau et définissez le type d’apprentissage." },
  { label: "Exploration (EDA)", title: "Explorer les données", description: "Repérez les distributions, les valeurs manquantes et les relations entre variables." },
  { label: "Prétraitement", title: "Choisir les transformations", description: "Examinez les recommandations statistiques et décidez du traitement de chaque feature." },
  { label: "Exécution & export", title: "Exécuter et exporter", description: "Ajustez le prétraitement sur le train, puis téléchargez les jeux de données et le rapport." },
];
const initialConfig: Configuration = { mode: "supervised", target: "", task: "auto" };

/** Render the stepper and keep results scoped to the current dataset and configuration. */
function App() {
  const [inWorkshop, setInWorkshop] = useState(window.location.hash === "#atelier");
  const [step, setStep] = useState(0);
  const [dataset, setDataset] = useState<Upload | null>(null);
  const [config, setConfig] = useState(initialConfig);
  const [configured, setConfigured] = useState(false);
  const [eda, setEda] = useState<Eda | null>(null);
  const [feature, setFeature] = useState<(Feature & { distribution: Bin[] }) | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[] | null>(null);
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    function navigate() {
      setInWorkshop(window.location.hash === "#atelier");
      if (window.location.hash !== "#parcours") {
        window.scrollTo(0, 0);
        requestAnimationFrame(() => document.getElementById("main-content")?.focus({ preventScroll: true }));
      }
    }
    window.addEventListener("hashchange", navigate);
    return () => window.removeEventListener("hashchange", navigate);
  }, []);

  /** Clear results that belong to the previous configuration. */
  function resetResults() {
    setEda(null); setFeature(null); setRecommendations(null); setChoices({});
    setSelected(null); setPreview(null); setResult(null);
  }

  /** Report progress and recover cleanly from network and processing failures. */
  async function perform(label: string, action: () => Promise<void>) {
    setBusy(label); setError("");
    try {
      await action();
    } catch (reason) {
      setError(reason instanceof TypeError
        ? "Le serveur local est inaccessible. Vérifiez que python serve.py est en cours d’exécution."
        : reason instanceof Error ? reason.message : "Une erreur est survenue.");
    } finally {
      setBusy("");
    }
  }

  /** Load data for a step once per configuration. */
  async function openStep(index: number, upload: Upload) {
    setStep(index);
    window.scrollTo(0, 0);
    if (index === 1 && !eda) {
      await perform("Calcul des statistiques…", async () => {
        const data = await api<Eda>(`/eda/${upload.session_id}`);
        setEda(data);
        const first = data.features.find(item => item.type === "numérique") ?? data.features[0];
        if (first) setFeature(await api(`/eda/${upload.session_id}/feature/${encodeURIComponent(first.column)}`));
      });
    }
    if (index === 2 && !recommendations) {
      await perform("Analyse des anomalies et des recommandations…", async () => {
        const data = await api<{ features: Recommendation[] }>("/analyze", { session_id: upload.session_id });
        setRecommendations(data.features);
      });
    }
  }

  /** Refresh the comparison for one proposed transformation. */
  function compare(column: string, choice = effectiveChoice(choices, column)) {
    if (!dataset) return;
    setSelected(column); setPreview(null);
    void perform("Calcul de l’aperçu…", async () => {
      setPreview(await api("/preview", { session_id: dataset.session_id, column, choice }));
    });
  }

  /** Send uploaded and demo files through the same local validation path. */
  async function uploadFile(file: File, nextConfig = initialConfig) {
    const form = new FormData();
    form.append("file", file);
    const next = await api<Upload>("/upload", form);
    setDataset(next);
    setConfigured(false);
    setConfig(nextConfig);
    setStep(0);
    resetResults();
  }

  /** Load the repository's fraud sample and preselect its known label. */
  function openDemo() {
    if (busy) return;
    window.location.hash = "atelier";
    setInWorkshop(true);
    void perform("Chargement du jeu de démonstration…", async () => {
      const response = await fetch(demoUrl);
      if (!response.ok) throw new Error("Le fichier de démonstration est indisponible.");
      const file = new File([await response.blob()], "fraud_signal_demo.csv", { type: "text/csv" });
      await uploadFile(file, { mode: "supervised", target: "is_fraud", task: "classification" });
    });
  }

  return <>
    <a className="skip-link" href="#main-content" onClick={event => {
      event.preventDefault();
      const content = document.getElementById("main-content");
      content?.focus();
      content?.scrollIntoView();
    }}>Aller au contenu</a>
    <header id="accueil" className="app-header">
      <div className="container brand">
        <a className="wordmark" href="#accueil" aria-label="AutoPrépa, accueil">
          <span className="brand-icon" aria-hidden="true"><svg viewBox="0 0 24 24">
            <path d="M3 18h18M6 14V9m4 5V5m4 9V8m4 6v-3" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg></span>
          <strong>AutoPrépa</strong>
        </a>
        <nav aria-label="Navigation principale">
          <a href={inWorkshop ? "#accueil" : "#parcours"}>{inWorkshop ? "Accueil" : "Le parcours"}</a>
          <a id="atelier" className="button" href="#atelier" aria-current={inWorkshop ? "page" : undefined}>Atelier</a>
        </nav>
      </div>
    </header>
    {!inWorkshop ? <Landing onDemo={openDemo} /> : <main id="main-content" className="container workspace" tabIndex={-1}>
      <nav className="stepper" aria-label="Étapes de préparation">
        {steps.map((item, index) => <button key={item.label} aria-current={index === step ? "step" : undefined}
          disabled={!!busy || (index > 0 && !configured)} onClick={() => {
            setError("");
            if (dataset) void openStep(index, dataset);
          }}>
          <span className="step-number">{String(index + 1).padStart(2, "0")}</span>{item.label}
        </button>)}
      </nav>
      <section className="intro" aria-labelledby="step-heading">
        <h1 id="step-heading">{steps[step].title}</h1>
        <p>{steps[step].description}</p>
        {step > 0 && dataset && <span className="dataset-tag">{dataset.filename} / {config.mode === "supervised" ? config.target : "Sans cible"}</span>}
      </section>
      {busy && <div role="status" className="notice"><span className="spinner" />{busy}</div>}
      {error && <div role="alert" className="notice error">{error}</div>}
      <Suspense fallback={<p className="notice" role="status">Ouverture de l’atelier…</p>}>
        {step === 0 && <ImportPanel dataset={dataset} config={config} busy={!!busy} onFile={file => {
          void perform("Import du fichier…", () => uploadFile(file));
        }} onChange={next => { setConfig(next); setConfigured(false); resetResults(); }} onConfigure={() => {
          if (!dataset) return;
          void (async () => {
            let valid = false;
            await perform("Validation de la configuration…", async () => {
              await api("/config", { session_id: dataset.session_id, mode: config.mode,
                target: config.mode === "supervised" ? config.target : null,
                task: config.mode === "supervised" ? config.task : "auto" });
              valid = true; setConfigured(true);
            });
            if (valid) await openStep(1, dataset);
          })();
        }} />}
        {step === 1 && (eda ? <>
          <EdaPanel eda={eda} feature={feature} busy={!!busy} onFeature={name => {
            if (dataset) void perform("Calcul de la distribution…", async () => {
              setFeature(await api(`/eda/${dataset.session_id}/feature/${encodeURIComponent(name)}`));
            });
          }} />
          <div className="advance"><button className="primary" disabled={!!busy} onClick={() => dataset && void openStep(2, dataset)}>
            Voir les recommandations</button></div>
        </> : !busy && <button onClick={() => dataset && void openStep(1, dataset)}>Relancer l’exploration</button>)}
        {step === 2 && (recommendations ? <>
          <CleaningPanel rows={recommendations} choices={choices} selected={selected} preview={preview} busy={!!busy}
            onPreview={compare} onChoice={(name, choice) => {
              setChoices(previous => ({ ...previous, [name]: choice })); setResult(null); compare(name, choice);
            }} onReset={() => { setChoices({}); setResult(null); if (selected) compare(selected, "recommended"); }} />
          <div className="advance"><button className="primary" disabled={!!busy} onClick={() => dataset && void openStep(3, dataset)}>
            Continuer vers l’exécution</button></div>
        </> : !busy && <button onClick={() => dataset && void openStep(2, dataset)}>Relancer l’analyse</button>)}
        {step === 3 && dataset && <ExecutionPanel sessionId={dataset.session_id} busy={!!busy} result={result} onRun={testSize => {
          void perform("Prétraitement et création des exports…", async () => {
            setResult(await api("/preprocess", { session_id: dataset.session_id, choices, options: { test_size_pct: testSize } }));
          });
        }} />}
      </Suspense>
    </main>}
    <footer className="container footer"><strong>AutoPrépa</strong><span>Atelier local de préparation de données</span></footer>
  </>;
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
