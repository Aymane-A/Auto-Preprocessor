/** Coordinate the local session and its four workflow steps. */
import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api";
import type { Bin, Eda, Feature, Preview, Recommendation, Result, Upload } from "./api";
import { effectiveChoice } from "./components";
import { CleaningPanel, EdaPanel, ExecutionPanel, ImportPanel } from "./panels";
import type { Configuration } from "./panels";
import "./style.css";

const steps = [
  { label: "Import & configuration", title: "Préparer un jeu de données", description: "Importez un tableau et définissez le type d’apprentissage." },
  { label: "Exploration (EDA)", title: "Explorer les données", description: "Repérez les distributions, les valeurs manquantes et les relations entre variables." },
  { label: "Prétraitement", title: "Choisir les transformations", description: "Examinez les recommandations statistiques et décidez du traitement de chaque feature." },
  { label: "Exécution & export", title: "Exécuter et exporter", description: "Ajustez le prétraitement sur le train, puis téléchargez les jeux de données et le rapport." },
];
const initialConfig: Configuration = { mode: "supervised", target: "", task: "auto" };

/** Render the stepper and keep results scoped to the current dataset and configuration. */
function App() {
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

  return <>
    <header className="app-header">
      <div className="container brand">
        <span className="brand-icon" aria-hidden="true"><svg viewBox="0 0 24 24">
          <path d="M4 19V5m0 14h16M8 15v-4m4 4V7m4 8v-6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg></span>
        <div><strong>AutoPrépa</strong><span>Atelier de données</span></div>
        <span className="local"><i /> Session locale</span>
      </div>
    </header>
    <main className="container">
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
        <div className="eyebrow">ÉTAPE {step + 1} / 4</div>
        <h1 id="step-heading">{steps[step].title}</h1>
        <p>{steps[step].description}</p>
        {step > 0 && dataset && <span className="dataset-tag">{dataset.filename} · {config.mode === "supervised" ? config.target : "Sans cible"}</span>}
      </section>
      {busy && <div role="status" className="notice"><span className="spinner" />{busy}</div>}
      {error && <div role="alert" className="notice error">{error}</div>}
      {step === 0 && <ImportPanel dataset={dataset} config={config} busy={!!busy} onFile={file => {
        void perform("Import du fichier…", async () => {
          const form = new FormData(); form.append("file", file);
          const next = await api<Upload>("/upload", form);
          setDataset(next); setConfigured(false); setConfig(initialConfig); resetResults();
        });
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
          Voir les recommandations →</button></div>
      </> : !busy && <button onClick={() => dataset && void openStep(1, dataset)}>Relancer l’exploration</button>)}
      {step === 2 && (recommendations ? <>
        <CleaningPanel rows={recommendations} choices={choices} selected={selected} preview={preview} busy={!!busy}
          onPreview={compare} onChoice={(name, choice) => {
            setChoices(previous => ({ ...previous, [name]: choice })); setResult(null); compare(name, choice);
          }} onReset={() => { setChoices({}); setResult(null); if (selected) compare(selected, "recommended"); }} />
        <div className="advance"><button className="primary" disabled={!!busy} onClick={() => dataset && void openStep(3, dataset)}>
          Continuer vers l’exécution →</button></div>
      </> : !busy && <button onClick={() => dataset && void openStep(2, dataset)}>Relancer l’analyse</button>)}
      {step === 3 && dataset && <ExecutionPanel sessionId={dataset.session_id} busy={!!busy} result={result} onRun={testSize => {
        void perform("Prétraitement et création des exports…", async () => {
          setResult(await api("/preprocess", { session_id: dataset.session_id, choices, options: { test_size_pct: testSize } }));
        });
      }} />}
    </main>
    <footer className="container footer">AutoPrépa <span>·</span> Exploration et prétraitement de données</footer>
  </>;
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
