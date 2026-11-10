/** Shared lifecycle for backend requests, selections and the six workflow stages. */
import { useRef, useState } from "react";
import { api, effectiveChoice, uploadTable } from "./api";
import type { Bin, Eda, Feature, Preview, Recommendation, Result, TransformationReport, Upload } from "./api";
import type { Configuration } from "./panels";
import type { SceneNode } from "./ScientificScene";
import { stages } from "./workflow";
import demoUrl from "../../data/fraud_signal_demo.csv?url";

const initialConfig: Configuration = { mode: "supervised", target: "", task: "auto" };

export function useWorkflow() {
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
  const [report, setReport] = useState<TransformationReport | null>(null);
  const [busy, setBusy] = useState("");
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");
  const [failedStep, setFailedStep] = useState<number | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [testSize, setTestSize] = useState(20);
  const [pipelineReviewed, setPipelineReviewed] = useState(false);
  const lock = useRef(false);
  const context = useRef(0);

  function resetResults() {
    context.current++;
    setEda(null); setFeature(null); setRecommendations(null); setChoices({});
    setSelected(null); setPreview(null); setResult(null); setReport(null); setPipelineReviewed(false);
  }

  async function perform(label: string, action: () => Promise<void>, stage = step) {
    if (lock.current) return false;
    lock.current = true;
    setBusy(label); setError(""); setFailedStep(null);
    try {
      await action();
      return true;
    } catch (reason) {
      setError(reason instanceof TypeError
        ? "Le serveur local est inaccessible. Lancez python serve.py puis réessayez."
        : reason instanceof Error ? reason.message : "Une erreur est survenue. Réessayez l’opération.");
      setFailedStep(stage);
      return false;
    } finally {
      lock.current = false;
      setBusy("");
    }
  }

  async function loadAnalysis() {
    if (!dataset) return;
    if (!recommendations) {
      const data = await api<{ features: Recommendation[] }>("/analyze", { session_id: dataset.session_id });
      setRecommendations(data.features);
    }
  }

  async function navigate(index: number) {
    if (lock.current || index < 0 || index > 5) return;
    if (index > 0 && !configured) return;
    if (index === 5 && !result) return;
    setStep(index);
    if (index === 1 && !eda && dataset) {
      await perform("Calcul des statistiques et des distributions…", async () => {
        const data = await api<Eda>(`/eda/${dataset.session_id}`);
        setEda(data);
        const first = data.features.find(item => item.type === "numérique") ?? data.features[0];
        if (first) setFeature(await api(`/eda/${dataset.session_id}/feature/${encodeURIComponent(first.column)}`));
      }, index);
    }
    if (index >= 2 && index <= 4 && !recommendations) {
      await perform("Analyse des recommandations statistiques…", loadAnalysis, index);
    }
  }

  async function importFile(file: File, nextConfig = initialConfig) {
    const suffix = file.name.split(".").pop()?.toLowerCase();
    if (!suffix || !["csv","tsv","txt","xlsx","xls","json","parquet"].includes(suffix)) {
      setError("Format non pris en charge. Choisissez un fichier CSV, Excel, JSON ou Parquet.");
      setFailedStep(0);
      return;
    }
    await perform("Import et validation du fichier…", async () => {
      setUploadProgress(0);
      try {
        const uploaded = await uploadTable(file, setUploadProgress);
        setDataset(uploaded); setConfigured(false); setConfig(nextConfig); setStep(0);
        resetResults();
      } finally {
        setUploadProgress(null);
      }
    }, 0);
  }

  async function openDemo() {
    if (lock.current) return;
    setStep(0);
    try {
      const response = await fetch(demoUrl);
      if (!response.ok) throw new Error("Le fichier de démonstration est indisponible.");
      await importFile(new File([await response.blob()], "fraud_signal_demo.csv", { type: "text/csv" }),
        { mode: "supervised", target: "is_fraud", task: "classification" });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Chargement impossible."); setFailedStep(0);
    }
  }

  function changeConfig(next: Configuration) {
    if (lock.current) return;
    setConfig(next); setConfigured(false); resetResults();
  }

  async function configure() {
    if (!dataset) return;
    const valid = await perform("Validation de la configuration…", async () => {
      await api("/config", { session_id: dataset.session_id, mode: config.mode,
        target: config.mode === "supervised" ? config.target : null,
        task: config.mode === "supervised" ? config.task : "auto" });
      setConfigured(true);
    }, 0);
    if (valid) {
      // The next call loads EDA directly, after the server accepted the configuration.
      setStep(1);
      await perform("Calcul des statistiques et des distributions…", async () => {
        const data = await api<Eda>(`/eda/${dataset.session_id}`);
        setEda(data);
        const first = data.features.find(item => item.type === "numérique") ?? data.features[0];
        if (first) setFeature(await api(`/eda/${dataset.session_id}/feature/${encodeURIComponent(first.column)}`));
      }, 1);
    }
  }

  function compare(column: string, choice = effectiveChoice(choices, column)) {
    if (!dataset || lock.current) return;
    setSelected(column); setPreview(null);
    void perform("Calcul de la comparaison…", async () => {
      setPreview(await api("/preview", { session_id: dataset.session_id, column, choice }));
    });
  }

  function choose(column: string, choice: string) {
    if (lock.current) return;
    setChoices(previous => ({ ...previous, [column]: choice }));
    setResult(null); setReport(null); setPipelineReviewed(false); setPreview(null);
    if (step === 4) compare(column, choice);
  }

  function acceptAll() {
    if (lock.current) return;
    setChoices({}); setResult(null); setReport(null); setPipelineReviewed(false); setPreview(null);
  }

  function changeTestSize(value: number) {
    if (lock.current) return;
    setTestSize(value);
    setResult(null);
    setReport(null);
  }

  function selectFeature(column: string) {
    if (!dataset) return;
    void perform("Calcul de la distribution…", async () => {
      setFeature(await api(`/eda/${dataset.session_id}/feature/${encodeURIComponent(column)}`));
    });
  }

  async function run(testSize: number) {
    if (!dataset || lock.current) return;
    setResult(null); setReport(null); setProcessing(true);
    try {
      await perform("Traitement dans le worker Python…", async () => {
        const generation = context.current;
        const next = await api<Result>("/preprocess", {
          session_id: dataset.session_id, choices, options: { test_size_pct: testSize },
        });
        if (generation !== context.current) return;
        setResult(next);
      }, 4);
    } finally {
      setProcessing(false);
    }
  }

  async function openExports() {
    if (!dataset || !result) return;
    setStep(5);
    if (!report) await perform("Chargement du rapport d’exécution…", async () => {
      setReport(await api(`/report/${dataset.session_id}`));
    }, 5);
  }

  const completed = [configured, !!eda, !!recommendations, pipelineReviewed, !!result, !!report];
  const nodes: SceneNode[] = stages.map((stage, index) => ({
    label: stage.short,
    state: failedStep === index ? "error" : index === step ? "active"
      : completed[index] ? index === 5 && result?.warnings.length ? "warning" : "complete" : "pending",
    disabled: !!busy || (index > 0 && !configured) || (index === 5 && !result),
  }));

  return { step, dataset, config, configured, eda, feature, recommendations, choices, selected, preview, result, report,
    busy, processing, error, uploadProgress, testSize, pipelineReviewed, nodes,
    changeTestSize, setPipelineReviewed, navigate, importFile, openDemo, changeConfig, configure, compare, choose,
    acceptAll, selectFeature, run, openExports,
  };
}
