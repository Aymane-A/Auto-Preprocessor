/** Shared lifecycle for backend requests, selections and the six workflow stages. */
import { useRef, useState } from "react";
import { api, ApiError, downloadFile, featureEdaPath, resolvedChoice, uploadTable } from "./api";
import type { Choice, Choices, Options, EvaluationResult, Eda, Feature, Preview, Recommendation, Result, TransformationReport, Upload } from "./api";
import type { Configuration } from "./panels";
import type { SceneNode } from "./ScientificScene";
import { stages } from "./workflow";

const initialConfig: Configuration = { mode: "supervised", target: "", task: "auto" };

export function useWorkflow() {
  const [step, setStep] = useState(0);
  const [dataset, setDataset] = useState<Upload | null>(null);
  const [config, setConfig] = useState(initialConfig);
  const [configured, setConfigured] = useState(false);
  const [eda, setEda] = useState<Eda | null>(null);
  const [feature, setFeature] = useState<Feature | null>(null);
  const [featureError, setFeatureError] = useState("");
  const [analysisNote, setAnalysisNote] = useState("");
  const [recommendations, setRecommendations] = useState<Recommendation[] | null>(null);
  const [choices, setChoices] = useState<Choices>({});
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
  const [options, setOptions] = useState<Options>({});
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);
  const [afterEda, setAfterEda] = useState<Eda | null>(null);
  const [pipelineReviewed, setPipelineReviewed] = useState(false);
  const lock = useRef(false);
  const context = useRef(0);

  function resetResults() {
    context.current++;
    setEda(null); setFeature(null); setFeatureError(""); setRecommendations(null); setAnalysisNote(""); setChoices({});
    setSelected(null); setPreview(null); setResult(null); setReport(null); setPipelineReviewed(false); setEvaluation(null); setAfterEda(null);
  }

  async function loadFeature(session: string, column: string, selectedChoices: Choices): Promise<Feature | null> {
    setFeatureError("");
    try { return await api<Feature>(featureEdaPath(session, column, selectedChoices)); }
    catch (reason) {
      if (reason instanceof ApiError && reason.status === 404 && reason.message.startsWith("Session")) throw reason;
      setFeatureError(`Visualisation indisponible pour cette colonne : ${column}.`);
      return null;
    }
  }

  async function perform(label: string, action: () => Promise<void>, stage = step) {
    if (lock.current) return false;
    lock.current = true;
    setBusy(label); setError(""); setFailedStep(null);
    try {
      await action();
      return true;
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 404 && reason.message.startsWith("Session")) {
        resetResults(); setDataset(null); setConfig(initialConfig); setOptions({}); setTestSize(20); setConfigured(false); setStep(0);
      }
      setError(reason instanceof Error ? reason.message : "Une erreur est survenue. Réessayez l’opération.");
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
      const data = await api<{ features: Recommendation[]; sample_note?: string }>("/analyze", { session_id: dataset.session_id, options: { ...options, test_size_pct: testSize }, choices });
      setRecommendations(data.features); setAnalysisNote(data.sample_note ?? "");
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
        if (first) setFeature(await loadFeature(dataset.session_id, first.column, choices));
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
        if (dataset) {
          try { await api(`/session/${dataset.session_id}`, undefined, "DELETE"); }
          catch (reason) {
            if (!(reason instanceof ApiError && reason.status === 404)) {
              await api(`/session/${uploaded.session_id}`, undefined, "DELETE");
              throw reason;
            }
          }
        }
        setDataset(uploaded); setConfigured(false); setConfig(nextConfig); setStep(0); setOptions({}); setTestSize(20);
        resetResults();
      } finally {
        setUploadProgress(null);
      }
    }, 0);
  }

  function changeConfig(next: Configuration) {
    if (lock.current) return;
    resetResults(); setConfig(next); setConfigured(false);
  }

  async function configure() {
    if (!dataset) return;
    const valid = await perform("Validation de la configuration…", async () => {
      await api("/config", { session_id: dataset.session_id, mode: config.mode,
        target: config.mode === "supervised" ? config.target : null,
        task: config.mode === "supervised" ? config.task : "auto" });
      resetResults(); setConfigured(true);
    }, 0);
    if (valid) {
      // The next call loads EDA directly, after the server accepted the configuration.
      setStep(1);
      await perform("Calcul des statistiques et des distributions…", async () => {
        const data = await api<Eda>(`/eda/${dataset.session_id}`);
        setEda(data);
        const first = data.features.find(item => item.type === "numérique") ?? data.features[0];
        if (first) setFeature(await loadFeature(dataset.session_id, first.column, {}));
      }, 1);
    }
  }

  function compare(column: string, choice = resolvedChoice(choices, column)) {
    if (!dataset || lock.current) return;
    setSelected(column); setPreview(null);
    void perform("Calcul de la comparaison…", async () => {
      setPreview(await api("/preview", { session_id: dataset.session_id, column, choice, options: { ...options, test_size_pct: testSize } }));
    });
  }

  function choose(column: string, choice: Choice) {
    if (lock.current) return;
    const previous = resolvedChoice(choices, column);
    const previousForced = typeof previous === "string" ? previous.startsWith("type:FORCE_") : previous.type?.action.startsWith("FORCE_");
    const changesType = previousForced || (typeof choice === "string" ? choice.startsWith("type:FORCE_") : choice.type?.action.startsWith("FORCE_"));
    if (changesType && dataset) {
      void perform("Recalcul des opérations pour le type choisi…", async () => {
        const next = { ...choices, [column]: choice };
        const data = await api<{ features: Recommendation[]; sample_note?: string }>("/analyze", { session_id: dataset.session_id, options: { ...options, test_size_pct: testSize }, choices: next });
        const refreshedFeature = feature ? await loadFeature(dataset.session_id, feature.column, next) : null;
        setChoices(next); setRecommendations(data.features); setAnalysisNote(data.sample_note ?? ""); setPreview(null); setResult(null); setReport(null); setAfterEda(null); setEvaluation(null); setPipelineReviewed(false);
        setFeature(refreshedFeature);
      });
      return;
    }
    setChoices(previous => ({ ...previous, [column]: choice }));
    setResult(null); setReport(null); setPipelineReviewed(false); setPreview(null); setAfterEda(null); setEvaluation(null);
    if (step === 4) compare(column, choice);
  }

  function acceptAll() {
    if (lock.current) return;
    if (!dataset) return;
    void perform("Recalcul des recommandations initiales…", async () => {
      const data = await api<{ features: Recommendation[]; sample_note?: string }>("/analyze", { session_id: dataset.session_id, options: { ...options, test_size_pct: testSize } });
      const refreshedFeature = feature ? await loadFeature(dataset.session_id, feature.column, {}) : null;
      setChoices({}); setRecommendations(data.features); setAnalysisNote(data.sample_note ?? "");
      setFeature(refreshedFeature);
      setResult(null); setReport(null); setPipelineReviewed(false); setPreview(null); setAfterEda(null); setEvaluation(null);
    });
  }

  function changeTestSize(value: number) {
    if (lock.current) return;
    setTestSize(value);
    setPreview(null); setSelected(null); setEvaluation(null); setPipelineReviewed(false);
    setResult(null);
    setReport(null); setAfterEda(null);
  }
  function changeImbalance(next: Options) {
    if (lock.current) return;
    setOptions(current => ({ ...current, ...next }));
    setResult(null); setReport(null); setAfterEda(null);
  }

  function selectFeature(column: string) {
    if (!dataset) return;
    setFeature(null); setFeatureError("");
    void perform("Calcul de la distribution…", async () => {
      setFeature(await loadFeature(dataset.session_id, column, choices));
    });
  }

  async function run(testSize: number) {
    if (!dataset || lock.current) return;
    setResult(null); setReport(null); setAfterEda(null); setProcessing(true);
    try {
      await perform("Traitement dans le worker Python…", async () => {
        const generation = context.current;
        const next = await api<Result>("/preprocess", {
          session_id: dataset.session_id, choices, options: { ...options, test_size_pct: testSize },
        });
        if (generation !== context.current) return;
        setResult(next);
      }, 4);
    } finally {
      setProcessing(false);
    }
  }

  async function openExports() {
    if (!dataset || !result || !configured) return;
    setStep(5);
    if (!report) await perform("Chargement du rapport d’exécution…", async () => {
      setReport(await api(`/report/${dataset.session_id}`));
    }, 5);
  }

  async function applyOptions(next: Options) {
    if (!dataset) return;
    await perform("Recalcul des recommandations avec les réglages…", async () => {
      const data = await api<{ features: Recommendation[]; sample_note?: string }>("/analyze", { session_id: dataset.session_id, options: { ...next, test_size_pct: next.test_size_pct ?? testSize }, choices });
      setOptions(next); setRecommendations(data.features); setAnalysisNote(data.sample_note ?? ""); setPreview(null); setResult(null); setReport(null);
      setAfterEda(null); setEvaluation(null); setPipelineReviewed(false); setSelected(null);
      setTestSize(typeof next.test_size_pct === "number" ? next.test_size_pct : testSize);
    });
  }

  async function evaluate(family: string, folds: number, timeColumn: string) {
    if (!dataset) return;
    await perform("Validation croisée dans le worker Python…", async () => {
      setEvaluation(null);
      setEvaluation(await api("/evaluate", { session_id: dataset.session_id, options: { ...options, test_size_pct: testSize }, family, folds,
        time_column: timeColumn || null }));
    });
  }

  async function importConfiguration(file: File) {
    if (!dataset) return;
    await perform("Validation de la configuration importée…", async () => {
      if (file.size > 1024 * 1024) throw new Error("La configuration dépasse 1 Mio.");
      const imported = JSON.parse(await file.text());
      if (!imported || typeof imported !== "object" || Array.isArray(imported)
        || (imported.options !== undefined && (!imported.options || typeof imported.options !== "object" || Array.isArray(imported.options)))
        || (imported.choices !== undefined && (!imported.choices || typeof imported.choices !== "object" || Array.isArray(imported.choices)))) {
        throw new Error("La configuration doit contenir un objet JSON avec des options et des choix valides.");
      }
      if (imported.schema_version !== 1 || imported.mode !== config.mode || imported.target !== (config.mode === "supervised" ? config.target : null)
        || imported.task !== (config.mode === "supervised" ? config.task : "auto")) throw new Error("La version, le mode, la cible et la tâche doivent correspondre à la session courante.");
      const data = await api<{ features: Recommendation[]; sample_note?: string }>("/analyze", { session_id: dataset.session_id, options: imported.options, choices: imported.choices });
      const refreshedFeature = feature ? await loadFeature(dataset.session_id, feature.column, imported.choices ?? {}) : null;
      setOptions(imported.options ?? {}); setChoices(imported.choices ?? {}); setRecommendations(data.features); setAnalysisNote(data.sample_note ?? "");
      setFeature(refreshedFeature);
      setTestSize(imported.options?.test_size_pct ?? 20); setPreview(null); setSelected(null);
      setResult(null); setReport(null); setAfterEda(null); setEvaluation(null); setPipelineReviewed(false);
    });
  }

  async function inspectResults() {
    if (!dataset || !result || !configured) return;
    await perform("Analyse du tableau transformé…", async () => {
      setAfterEda(await api(`/results-eda/${dataset.session_id}`));
    });
  }

  async function prepareExcel() {
    if (!dataset || !result || !configured) return;
    await perform("Préparation du dataset Excel…", async () => {
      await api("/export-excel", { session_id: dataset.session_id });
      setResult(previous => previous ? { ...previous, downloads: [...new Set([...previous.downloads, "full_xlsx"])] } : previous);
    });
  }

  async function closeSession() {
    if (!dataset) return;
    await perform("Fermeture de la session…", async () => {
      await api(`/session/${dataset.session_id}`, undefined, "DELETE");
      resetResults(); setDataset(null); setConfigured(false); setConfig(initialConfig); setOptions({}); setTestSize(20); setStep(0);
    });
  }

  async function download(kind: string) {
    if (!dataset || !result || !configured || !result.downloads.includes(kind)) return;
    await perform("Téléchargement du fichier…", () => downloadFile(dataset.session_id, kind), 5);
  }

  const completed = [configured, !!eda, !!recommendations, pipelineReviewed, !!result, !!report];
  const nodes: SceneNode[] = stages.map((stage, index) => ({
    label: stage.short,
    state: failedStep === index ? "error" : index === step ? "active"
      : completed[index] ? index === 5 && result?.warnings.length ? "warning" : "complete" : "pending",
    disabled: !!busy || (index > 0 && !configured) || (index === 5 && !result),
  }));

  return { step, dataset, config, configured, eda, feature, featureError, recommendations, choices, selected, preview, result, report,
    busy, processing, error, uploadProgress, testSize, pipelineReviewed, nodes, options, evaluation, afterEda, analysisNote,
    applyOptions, evaluate, inspectResults, prepareExcel, importConfiguration, closeSession, download,
    changeTestSize, changeImbalance, setPipelineReviewed, navigate, importFile, changeConfig, configure, compare, choose,
    acceptAll, selectFeature, run, openExports,
  };
}
