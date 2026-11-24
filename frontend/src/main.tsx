/** Six-stage local scientific workspace, backed by the existing Python API. */
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { AnimatePresence, motion } from "framer-motion";
import { AnimatedNumber, AnimationSystem, Notice, WorkflowStage, motionTiming, useReducedMotionPreference } from "./animation";
import { ScientificScene } from "./ScientificScene";
import type { Atmosphere } from "./OrbitalScene";
import { stages } from "./workflow";
import { useWorkflow } from "./useWorkflow";
import { api } from "./api";
import type { SelectionDetails, SelectionPreview } from "./api";
import "./style.css";

const ImportPanel = lazy(() => import("./panels").then(module => ({ default: module.ImportPanel })));
const EdaPanel = lazy(() => import("./panels").then(module => ({ default: module.EdaPanel })));
const CleaningPanel = lazy(() => import("./panels").then(module => ({ default: module.CleaningPanel })));
const ExecutionPanel = lazy(() => import("./panels").then(module => ({ default: module.ExecutionPanel })));
const PipelinePanel = lazy(() => import("./PipelinePanel").then(module => ({ default: module.PipelinePanel })));
const AppliedReport = lazy(() => import("./PipelinePanel").then(module => ({ default: module.AppliedReport })));
const AdvancedSettings = lazy(() => import("./AdvancedSettings").then(module => ({ default: module.AdvancedSettings })));
const EvaluationPanel = lazy(() => import("./AdvancedSettings").then(module => ({ default: module.EvaluationPanel })));
const NestedEvaluationPanel = lazy(() => import("./NestedEvaluation").then(module => ({ default: module.NestedEvaluationPanel })));
const OrbitalScene = lazy(() => import("./OrbitalScene").then(module => ({ default: module.OrbitalScene })));

function Mark() {
  return <svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
    <path d="m16 3 12 7v13L16 30 4 23V10L16 3Z" stroke="currentColor" strokeWidth="1.4" />
    <path d="m4 10 12 7 12-7M16 17v13M10 6.5l12 7v13" stroke="currentColor" strokeWidth="1.1" />
  </svg>;
}

function App() {
  const state = useWorkflow();
  const [selectionPreview, setSelectionPreview] = useState<SelectionPreview | null>(null);
  const [selectionError, setSelectionError] = useState("");
  const [selectionLoading, setSelectionLoading] = useState(false);
  const [selectionConfirmed, setSelectionConfirmed] = useState(false);
  const reducedMotion = useReducedMotionPreference();
  const { step, dataset, busy, result } = state;
  const current = stages[step];
  const appliedOptions = useMemo(() => ({ ...state.options, test_size_pct: state.testSize }), [state.options, state.testSize]);
  const selectionIdentity = JSON.stringify([dataset?.session_id, state.config, state.options, state.choices, state.testSize]);
  const selectionIdentityRef = useRef(selectionIdentity);
  selectionIdentityRef.current = selectionIdentity;
  useEffect(() => { setSelectionPreview(null); setSelectionConfirmed(false); setSelectionError(""); }, [selectionIdentity]);

  async function previewSelection() {
    if (!dataset || selectionLoading) return;
    setSelectionLoading(true); setSelectionError(""); setSelectionConfirmed(false);
    const requestedIdentity = selectionIdentity;
    try { const response = await api<SelectionPreview>("/selection-preview", { session_id: dataset.session_id, options: appliedOptions, choices: state.choices });
      if (requestedIdentity === selectionIdentityRef.current) setSelectionPreview(response); }
    catch (reason) { if (requestedIdentity === selectionIdentityRef.current) { setSelectionError(reason instanceof Error ? reason.message : "Aperçu impossible."); setSelectionPreview(null); } }
    finally { setSelectionLoading(false); }
  }

  useEffect(() => {
    document.title = current.name + " | AutoPrépa";
    window.scrollTo(0, 0);
    document.getElementById("stage-title")?.focus({ preventScroll: true });
  }, [current.name]);

  function go(index: number) {
    if (index === 5) void state.openExports();
    else void state.navigate(index);
  }

  function exportConfig() {
    const data = JSON.stringify({
      schema_version: 1, mode: state.config.mode,
      target: state.config.mode === "supervised" ? state.config.target : null,
      task: state.config.mode === "supervised" ? state.config.task : "auto",
      options: { ...state.options, test_size_pct: state.testSize }, choices: state.choices,
    }, null, 2);
    const url = URL.createObjectURL(new Blob([data], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url; link.download = "configuration-pipeline.json"; link.click();
    URL.revokeObjectURL(url);
  }

  return <div className="application">
    <a className="skip-link" href="#main-content">Aller au contenu</a>
    <aside className="workflow-sidebar">
      <a className="wordmark" href="#main-content" onClick={() => go(0)}><Mark /><span>AutoPrépa<small>Laboratoire de données</small></span></a>
      <p className="sidebar-heading">Espace de préparation</p>
      <nav aria-label="Étapes de préparation" className="workflow-nav">
        {stages.map((stage, index) => <button type="button" key={stage.name}
          data-state={state.nodes[index].state} aria-current={step === index ? "step" : undefined}
          disabled={state.nodes[index].disabled} onClick={() => go(index)}>
          {step === index && <motion.span className="nav-selection" layoutId={reducedMotion ? undefined : "workflow-selection"}
            transition={{ type: "spring", bounce: 0, duration: reducedMotion ? 0 : .3 }} aria-hidden="true" />}
          <span className="stage-number">{state.nodes[index].state === "complete" ? "✓" : String(index + 1).padStart(2, "0")}</span>
          <span>{stage.name}<small>{state.nodes[index].state === "error" ? "Erreur à résoudre"
            : index === step ? "Étape active" : state.nodes[index].state === "complete" ? "Terminé"
            : state.nodes[index].state === "warning" ? "Avec avertissements" : "À venir"}</small></span>
        </button>)}
      </nav>
      <div className="sidebar-note">
        <p>Exploration statistique et prétraitement reproductible.</p><span>React / Python / WebGL</span>
      </div>
    </aside>

    <div className="workspace-shell">
      <header className="workspace-toolbar">
        <span>Atelier scientifique <span className="toolbar-divider">/</span> {current.name}</span>
        <span className="environment-tag">Exécution locale</span>
      </header>
      <main id="main-content" className="workspace" tabIndex={-1}>
        <div className="stage-heading">
          <div><p className="stage-caption">Étape {String(step + 1).padStart(2, "0")} sur 06</p>
            <h1 id="stage-title" tabIndex={-1}>{current.title}</h1><p>{current.text}</p></div>
          {dataset && <div className="dataset-badge"><strong>{dataset.filename}</strong>
            <span><AnimatedNumber value={dataset.shape[0]} digits={0} /> lignes / <AnimatedNumber value={dataset.shape[1]} digits={0} /> colonnes</span></div>}
        </div>

        <AnimatePresence initial={false}>
          {state.error && <Notice key="error" message={state.error} error />}
          {busy && <Notice key="busy" message={busy}>
            {state.uploadProgress !== null && <><div className="upload-progress" role="progressbar" aria-label="Envoi du fichier"
              aria-valuemin={0} aria-valuemax={100} aria-valuenow={state.uploadProgress}>
              <motion.div initial={false} animate={{ scaleX: state.uploadProgress / 100 }} transition={{ duration: motionTiming.interaction }} />
            </div>
              <span>{state.uploadProgress < 100 ? `Envoi du fichier, ${state.uploadProgress} %` : "Envoi terminé, lecture et validation côté serveur."}</span></>}
          </Notice>}
        </AnimatePresence>

        <AnimatePresence mode="popLayout">
        <WorkflowStage key={step}>
          <Suspense fallback={<p className="notice" role="status">Ouverture du panneau…</p>}>
            {step === 0 && <>
              <section className="import-introduction">
                <div className="intro-copy">
                  <h2>Du brut.<br />Au signal.</h2>
                  <p>Un environnement pour comprendre vos données et décider de chaque transformation avant l’entraînement.</p>
                  <div className="import-principles"><span>Ajustement sur train</span><span>Choix par variable</span><span>Rapport Excel</span></div>
                </div>
                <div className="hero-science" data-loaded={!!dataset}>
                  <ScientificScene label="Réseau scientifique tridimensionnel, illustration du moteur" running={!!busy} />
                  <div className="scene-heading"><span>Illustration du moteur</span></div>
                </div>
              </section>
              <ImportPanel dataset={dataset} config={state.config} busy={!!busy} onFile={file => void state.importFile(file)}
                onChange={state.changeConfig} onConfigure={() => void state.configure()} />
              {dataset && <button disabled={!!busy} onClick={() => void state.closeSession()}>Fermer la session et ses fichiers temporaires</button>}
            </>}

            {step === 1 && (state.eda ? <>
              <EdaPanel eda={state.eda} feature={state.feature} featureError={state.featureError} busy={!!busy} onFeature={state.selectFeature} />
              <div className="advance"><button className="primary" disabled={!!busy} onClick={() => go(2)}>Voir les recommandations</button></div>
            </> : !busy && <button onClick={() => go(1)}>Relancer l’analyse statistique</button>)}

            {step === 2 && (state.recommendations ? <>
              {state.eda?.imbalance?.level && <p className="notice">Cible déséquilibrée ({state.eda.imbalance.level}) : {(state.eda.imbalance.minority_ratio!*100).toFixed(2)} % dans la classe minoritaire. L’accuracy peut être trompeuse ; comparer PR-AUC, rappel et F1.</p>}
              {state.analysisNote && <p className="notice">{state.analysisNote}</p>}
              <CleaningPanel rows={state.recommendations} choices={state.choices} selected={state.selected}
                preview={state.preview} busy={!!busy} onChoice={state.choose} onReset={state.acceptAll}
                onPreview={column => { state.compare(column); }} view="recommendations" />
              <div className="advance"><button className="primary" disabled={!!busy} onClick={() => go(3)}>Configurer le pipeline</button></div>
            </> : !busy && <button onClick={() => go(2)}>Relancer les recommandations</button>)}

            {step === 3 && state.recommendations && <>
              <AdvancedSettings options={appliedOptions} busy={!!busy} onApply={next => void state.applyOptions(next)} onImport={file => void state.importConfiguration(file)} />
              <PipelinePanel rows={state.recommendations} choices={state.choices} busy={!!busy}
                onChoice={state.choose} workflow={state.nodes} onNavigate={go} />
              <div className="advance"><button className="primary" disabled={!!busy} onClick={() => {
                state.setPipelineReviewed(true); go(4);
              }}>Valider le pipeline et comparer</button></div>
            </>}

            {step === 4 && dataset && <>
              <section className="panel" aria-labelledby="selection-preview-title"><h2 id="selection-preview-title">Aperçu de la sélection</h2>
                <p className="muted">Calculé sur le train du découpage choisi. La source reste intacte ; l’exécution ajuste de nouveau les paramètres.</p>
                <button type="button" disabled={!!busy || selectionLoading} onClick={() => void previewSelection()}>{selectionLoading ? "Calcul de l’aperçu…" : "Calculer l’aperçu de sélection"}</button>
                {selectionError && <p role="alert" className="notice error">{selectionError}</p>}
                {selectionPreview && <><p>{selectionPreview.method} · {selectionPreview.original.length} colonnes source · {selectionPreview.encoded_count} features encodées · {selectionPreview.selected_count} retenues (dont colonnes brutes) · sortie estimée {selectionPreview.expected_rows} × {selectionPreview.output_count} colonnes{selectionPreview.target_included ? " (cible incluse)" : ""}</p>
                  <details><summary>Colonnes source</summary><p>{selectionPreview.original.join(" · ")}</p></details>
                  <p className="muted">{selectionPreview.note}</p>
                  <div className="two-col"><div><h3>Features attendues</h3><ul>{selectionPreview.selected.map(name => <li key={name}><strong>{name}</strong>
                    {selectionPreview.feature_sources?.[name] && <small>Source : {selectionPreview.feature_sources[name]}</small>}
                    {selectionPreview.details?.scores?.[name] != null && <small>Score : {selectionPreview.details.scores[name]}</small>}
                    {selectionPreview.details?.rankings?.[name] != null && <small>Rang : {selectionPreview.details.rankings[name]}</small>}
                    {selectionPreview.details?.reasons?.[name] && <small>{selectionPreview.details.reasons[name]}</small>}</li>)}</ul></div>
                    <div><h3>Rejetées par la méthode</h3><ul>{selectionPreview.rejected.map(name => <li key={name}><strong>{name}</strong>
                      {selectionPreview.feature_sources?.[name] && <small>Source : {selectionPreview.feature_sources[name]}</small>}
                      {selectionPreview.details?.reasons?.[name] && <small>{selectionPreview.details.reasons[name]}</small>}</li>)}</ul></div></div>
                  {selectionPreview.details && <p className="muted">Sélecteur ajusté sur {selectionPreview.details.fit_rows} lignes du train en {selectionPreview.details.elapsed_seconds} s{selectionPreview.details.scoring ? ` · métrique ${selectionPreview.details.scoring}` : ""}.</p>}
                  {selectionPreview.warnings.map((warning, index) => <p className="notice" key={index}>{warning}</p>)}
                  <label className="selection-confirm"><input type="checkbox" checked={selectionConfirmed} onChange={event => setSelectionConfirmed(event.target.checked)} /> Je confirme la sélection de variables et l’exécution sur une copie des données.</label>
                </>}
              </section>
              <EvaluationPanel columns={dataset.columns.filter(column => column !== state.config.target)} supervised={state.config.mode === "supervised"}
                busy={!!busy} result={state.evaluation} onEvaluate={(family, folds, time) => void state.evaluate(family, folds, time)} />
              {state.config.mode === "supervised" && <NestedEvaluationPanel
                key={`${dataset.session_id}|${state.config.target}|${state.config.task}|${JSON.stringify(state.options)}`}
                sessionId={dataset.session_id} columns={dataset.columns} target={state.config.target}
                task={state.config.task} options={state.options} busy={!!busy}
                onTaskChange={task => { state.changeConfig({ ...state.config, task }); go(0); }} />}
              <section className="processing-overview">
                <ScientificScene label="État du parcours" nodes={state.nodes} running={state.processing} onSelect={go} />
                <div className="processing-message">
                  <h2>{state.processing ? "Traitement en cours" : result ? "Traitement terminé" : "Prêt pour l’exécution"}</h2>
                  <p>{state.processing ? "Le worker Python exécute le pipeline sélectionné. Le résultat sera disponible à la fin de la requête."
                    : result ? `${result.train_shape[0].toLocaleString("fr-FR")} lignes de train et ${result.test_shape[0].toLocaleString("fr-FR")} lignes de test exportées.`
                    : "L’aperçu décrit une variable. Le traitement final ajuste ses paramètres uniquement sur le train."}</p>
                  <p className="muted">L’API ne diffuse ni pourcentage de traitement, ni journal ou étape interne en direct.</p>
                </div>
              </section>
              {state.recommendations && <CleaningPanel rows={state.recommendations} choices={state.choices} selected={state.selected}
                preview={state.preview} busy={!!busy} onChoice={state.choose} onReset={state.acceptAll}
                onPreview={state.compare} view="preview" />}
              <ExecutionPanel sessionId={dataset.session_id} busy={!!busy} runDisabled={!selectionConfirmed} result={result} onRun={size => void state.run(size)}
                imbalance={state.eda?.imbalance} options={state.options} onImbalance={state.changeImbalance}
                rawCategorical={dataset.columns.some(col => state.choices[col] === "none" && /object|string|category/.test(dataset.dtypes[col] ?? ""))}
                view="run" testSize={state.testSize} onTestSize={value => {
                  state.changeTestSize(value);
                }} />
              {result && <div className="advance"><button className="primary" disabled={!!busy} onClick={() => void state.openExports()}>Voir les exports</button></div>}
            </>}

            {step === 5 && dataset && result && <>
              <section className="completion-overview">
                <div><h2>Exécution disponible</h2><p>Les fichiers et le rapport correspondent aux choix de cette exécution.</p></div>
                <ScientificScene label="Réseau scientifique, confirmation de fin du traitement" celebrate />
              </section>
              <ExecutionPanel sessionId={dataset.session_id} busy={!!busy} result={result} onRun={size => void state.run(size)}
                onDownload={kind => void state.download(kind)} view="exports" testSize={state.testSize} onTestSize={state.changeTestSize} />
              {state.report ? <AppliedReport report={state.report} /> : !busy &&
                <button onClick={() => void state.openExports()}>Recharger le rapport des transformations</button>}
              {state.report?.core?.selection_features && <section className="panel" aria-labelledby="selection-title">
                <h2 id="selection-title">Sélection des variables</h2>
                {(() => { const selection = state.report!.core!.selection_features as SelectionDetails & { original_count: number; selected_count: number };
                  return <><p>{selection.method} · {selection.selected_count} sur {selection.original_count} features conservées · {selection.elapsed_seconds} s</p>
                     <div className="two-col"><div><h3>Conservées</h3><ul>{selection.selected.map(name => <li key={name}>{name}{selection.scores?.[name] == null ? "" : ` · score ${selection.scores[name]}`}{selection.rankings?.[name] == null ? "" : ` · rang ${selection.rankings[name]}`}{selection.reasons?.[name] ? ` · ${selection.reasons[name]}` : ""}</li>)}</ul></div>
                       <div><h3>Retirées</h3><ul>{selection.rejected.map(name => <li key={name}>{name}{selection.reasons?.[name] ? ` · ${selection.reasons[name]}` : ""}</li>)}</ul></div></div>
                    {selection.scores && <div className="selection-scores" role="img" aria-label={`Classement des scores de ${selection.method}`}>
                      <h3>Scores · {selection.method}</h3>
                      {Object.entries(selection.scores).filter((entry): entry is [string, number] => entry[1] !== null && Number.isFinite(entry[1]))
                        .sort((a, b) => b[1] - a[1]).slice(0, 20).map(([name, score]) => {
                          const max = Math.max(...Object.values(selection.scores!).filter((value): value is number => value !== null && Number.isFinite(value)), 1e-12);
                          return <div className="selection-score" key={name}><span>{name}</span><span className="selection-track"><span style={{ width: `${Math.max(0, score / max * 100)}%` }} /></span><strong>{score.toPrecision(4)}</strong></div>;
                        })}
                    </div>}
                    <p className="muted">Les scores sont propres à la méthode choisie ; « — » signifie indisponible, pas zéro.</p></>; })()}
              </section>}
              <section className="panel"><h2>Analyse et exports complémentaires</h2>
                <div className="downloads"><button disabled={!!busy} onClick={() => void state.prepareExcel()}>Préparer le dataset Excel</button>
                  <button disabled={!!busy} onClick={() => void state.inspectResults()}>Analyser le dataset transformé</button></div>
                {state.afterEda && <><p>{state.afterEda.shape[0]} lignes · {state.afterEda.shape[1]} colonnes · {state.afterEda.missing_pct.toFixed(4)} % de manquants · {state.afterEda.duplicate_count} doublons</p>
                  <details><summary>Statistiques des sorties</summary><pre className="json-report">{JSON.stringify(state.afterEda, null, 2)}</pre></details></>}
              </section>
              <section className="panel"><h2>Configuration du pipeline</h2>
                <p className="muted">Le fichier JSON contient les choix et les options de cette exécution. Il peut être archivé avec le rapport.</p>
                <button type="button" onClick={exportConfig}>Exporter la configuration JSON</button>
              </section>
            </>}
          </Suspense>
        </WorkflowStage>
        </AnimatePresence>
        {step > 0 && <button className="back-button" disabled={!!busy} onClick={() => go(step - 1)}>Retour à l’étape précédente</button>}
      </main>
      <footer className="workspace-footer"><span aria-hidden="true" /><span>Statistiques, décisions, transformations</span></footer>
    </div>
  </div>;
}

function VisualLab() {
  const [mode, setMode] = useState<Atmosphere>("orbital");
  const modes: { key: Atmosphere; label: string }[] = [
    { key: "orbital", label: "Orbites scientifiques" }, { key: "particles", label: "Particules organiques" },
    { key: "geometry", label: "Structures géométriques" }, { key: "leaves", label: "Feuilles" },
    { key: "spatial", label: "Atmosphère spatiale" },
  ];
  return <main className="visual-lab"><a href="/">← Retour à l’atelier</a><p>Laboratoire visuel · développement</p>
    <h1>Cinq études de profondeur.</h1><div role="group" aria-label="Choisir une ambiance">
      {modes.map(item => <button key={item.key} aria-pressed={mode === item.key} onClick={() => setMode(item.key)}>{item.label}</button>)}
    </div><section><Suspense fallback={<div className="orbital-fallback" />}><OrbitalScene mode={mode} /></Suspense><p>{modes.find(item => item.key === mode)?.label}</p></section></main>;
}

createRoot(document.getElementById("root")!).render(import.meta.env.DEV && location.search.includes("visual-lab")
  ? <VisualLab /> : <AnimationSystem><App /></AnimationSystem>);
