/** Six-stage local scientific workspace, backed by the existing Python API. */
import { lazy, Suspense, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { AnimatePresence, motion } from "framer-motion";
import { AnimatedNumber, AnimationSystem, Notice, WorkflowStage, motionTiming, useReducedMotionPreference } from "./animation";
import { ScientificScene } from "./ScientificScene";
import { stages } from "./workflow";
import { useWorkflow } from "./useWorkflow";
import "./style.css";

const ImportPanel = lazy(() => import("./panels").then(module => ({ default: module.ImportPanel })));
const EdaPanel = lazy(() => import("./panels").then(module => ({ default: module.EdaPanel })));
const CleaningPanel = lazy(() => import("./panels").then(module => ({ default: module.CleaningPanel })));
const ExecutionPanel = lazy(() => import("./panels").then(module => ({ default: module.ExecutionPanel })));
const PipelinePanel = lazy(() => import("./PipelinePanel").then(module => ({ default: module.PipelinePanel })));
const AppliedReport = lazy(() => import("./PipelinePanel").then(module => ({ default: module.AppliedReport })));

function Mark() {
  return <svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
    <path d="m16 3 12 7v13L16 30 4 23V10L16 3Z" stroke="currentColor" strokeWidth="1.4" />
    <path d="m4 10 12 7 12-7M16 17v13M10 6.5l12 7v13" stroke="currentColor" strokeWidth="1.1" />
  </svg>;
}

function App() {
  const state = useWorkflow();
  const reducedMotion = useReducedMotionPreference();
  const { step, dataset, busy, result } = state;
  const current = stages[step];

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
      options: { test_size_pct: state.testSize }, choices: state.choices,
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
      <div className="sidebar-note"><strong>Projet d’ingénierie</strong>
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
                  <button disabled={!!busy} className="demo-button" type="button" onClick={() => void state.openDemo()}>
                    Charger la démo fraude <span aria-hidden="true">↗</span>
                  </button>
                  <div className="import-principles"><span>Ajustement sur train</span><span>Choix par variable</span><span>Rapport Excel</span></div>
                </div>
                <div className="hero-science" data-loaded={!!dataset}>
                  <ScientificScene label="Réseau scientifique tridimensionnel, illustration du moteur" running={!!busy} />
                  <div className="scene-heading"><span>Illustration du moteur</span><span>3D interactive</span></div>
                  <p className="scene-help">Glissez pour orbiter · Molette pour zoomer</p>
                </div>
              </section>
              <ImportPanel dataset={dataset} config={state.config} busy={!!busy} onFile={file => void state.importFile(file)}
                onChange={state.changeConfig} onConfigure={() => void state.configure()} />
            </>}

            {step === 1 && (state.eda ? <>
              <EdaPanel eda={state.eda} feature={state.feature} busy={!!busy} onFeature={state.selectFeature} />
              <div className="advance"><button className="primary" disabled={!!busy} onClick={() => go(2)}>Voir les recommandations</button></div>
            </> : !busy && <button onClick={() => go(1)}>Relancer l’analyse statistique</button>)}

            {step === 2 && (state.recommendations ? <>
              <CleaningPanel rows={state.recommendations} choices={state.choices} selected={state.selected}
                preview={state.preview} busy={!!busy} onChoice={state.choose} onReset={state.acceptAll}
                onPreview={column => { state.compare(column); }} view="recommendations" />
              <div className="advance"><button className="primary" disabled={!!busy} onClick={() => go(3)}>Configurer le pipeline</button></div>
            </> : !busy && <button onClick={() => go(2)}>Relancer les recommandations</button>)}

            {step === 3 && state.recommendations && <>
              <PipelinePanel rows={state.recommendations} choices={state.choices} busy={!!busy}
                onChoice={state.choose} workflow={state.nodes} onNavigate={go} />
              <div className="advance"><button className="primary" disabled={!!busy} onClick={() => {
                state.setPipelineReviewed(true); go(4);
              }}>Valider le pipeline et comparer</button></div>
            </>}

            {step === 4 && dataset && <>
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
              <ExecutionPanel sessionId={dataset.session_id} busy={!!busy} result={result} onRun={size => void state.run(size)}
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
                view="exports" testSize={state.testSize} onTestSize={state.changeTestSize} />
              {state.report ? <AppliedReport report={state.report} /> : !busy &&
                <button onClick={() => void state.openExports()}>Recharger le rapport des transformations</button>}
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
      <footer className="workspace-footer"><span>AutoPrépa / Projet universitaire d’ingénierie</span><span>Statistiques, décisions, transformations</span></footer>
    </div>
  </div>;
}

createRoot(document.getElementById("root")!).render(<AnimationSystem><App /></AnimationSystem>);
