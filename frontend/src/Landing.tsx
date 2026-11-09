import { DataScene } from "./DataScene";

const pipeline = [
  ["Import", "Chargez votre fichier et choisissez la colonne cible si vous en avez une."],
  ["Exploration", "Repérez les valeurs manquantes, les distributions et les corrélations."],
  ["Prétraitement", "Comparez les transformations proposées avant de les appliquer."],
  ["Export", "Récupérez les jeux train et test en CSV ainsi que le rapport Excel."],
];

/** Present the real data workflow and a manipulable example dataset. */
export function Landing({ onDemo }: { onDemo: () => void }) {
  return <main id="main-content" className="landing" tabIndex={-1}>
    <section className="container hero" aria-labelledby="hero-title" data-enter>
      <div className="hero-copy">
        <h1 id="hero-title">Vos données, prêtes à apprendre.</h1>
        <p>Explorez, nettoyez et transformez vos jeux de données dans un atelier local.</p>
        <div className="hero-actions">
          <a className="primary" href="#atelier">Ouvrir l’atelier</a>
          <button type="button" className="secondary" onClick={onDemo}>Essayer la démo</button>
        </div>
      </div>
      <DataScene />
    </section>
    <section className="container workflow" id="parcours" aria-labelledby="workflow-title">
      <div>
        <h2 id="workflow-title">Le parcours de préparation</h2>
        <p>Les transformations sont ajustées sur le train, puis appliquées au test avec les mêmes paramètres.</p>
        <div className="split-example" data-visual aria-label="Découpage par défaut, 80 % train et 20 % test">
          <span>80 % train</span><span>20 % test</span>
        </div>
        <a className="primary" href="#atelier">Préparer un fichier</a>
      </div>
      <ol>{pipeline.map(([title, detail]) => <li key={title}><h3>{title}</h3><p>{detail}</p></li>)}</ol>
    </section>
  </main>;
}
