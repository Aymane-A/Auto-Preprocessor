import { Component, lazy, Suspense, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useReducedMotionPreference } from "./animation";

const OrbitalScene = lazy(() => import("./OrbitalScene").then(module => ({ default: module.OrbitalScene })));

export type SceneNode = { label: string; state: "pending" | "active" | "complete" | "warning" | "error"; disabled?: boolean };
type Props = { nodes?: SceneNode[]; running?: boolean; celebrate?: boolean; onSelect?: (index: number) => void; label: string };

class SceneBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? this.props.fallback : this.props.children; }
}
function SceneFallback({ nodes, failed }: { nodes?: SceneNode[]; failed?: boolean }) {
  return <div className="scene-fallback">
    {nodes ? <ol>{nodes.map(node => <li key={node.label} data-state={node.state}>{node.label}</li>)}</ol> :
      <svg viewBox="0 0 240 150" aria-hidden="true">
        <ellipse cx="120" cy="75" rx="88" ry="37" transform="rotate(-25 120 75)" />
        <ellipse cx="120" cy="75" rx="88" ry="37" transform="rotate(28 120 75)" />
        <path d="m62 86 28-40 37 52 31-61 26 53-57 8-65-12 96-49-68 9 94 44" />
        {[[62,86],[90,46],[127,98],[158,37],[184,90]].map(([x,y]) => <circle key={x} cx={x} cy={y} r="4" />)}
      </svg>}
    {failed && <span>Vue schématique · WebGL indisponible</span>}
  </div>;
}

/** Keep GPU scenes out of the initial bundle and stop drawing them offscreen. */
export function ScientificScene({ nodes, running, celebrate, onSelect, label }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [started, setStarted] = useState(false);
  const [pageVisible, setPageVisible] = useState(!document.hidden);
  const [ambient, setAmbient] = useState(false);
  const reducedMotion = useReducedMotionPreference();
  const signature = nodes?.map(node => node.state).join("|") ?? "network";
  useEffect(() => {
    if (!visible || !pageVisible || reducedMotion) { setAmbient(false); return; }
    setAmbient(true);
    const timer = window.setTimeout(() => setAmbient(false), celebrate ? 1800 : 1400);
    return () => window.clearTimeout(timer);
  }, [visible, pageVisible, reducedMotion, signature, celebrate]);
  useEffect(() => {
    if (!host.current) return;
    const observer = new IntersectionObserver(([entry]) => {
      setVisible(entry.isIntersecting);
      if (entry.isIntersecting) setStarted(true);
    });
    observer.observe(host.current);
    const updateVisibility = () => setPageVisible(!document.hidden);
    document.addEventListener("visibilitychange", updateVisibility);
    return () => {
      observer.disconnect();
      document.removeEventListener("visibilitychange", updateVisibility);
    };
  }, []);

  const fallback = <SceneFallback nodes={nodes} failed />;
  return <figure className={nodes ? "science-scene pipeline-scene" : "science-scene"} ref={host} aria-label={label}
    data-animation={reducedMotion || !visible || !pageVisible ? "paused" : running ? "processing" : ambient ? "entrance" : "idle"}>
    <div className="scene-viewport">
      {started ? <SceneBoundary fallback={fallback}>
        <Suspense fallback={<SceneFallback nodes={nodes} />}>
          <OrbitalScene nodes={nodes} active={visible && pageVisible && !reducedMotion} />
        </Suspense>
      </SceneBoundary> : <SceneFallback nodes={nodes} />}
    </div>
    {nodes && onSelect && <div className="scene-node-controls" role="group" aria-label={label}>
      {nodes.map((node, index) => <button key={node.label} type="button" disabled={node.disabled}
        data-state={node.state} aria-pressed={node.state === "active" || node.state === "error"}
        onClick={() => onSelect(index)}><span aria-hidden="true">{String(index + 1).padStart(2, "0")} · </span>{node.label}</button>)}
    </div>}
  </figure>;
}