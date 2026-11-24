import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { SceneNode } from "./ScientificScene";

export type Atmosphere = "orbital" | "particles" | "geometry" | "leaves" | "spatial";
export function sceneAnimationAllowed(active: boolean, visible: boolean, pageVisible: boolean, reducedMotion: boolean) {
  return active && visible && pageVisible && !reducedMotion;
}

export function OrbitalScene({ mode = "orbital", active = true, nodes }: { mode?: Atmosphere; active?: boolean; nodes?: SceneNode[] }) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const mount = host.current;
    if (!mount) return;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" }); }
    catch { mount.dataset.webgl = "unavailable"; return; }
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, .1, 100);
    camera.position.set(0, 0, 8);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.7));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = false;
    mount.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("aria-hidden", "true");
    renderer.domElement.tabIndex = -1;
    const group = new THREE.Group(); scene.add(group);
    scene.add(new THREE.AmbientLight(0xcde8d2, 1.3));
    const light = new THREE.PointLight(0xeac999, 85); light.position.set(-3, 4, 4); scene.add(light);
    const green = new THREE.MeshStandardMaterial({ color: 0x285740, metalness: .55, roughness: .32 });
    const gold = new THREE.MeshStandardMaterial({ color: 0xcdb78b, metalness: .62, roughness: .3 });
    const pale = new THREE.MeshBasicMaterial({ color: 0xd9e9da, transparent: true, opacity: .72 });
    const core = new THREE.Mesh(new THREE.IcosahedronGeometry(mode === "geometry" ? 1.18 : 1.05, mode === "geometry" ? 0 : 3), green);
    group.add(core);
    if (mode === "orbital" || mode === "spatial") {
      for (let i = 0; i < 3; i++) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(1.8 + i * .43, .008 + i * .002, 5, 160), pale);
        ring.rotation.set(.45 + i * .41, i * .7, i * .44); group.add(ring);
      }
    }
    if (mode === "leaves") {
      for (let i = 0; i < 17; i++) {
        const leaf = new THREE.Mesh(new THREE.SphereGeometry(1, 8, 5), i % 4 ? green : gold);
        const a = i * 2.399; const r = 1.55 + (i % 5) * .25;
        leaf.position.set(Math.cos(a) * r, Math.sin(a) * r * .65, Math.sin(a * 1.5));
        leaf.scale.set(.13, .35, .035); leaf.rotation.z = a; group.add(leaf);
      }
    }
    const count = mode === "particles" ? 420 : 150;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const a = i * 2.399, r = 1.45 + (i % 29) / 29 * 2.6;
      positions[i * 3] = Math.cos(a) * r;
      positions[i * 3 + 1] = Math.sin(a) * r * .7;
      positions[i * 3 + 2] = Math.sin(i * 1.7) * 1.5;
    }
    const pointsGeometry = new THREE.BufferGeometry();
    pointsGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const particles = new THREE.Points(pointsGeometry, new THREE.PointsMaterial({ color: 0xe4d3a9, size: .025, transparent: true, opacity: .7 }));
    group.add(particles);
    const resize = () => { const width = mount.clientWidth, height = mount.clientHeight; if (!width || !height) return;
      camera.aspect = width / height; camera.updateProjectionMatrix(); renderer.setSize(width, height); renderer.render(scene, camera); };
    const observer = new ResizeObserver(resize); observer.observe(mount); resize();
    const reduced = matchMedia("(prefers-reduced-motion: reduce)");
    let frame = 0, visible = false; const clock = new THREE.Clock();
    const allowed = () => sceneAnimationAllowed(active, visible, !document.hidden, reduced.matches);
    const draw = () => {
      if (!allowed()) { frame = 0; return; }
      const delta = Math.min(clock.getDelta(), .05); group.rotation.y += delta * .08; particles.rotation.z += delta * .014;
      core.rotation.x += delta * .05; renderer.render(scene, camera); frame = requestAnimationFrame(draw);
    };
    const sync = () => {
      if (!allowed()) { cancelAnimationFrame(frame); frame = 0; return; }
      if (!frame) { clock.getDelta(); frame = requestAnimationFrame(draw); }
    };
    const viewport = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync(); });
    viewport.observe(mount);
    document.addEventListener("visibilitychange", sync);
    reduced.addEventListener("change", sync);
    return () => { cancelAnimationFrame(frame); document.removeEventListener("visibilitychange", sync); reduced.removeEventListener("change", sync);
      viewport.disconnect(); observer.disconnect();
      scene.traverse(object => { if (object instanceof THREE.Mesh || object instanceof THREE.Points) object.geometry.dispose(); });
      green.dispose(); gold.dispose(); pale.dispose(); (particles.material as THREE.Material).dispose(); renderer.dispose(); renderer.domElement.remove(); };
  }, [mode, active, nodes?.map(node => node.state).join("|")]);
  return <div className="orbital-canvas" ref={host} aria-hidden="true"><div className="orbital-fallback" /></div>;
}
