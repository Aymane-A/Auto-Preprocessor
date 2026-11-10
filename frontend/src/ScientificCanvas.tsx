import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useReducedMotionPreference } from "./animation";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { CanvasTexture, Color, Group, InstancedMesh, Object3D, Vector3 } from "three";
import type { OrbitControls as OrbitControlsInstance } from "three-stdlib";
import type { SceneNode } from "./ScientificScene";

const colors = {
  mint: "#a7f3d0", emerald: "#10b981", beige: "#d8c3a5", dark: "#24513d",
  pending: "#52725e", active: "#34d399", complete: "#a7f3d0", warning: "#e9b768", error: "#f08c83",
};

function Network({ flowing }: { flowing: boolean }) {
  const instances = useRef<InstancedMesh>(null);
  const satellites = useRef<Group>(null);
  useFrame((_, delta) => {
    if (flowing && satellites.current) satellites.current.rotation.y += Math.min(delta, .05) * .12;
  });
  const { points, links } = useMemo(() => {
    const points = Array.from({ length: 220 }, (_, i) => {
      const y = 1 - 2 * (i + .5) / 220;
      const phase = i * Math.PI * (3 - Math.sqrt(5));
      const cross = Math.sqrt(1 - y * y);
      const radius = i % 4 === 0 ? .8 + .45 * Math.sin(i * 1.73) : 1.7 + .24 * Math.sin(i * 1.73);
      return new Vector3(Math.cos(phase) * cross, y, Math.sin(phase) * cross).multiplyScalar(radius);
    });
    const links: number[] = [];
    points.forEach((point, i) => {
      for (let j = i + 1; j < points.length; j++) {
        if (point.distanceToSquared(points[j]) < .34) links.push(...point.toArray(), ...points[j].toArray());
      }
    });
    return { points, links: new Float32Array(links) };
  }, []);
  useLayoutEffect(() => {
    const mesh = instances.current;
    if (!mesh) return;
    const transform = new Object3D();
    points.forEach((point, i) => {
      transform.position.copy(point);
      transform.scale.setScalar(i % 13 === 0 ? 1.9 : .8 + (i % 7) / 12);
      transform.updateMatrix();
      mesh.setMatrixAt(i, transform.matrix);
      mesh.setColorAt(i, new Color(i % 17 === 0 ? colors.beige : i % 3 ? colors.emerald : colors.mint));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [points]);
  return <group rotation={[-.13, .2, -.26]}>
    <lineSegments>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[links, 3]} /></bufferGeometry>
      <lineBasicMaterial color={colors.emerald} transparent opacity={.29} />
    </lineSegments>
    <instancedMesh ref={instances} args={[undefined, undefined, points.length]}>
      <icosahedronGeometry args={[.034, 0]} />
      <meshStandardMaterial roughness={.3} metalness={.24} emissive={colors.dark} emissiveIntensity={.7} />
    </instancedMesh>
    <mesh>
      <icosahedronGeometry args={[.45, 1]} />
      <meshStandardMaterial color={colors.emerald} metalness={.65} roughness={.24} emissive={colors.emerald} emissiveIntensity={.25} />
    </mesh>
    <mesh rotation={[.4, .2, .7]}><icosahedronGeometry args={[.68, 0]} />
      <meshBasicMaterial color={colors.beige} wireframe transparent opacity={.4} /></mesh>
    <group ref={satellites}>
      {[[2.15, .6, .4], [-1.8, -.8, .8], [.5, 1.9, -.6]].map((position, index) =>
        <mesh key={index} position={position as [number, number, number]} rotation={[.4 + index, .6, .3]}>
          <octahedronGeometry args={[.11 + index * .025, 0]} />
          <meshStandardMaterial color={colors.beige} roughness={.3} metalness={.6} />
        </mesh>)}
    </group>
    {[.25, 1.04, 1.54].map((tilt, index) => <mesh key={tilt} rotation={[tilt, .3 + index * .53, .15]}>
      <torusGeometry args={[2.01 + index * .11, .008, 5, 120]} />
      <meshBasicMaterial color={index === 1 ? colors.beige : colors.emerald} transparent opacity={.45} />
    </mesh>)}
  </group>;
}

const nodePosition = (index: number, count: number): [number, number, number] =>
  [(index - (count - 1) / 2) * 1.12, index % 2 ? -.23 : .23, index % 2 ? -.18 : .18];

function SceneCamera({ nodes, focused, resetKey, reducedMotion }: {
  nodes?: SceneNode[]; focused: boolean; resetKey: number; reducedMotion: boolean;
}) {
  const { camera, size, controls, invalidate } = useThree();
  const orbit = controls as OrbitControlsInstance | null;
  const moving = useRef(false);
  const initialized = useRef(false);
  const destination = useRef(new Vector3());
  const target = useRef(new Vector3());
  const active = nodes?.findIndex(node => node.state === "active") ?? -1;
  const count = nodes?.length ?? 0;
  useLayoutEffect(() => {
    const aspect = size.width / Math.max(1, size.height);
    const distance = count ? Math.max(3.6, (count * 1.12 + .8) / (2 * Math.tan(21 * Math.PI / 180) * aspect)) : 7.6;
    const selectedX = active >= 0 ? nodePosition(active, count)[0] : 0;
    target.current.set(focused ? selectedX : selectedX * .07, 0, 0);
    destination.current.set(target.current.x + (focused ? .3 : 0), focused ? 1 : .65, focused ? distance * .72 : distance);
    if (reducedMotion || !initialized.current) {
      camera.position.copy(destination.current);
      orbit?.target.copy(target.current);
      orbit?.update();
    } else moving.current = true;
    initialized.current = true;
    invalidate();
  }, [camera, orbit, active, count, focused, resetKey, reducedMotion, size.width, size.height, invalidate]);
  useEffect(() => {
    const stop = () => { moving.current = false; };
    orbit?.addEventListener("start", stop);
    return () => orbit?.removeEventListener("start", stop);
  }, [orbit]);
  useFrame((_, delta) => {
    if (!moving.current || !orbit) return;
    const blend = 1 - Math.exp(-Math.min(delta, .05) * 9);
    camera.position.lerp(destination.current, blend);
    orbit.target.lerp(target.current, blend);
    orbit.update();
    moving.current = camera.position.distanceToSquared(destination.current) + orbit.target.distanceToSquared(target.current) > .00001;
    if (moving.current) invalidate();
  });
  return null;
}

function NodeNumber({ index }: { index: number }) {
  const texture = useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 48;
    const context = canvas.getContext("2d");
    if (context) {
      context.fillStyle = colors.mint;
      context.font = "600 30px Segoe UI, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(String(index + 1).padStart(2, "0"), 48, 24);
    }
    return new CanvasTexture(canvas);
  }, [index]);
  useEffect(() => () => texture.dispose(), [texture]);
  return <sprite position={[0, -.55, 0]} scale={[.6, .3, 1]}>
    <spriteMaterial map={texture} transparent depthTest={false} />
  </sprite>;
}

function Pipeline({ nodes, onSelect }: { nodes: SceneNode[]; onSelect?: (index: number) => void }) {
  const [hovered, setHovered] = useState<number | null>(null);
  const positions = useMemo(() => nodes.map((_, i) => nodePosition(i, nodes.length)), [nodes.length]);
  const links = useMemo(() => new Float32Array(positions.slice(1).flatMap((point, i) => [...positions[i], ...point])), [positions]);
  return <>
    <lineSegments>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[links, 3]} /></bufferGeometry>
      <lineBasicMaterial color={colors.emerald} transparent opacity={.72} />
    </lineSegments>
    {nodes.map((node, i) => <group key={node.label} position={positions[i]}>
      <mesh scale={hovered === i ? 1.18 : 1}
        onPointerOver={node.disabled ? undefined : event => { event.stopPropagation(); setHovered(i); }}
        onPointerOut={() => setHovered(null)}
        onClick={node.disabled ? undefined : event => { event.stopPropagation(); onSelect?.(i); }}>
        <icosahedronGeometry args={[node.state === "active" ? .3 : .25, 0]} />
        <meshStandardMaterial color={hovered === i ? colors.beige : colors[node.state]} emissive={colors[node.state]}
          emissiveIntensity={node.state === "active" ? .65 : .15} roughness={.25} metalness={.45} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[.43, node.state === "active" ? .025 : .012, 6, 48]} />
        <meshBasicMaterial color={node.state === "active" || hovered === i ? colors.beige : colors[node.state]} />
      </mesh>
      <NodeNumber index={i} />
    </group>)}
  </>;
}

function Flow({ count }: { count?: number }) {
  const particles = useRef<InstancedMesh>(null);
  const transform = useMemo(() => new Object3D(), []);
  const start = useMemo(() => new Vector3(), []);
  const end = useMemo(() => new Vector3(), []);
  const elapsed = useRef(0);
  useFrame((_, delta) => {
    if (!particles.current) return;
    elapsed.current += Math.min(delta, .05);
    for (let i = 0; i < 18; i++) {
      const t = (elapsed.current * .38 + i / 18) % 1;
      if (count && count > 1) {
        const distance = t * (count - 1);
        const segment = Math.floor(distance);
        start.set(...nodePosition(segment, count));
        end.set(...nodePosition(segment + 1, count));
        transform.position.lerpVectors(start, end, distance - segment);
        transform.position.z += .07 + (i % 3) * .07;
      } else {
        const radius = .32 + 1.95 * t;
        const phase = i * .77 + t * 5;
        transform.position.set(Math.cos(phase) * radius, Math.sin(phase) * radius * .57, Math.sin(phase) * radius);
      }
      transform.updateMatrix();
      particles.current.setMatrixAt(i, transform.matrix);
    }
    particles.current.instanceMatrix.needsUpdate = true;
  });
  return <instancedMesh ref={particles} args={[undefined, undefined, 18]} frustumCulled={false}>
    <sphereGeometry args={[.042, 7, 5]} />
    <meshBasicMaterial color={colors.mint} />
  </instancedMesh>;
}

export default function ScientificCanvas({ nodes, running, visible, rotation, focused, resetKey, onSelect, onUnavailable, fallback }: {
  nodes?: SceneNode[]; running: boolean; visible: boolean; onSelect?: (index: number) => void;
  onUnavailable: () => void; fallback: ReactNode;
  rotation: number;
  focused: boolean; resetKey: number;
}) {
  const reducedMotion = useReducedMotionPreference();
  const container = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const node = container.current;
    if (!node) return;
    const loseContext = (event: Event) => {
      event.preventDefault();
      onUnavailable();
    };
    node.addEventListener("webglcontextlost", loseContext, true);
    return () => node.removeEventListener("webglcontextlost", loseContext, true);
  }, [onUnavailable]);

  const flowing = visible && running && !reducedMotion;
  return <div className="gpu-canvas" ref={container} aria-hidden="true">
    <Canvas fallback={fallback} frameloop={!visible ? "never" : flowing ? "always" : "demand"}
      dpr={[1, 1.5]} camera={{ position: nodes ? [0, 1.1, 10] : [0, .25, 7.25], fov: 42 }}
      gl={{ alpha: true, antialias: true, powerPreference: "low-power" }}>
      <SceneCamera nodes={nodes} focused={focused} resetKey={resetKey} reducedMotion={!!reducedMotion} />
      <fog attach="fog" args={["#111b14", 9, 20]} />
      <ambientLight intensity={.6} />
      <directionalLight color={colors.mint} position={[1, 4, 4]} intensity={3} />
      <pointLight color={colors.beige} position={[3, -1, -2]} intensity={18} />
      <pointLight color={colors.emerald} position={[-3, 0, 3]} intensity={16} />
      <group rotation={[0, rotation * Math.PI / 180, 0]}>
        {nodes ? <Pipeline nodes={nodes} onSelect={onSelect} /> : <Network flowing={flowing} />}
        {flowing && <Flow count={nodes?.length} />}
      </group>
      <OrbitControls makeDefault enablePan={false} enableZoom enableDamping={!reducedMotion}
        minDistance={2} maxDistance={18} zoomSpeed={.55} minPolarAngle={.65} maxPolarAngle={2.5} />
    </Canvas>
  </div>;
}
