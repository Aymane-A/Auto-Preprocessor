import { useState } from "react";
import propertyPoints from "./property-points.json";

type Point = [number, number, number];

const labels = ["Surface", "Prix", "Pièces"];
const ranges = labels.map((_, axis) => {
  const values = propertyPoints.map(point => point[axis]);
  return [Math.min(...values), Math.max(...values)];
});

const points = propertyPoints.map(point => point.map((value, axis) => {
  const [min, max] = ranges[axis];
  return (value - min) / (max - min) - .5;
}) as Point);

const grid = Array.from({ length: 7 }, (_, index) => index / 6 - .5);
const axisEnds: Point[] = [[.62, -.5, -.5], [-.5, .62, -.5], [-.5, -.5, .62]];
const origin: Point = [-.5, -.5, -.5];

/** Project a bundled housing sample from three measured dimensions. */
export function DataScene() {
  const [angle, setAngle] = useState(-30);

  function project([x, y, z]: Point) {
    const yaw = angle * Math.PI / 180;
    const horizontal = x * Math.cos(yaw) - z * Math.sin(yaw);
    const depth = x * Math.sin(yaw) + z * Math.cos(yaw);
    const vertical = y * .91 - depth * .41;
    const distance = y * .41 + depth * .91;
    const scale = 234 / (1.4 - distance * .32);
    return { x: 260 + horizontal * scale, y: 182 - vertical * scale, distance };
  }

  const start = project(origin);
  const plotted = points.map((point, index) => ({ ...project(point), index }))
    .sort((left, right) => left.distance - right.distance);

  return <div className="data-scene">
    <div className="scene-top">
      <div><strong>Nuage de points</strong><span>Jeu de démonstration immobilier</span></div>
      <span className="scene-count">{points.length} observations</span>
    </div>
    <svg viewBox="55 23 410 314" role="img" aria-labelledby="scatter-title scatter-desc">
      <title id="scatter-title">Nuage immobilier en trois dimensions</title>
      <desc id="scatter-desc">Chaque point représente un logement selon sa surface, son prix et son nombre de pièces.</desc>
      <defs>
        <radialGradient id="point-light" cx="34%" cy="29%" r="70%">
          <stop className="point-highlight" />
          <stop offset="1" className="point-shadow" />
        </radialGradient>
      </defs>
      {grid.flatMap(value => [
        [[-.5, -.5, value], [.5, -.5, value]],
        [[value, -.5, -.5], [value, -.5, .5]],
        [[-.5, value, -.5], [.5, value, -.5]],
        [[value, -.5, -.5], [value, .5, -.5]],
        [[-.5, value, -.5], [-.5, value, .5]],
        [[-.5, -.5, value], [-.5, .5, value]],
      ].map(([from, to], index) => {
        const a = project(from as Point), b = project(to as Point);
        return <line className="scene-grid" key={`${value}-${index}`} x1={a.x} x2={b.x} y1={a.y} y2={b.y} />;
      }))}
      {axisEnds.map((point, index) => {
        const end = project(point);
        return <g key={labels[index]}>
          <line className="scene-axis" x1={start.x} y1={start.y} x2={end.x} y2={end.y} />
          <text className="scene-label" x={Math.min(435, Math.max(85, end.x))}
            y={Math.min(329, Math.max(39, end.y + (index === 1 ? -9 : 16)))}
            textAnchor="middle">{labels[index]}</text>
        </g>;
      })}
      {plotted.map(({ x, y, distance, index }) => <circle key={index}
        className="scene-point" cx={x} cy={y}
        r={3.1 + distance * 1.3} opacity={.76 + distance * .26} />)}
    </svg>
    <div className="scene-controls">
      <label htmlFor="scene-angle">Rotation 3D</label>
      <input id="scene-angle" type="range" min="-85" max="85" value={angle}
        onChange={event => setAngle(event.currentTarget.valueAsNumber)} />
      <output htmlFor="scene-angle">{angle}°</output>
    </div>
  </div>;
}
