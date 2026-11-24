import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OrbitalScene, sceneAnimationAllowed } from "./OrbitalScene";
import { ScientificScene } from "./ScientificScene";

afterEach(() => vi.unstubAllGlobals());

describe("decorative scientific scene", () => {
  it("keeps the six sub-step buttons without scene controls", () => {
    vi.stubGlobal("document", { hidden: false });
    const nodes = ["Nettoyage", "Types", "Imputation", "Encodage", "Mise à l’échelle", "Sélection finale"]
      .map(label => ({ label, state: "pending" as const }));
    const html = renderToStaticMarkup(<ScientificScene label="Opérations Python" nodes={nodes} onSelect={() => {}} />);
    expect((html.match(/<button /g) ?? []).length).toBe(6);
    expect(html).not.toContain("Orientation 3D");
    expect(html).not.toContain("Rapprocher");
    expect(html).not.toContain("Recentrer");
    expect(html).not.toContain("animations en pause");
    expect(html).not.toContain('class="scene-tools"');
  });

  it("hides the canvas and pauses animation for reduced motion, hidden tabs and off-screen scenes", () => {
    expect(renderToStaticMarkup(<OrbitalScene />)).toContain('aria-hidden="true"');
    expect(sceneAnimationAllowed(true, true, true, false)).toBe(true);
    expect(sceneAnimationAllowed(true, true, true, true)).toBe(false);
    expect(sceneAnimationAllowed(true, true, false, false)).toBe(false);
    expect(sceneAnimationAllowed(true, false, true, false)).toBe(false);
  });
});
