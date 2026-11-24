import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, downloadFile, featureEdaPath, validateResponse } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("API contracts and honest failures", () => {
  it("reuses existing forced types for feature EDA and clears the override on reset", () => {
    expect(featureEdaPath("s","agent",{agent:"type:FORCE_NUMERIC"})).toBe('/eda/s/feature/agent?display_type=FORCE_NUMERIC');
    expect(featureEdaPath("s","agent",{agent:{type:{action:"FORCE_CATEGORICAL",parameters:""}}})).toBe('/eda/s/feature/agent?display_type=FORCE_CATEGORICAL');
    expect(featureEdaPath("s","agent",{})).toBe('/eda/s/feature/agent');
    expect(featureEdaPath("s","agent",{agent:"none"})).toBe('/eda/s/feature/agent');
  });
  it("serializes composed operations without losing unusual feature names", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ features: [] })));
    vi.stubGlobal("fetch", fetch);
    const body = { session_id: "session", choices: { "âge / 年齢": { missing: { action: "CONSTANT", parameters: "7" }, scaling: { action: "MINMAX", parameters: "" } } } };
    await api("/analyze", body);
    expect(fetch.mock.calls[0][0]).toContain("/api/analyze");
    expect(fetch.mock.calls[0][1].method).toBe("POST");
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual(body);
  });
  it.each([400, 404, 409, 422, 500, 504])("preserves HTTP %s and backend detail", async status => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Erreur réelle" }), { status })));
    await expect(api("/analyze", {})).rejects.toMatchObject({ message: "Erreur réelle", status });
  });
  it("rejects an unavailable backend and a timed out request", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api("/capabilities")).rejects.toThrow("inaccessible");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new DOMException("timeout", "TimeoutError")));
    await expect(api("/capabilities")).rejects.toThrow("délai");
  });
  it("rejects malformed successful JSON instead of fabricating empty analysis", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>error</html>")));
    await expect(api("/analyze", {})).rejects.toMatchObject({ status: 502 });
    expect(() => validateResponse("/analyze", { features: [{ column: "fake" }] })).toThrow(ApiError);
    expect(() => validateResponse("/preprocess", { success: true })).toThrow(ApiError);
    expect(() => validateResponse("/upload", null)).toThrow(ApiError);
  });
  it("validates selection preview including the exported target count", async () => {
    const valid = { status: "estimated_train_fit", original: ["measure", "raw"], selected: ["measure", "raw"],
      rejected: [], encoded_count: 1, selected_count: 2, output_count: 3, target_included: true,
      expected_rows: 64, method: "variance", warnings: [], note: "Aperçu estimé." };
    expect(() => validateResponse("/selection-preview", valid)).not.toThrow();
    expect(() => validateResponse("/selection-preview", { ...valid, selected: undefined })).toThrow(ApiError);
    expect(() => validateResponse("/selection-preview", { ...valid, output_count: "3" })).toThrow(ApiError);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...valid, warnings: null }), { status: 200 })));
    await expect(api("/selection-preview", {})).rejects.toMatchObject({ status: 502,
      message: "Réponse du serveur incompatible ou incomplète. Réessayez l’opération." });
  });
  it("keeps unsupported exports as HTTP failures without initiating a download", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Session introuvable ou expirée." }), { status: 404 })));
    await expect(downloadFile("expired", "full")).rejects.toMatchObject({ status: 404 });
  });
  it("sends DELETE without a request body", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ closed: true })));
    vi.stubGlobal("fetch", fetch);
    await api("/session/id", undefined, "DELETE");
    expect(fetch.mock.calls[0][1].method).toBe("DELETE");
    expect(fetch.mock.calls[0][1].body).toBeUndefined();
  });
});
