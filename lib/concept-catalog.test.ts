import assert from "node:assert/strict";
import test from "node:test";
import {
  defaultConceptSelection,
  expandCompactBins,
  loadConceptCatalog,
  loadConceptEvidence,
  loadConceptSearch,
  resetConceptCatalogCache,
  resolveConcept,
  suggestConcepts,
  UnknownConceptError,
} from "./concept-catalog.ts";

const responses: Record<string, unknown> = {
  "/data/v1/manifest.json": { release: "releases/r1", complete: true, fullCatalog: true },
  "/data/v1/releases/r1/manifest.json": {
    releaseFingerprint: "fingerprint",
    catalogVersion: "v1",
    complete: true,
    corpus: { id: "met", version: "c1", label: "The Met", count: 10, countingUnit: "physical-object" },
    model: { id: "siglip", revision: "rev", promptTemplateVersion: "p1" },
    metric: { id: "score-qualified-visual-concentration-lift", version: "m1", qualifiedFraction: 0.001 },
    files: { bins: "bins.json", concepts: "concepts.json", seriesTemplate: "series/{conceptId}.json", evidenceTemplate: "evidence/{conceptId}.json" },
  },
  "/data/v1/releases/r1/concepts.json": {
    concepts: [
      { id: "horse", label: "Horse", normalized: "horse", aliases: ["stallion", "mare"], category: "Animals" },
      { id: "ship", label: "Ship", normalized: "ship", aliases: ["vessel"], category: "Objects" },
      { id: "warrior", label: "Warrior", normalized: "warrior", aliases: ["soldier"], category: "People" },
      { id: "battle", label: "Battle", normalized: "battle", aliases: ["war"], category: "Themes" },
    ],
  },
  "/data/v1/releases/r1/bins.json": {
    keys: ["1900", "1910"], labels: ["1900s", "1910s"], starts: [1900, 1910], ends: [1909, 1919],
    denominators: [100, 2], objectCounts: [90, 2], clusterCounts: [80, 2], unreliableIndices: [1],
  },
  "/data/v1/releases/r1/series/horse.json": {
    conceptId: "horse", k: 4, threshold: 0.2, candidateK: 10, candidateThreshold: 0.1, lowSignal: false,
    diagnostics: { standardizedSeparation: 3, controlMean: 0, controlStdDev: 1, promptTopKJaccard: 0.8, reasons: [] },
    pointIndices: [0], values: [2.5], shares: [0.025], hitMasses: [2.5], objectCounts: [4], clusterCounts: [3],
    suppressedBinIndices: [1], defaultEvidenceBinIndex: 0,
  },
  "/data/v1/releases/r1/series/ship.json": {
    conceptId: "ship", k: 2, threshold: 0.2, lowSignal: false,
    diagnostics: { standardizedSeparation: 2, controlMean: 0, controlStdDev: 1, promptTopKJaccard: 0.7, reasons: [] },
    pointIndices: [0], values: [1.5], shares: [0.015], hitMasses: [1.5], objectCounts: [2], clusterCounts: [2], defaultEvidenceBinIndex: 0,
  },
  "/data/v1/releases/r1/evidence/horse.json": {
    conceptId: "horse", percentile: 0.001, threshold: 0.2,
    artworks: {
      a1: {
        artworkId: "a1", physicalObjectId: "a1", visualClusterId: "v1", institution: "met", title: "Horse",
        artist: "Artist", dateDisplay: "1905", dateStart: 1905, dateEnd: 1905, dateQualifier: "exact",
        imageUrl: "https://example.test/horse.jpg", sourceRecordUrl: "https://example.test/a1", metadataLicense: "cc0",
        imageRightsUri: "cc0", creditLine: "Gift", publicDomain: true, rawScore: 0.3, contributor: true,
      },
    },
    periods: [{ binIndex: 0, contributorCount: 4, artworkIds: ["a1"], contributionWeights: [0.75] }],
  },
};

function fixtureFetch(calls: string[]) {
  return (async (input: RequestInfo | URL) => {
    const url = String(input);
    calls.push(url);
    const payload = responses[url];
    return payload === undefined
      ? new Response("missing", { status: 404 })
      : Response.json(payload);
  }) as typeof fetch;
}

test("resolves labels and aliases exactly while exposing the canonical concept", async () => {
  resetConceptCatalogCache();
  const catalog = await loadConceptCatalog({ fetch: fixtureFetch([]) });
  assert.deepEqual(resolveConcept(catalog, "  STALLION "), {
    requested: "STALLION",
    requestedNormalized: "stallion",
    concept: catalog.concepts[0],
    matchedText: "stallion",
    matchedBy: "alias",
  });
  assert.equal(resolveConcept(catalog, "pony"), null);
  assert.equal(suggestConcepts(catalog, "hors")[0].concept.id, "horse");
  assert.equal(suggestConcepts(catalog, "war")[0].concept.id, "battle");
  assert.equal(suggestConcepts(catalog, "war")[0].matchedText, "war");
  await assert.rejects(loadConceptSearch(catalog, ["pony"]), UnknownConceptError);
});

test("expands compact bins and selected series without fetching unselected series", async () => {
  assert.deepEqual(expandCompactBins(responses["/data/v1/releases/r1/bins.json"]).map((bin) => ({ key: bin.key, unreliable: bin.belowMinimumDenominator })), [
    { key: "1900", unreliable: false },
    { key: "1910", unreliable: true },
  ]);
  resetConceptCatalogCache();
  const calls: string[] = [];
  const catalog = await loadConceptCatalog({ fetch: fixtureFetch(calls) });
  const response = await loadConceptSearch(catalog, ["stallion"]);
  assert.equal(response.queries[0].label, "Horse");
  assert.deepEqual(response.series[0].points[0], {
    binKey: "1900", value: 2.5, share: 0.025, lift: 2.5, hitMass: 2.5, objectCount: 4, clusterCount: 3,
  });
  assert.deepEqual(response.series[0].suppressedBinKeys, ["1910"]);
  assert.equal(calls.filter((url) => url.endsWith("series/horse.json")).length, 1);
  assert.equal(calls.some((url) => url.endsWith("series/ship.json")), false);
  await loadConceptSearch(catalog, ["Horse"]);
  assert.equal(calls.filter((url) => url.endsWith("series/horse.json")).length, 1);
  assert.deepEqual(defaultConceptSelection(catalog, response), { queryId: "concept:horse", binKey: "1900" });
});

test("reconstructs deduplicated evidence lazily and caches the concept bundle", async () => {
  resetConceptCatalogCache();
  const calls: string[] = [];
  const catalog = await loadConceptCatalog({ fetch: fixtureFetch(calls) });
  await loadConceptSearch(catalog, ["Horse"]);
  assert.equal(calls.some((url) => url.includes("/evidence/")), false);
  const selected = await loadConceptEvidence(catalog, "horse", "1900");
  assert.equal(selected?.contributorCount, 4);
  assert.equal(selected?.slices.strongest[0].contributionWeight, 0.75);
  assert.equal(selected?.slices.strongest[0].title, "Horse");
  assert.equal(await loadConceptEvidence(catalog, "horse", "1910"), null);
  assert.equal(calls.filter((url) => url.endsWith("evidence/horse.json")).length, 1);
  assert.equal(calls.filter((url) => url.endsWith("series/horse.json")).length, 1);
});
