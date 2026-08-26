import assert from "node:assert/strict";
import test from "node:test";
import {
  evidencePreviewLabel,
  keywordEvidenceSlices,
  nearestMatchGroups,
  selectedEvidenceItems,
} from "./evidence.ts";
import type { EvidenceArtwork, SearchSeries, SelectedEvidence } from "./types.ts";

function artwork(artworkId: string): EvidenceArtwork {
  return {
    artworkId,
    physicalObjectId: artworkId,
    visualClusterId: artworkId,
    institution: "The Metropolitan Museum of Art",
    title: artworkId,
    artist: "Unknown artist",
    dateDisplay: "1880",
    dateStart: 1880,
    dateEnd: 1880,
    dateQualifier: "exact",
    imageUrl: null,
    sourceRecordUrl: `https://example.test/${artworkId}`,
    metadataLicense: "CC0",
    imageRightsUri: "",
    creditLine: "",
    publicDomain: true,
    rawScore: null,
    contributionWeight: 1,
    contributor: true,
  };
}

function response(selectedEvidence: SelectedEvidence) {
  return { selectedEvidence };
}

function series(
  queryId: string,
  k: number,
  nearestMatches?: EvidenceArtwork[],
): SearchSeries {
  return {
    queryId,
    k,
    threshold: k ? 0.125 : null,
    lowSignal: k === 0,
    diagnostics: {
      standardizedSeparation: null,
      controlMean: null,
      controlStdDev: null,
      promptTopKJaccard: null,
      reasons: [],
    },
    points: [],
    nearestMatches,
  };
}

test("keyword evidence populates the canonical and backward-compatible slices", () => {
  const cards = [artwork("MET_10"), artwork("MET_11")];
  const slices = keywordEvidenceSlices(cards);

  assert.deepEqual(slices.strongest, cards);
  assert.deepEqual(slices.randomContributors, cards);
});

test("evidence preview labels distinguish rendered cards from all period matches", () => {
  assert.equal(
    evidencePreviewLabel(5, 8, 55, "catalog-record"),
    "Showing 5 of 8 preview cards from 55 matching catalog records",
  );
  assert.equal(
    evidencePreviewLabel(8, 8, 55, "catalog-record"),
    "Showing 8 preview cards from 55 matching catalog records",
  );
  assert.equal(
    evidencePreviewLabel(5, 25, 162, "physical-object"),
    "Showing 5 of 25 preview cards from 162 matching works",
  );
  assert.equal(
    evidencePreviewLabel(1, 1, 1, "visual-cluster"),
    "Showing 1 preview card from 1 matching work",
  );
});

test("frontend evidence selection prefers strongest and falls back to legacy contributors", () => {
  const selection = { queryId: "q-horse", binKey: "1880:1889" };
  const canonical = artwork("MET_10");
  const legacy = artwork("MET_11");
  const slices = keywordEvidenceSlices([canonical]);
  slices.randomContributors = [legacy];

  assert.deepEqual(
    selectedEvidenceItems(
      response({ ...selection, slices }),
      selection,
    ).map((item) => item.artworkId),
    ["MET_10"],
  );

  slices.strongest = [];
  slices.randomContributors = [legacy, legacy];
  assert.deepEqual(
    selectedEvidenceItems(
      response({ ...selection, slices }),
      selection,
    ).map((item) => item.artworkId),
    ["MET_11"],
  );
});

test("nearest matches are capped at 20 unique works for one all-unmatched search", () => {
  const first = artwork("MET_0");
  const nearest = [first, first, ...Array.from({ length: 24 }, (_, index) => artwork(`MET_${index + 1}`))];
  const groups = nearestMatchGroups({
    queries: [{ id: "q-lonely", label: "lonely", normalized: "lonely" }],
    series: [series("q-lonely", 0, nearest)],
  });

  assert.equal(groups.length, 1);
  assert.equal(groups[0].query.label, "lonely");
  assert.equal(groups[0].artworks.length, 20);
  assert.deepEqual(
    groups[0].artworks.slice(0, 3).map((item) => item.artworkId),
    ["MET_0", "MET_1", "MET_2"],
  );
  assert.equal(new Set(groups[0].artworks.map((item) => item.artworkId)).size, 20);
});

test("nearest matches require every term to miss the strict cutoff", () => {
  assert.deepEqual(nearestMatchGroups({
    queries: [
      { id: "q-lonely", label: "lonely", normalized: "lonely" },
      { id: "q-horse", label: "horse", normalized: "horse" },
    ],
    series: [
      series("q-lonely", 0, [artwork("MET_nearest")]),
      series("q-horse", 4),
    ],
  }), []);
});

test("multi-term nearest matches share one round-robin budget of 20", () => {
  const shared = artwork("MET_shared");
  const groups = nearestMatchGroups({
    queries: [
      { id: "q-lonely", label: "lonely", normalized: "lonely" },
      { id: "q-happy", label: "happy", normalized: "happy" },
    ],
    series: [
      series("q-lonely", 0, [shared, ...Array.from({ length: 20 }, (_, index) => artwork(`L_${index}`))]),
      series("q-happy", 0, [shared, ...Array.from({ length: 20 }, (_, index) => artwork(`H_${index}`))]),
    ],
  });

  const selected = groups.flatMap((group) => group.artworks);
  assert.equal(groups.length, 2);
  assert.equal(selected.length, 20);
  assert.equal(new Set(selected.map((item) => item.artworkId)).size, 20);
  assert.deepEqual(groups.map((group) => group.artworks.length), [10, 10]);
  assert.deepEqual(groups[0].artworks.slice(0, 2).map((item) => item.artworkId), ["MET_shared", "L_0"]);
  assert.deepEqual(groups[1].artworks.slice(0, 2).map((item) => item.artworkId), ["H_0", "H_1"]);
});

test("nearest matches remain an optional additive contract", () => {
  assert.deepEqual(nearestMatchGroups({
    queries: [{ id: "q-lonely", label: "lonely", normalized: "lonely" }],
    series: [series("q-lonely", 0)],
  }), []);
});
