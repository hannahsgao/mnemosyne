import assert from "node:assert/strict";
import test from "node:test";
import { keywordEvidenceSlices, selectedEvidenceItems } from "./evidence.ts";
import type { EvidenceArtwork, SelectedEvidence } from "./types.ts";

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

test("keyword evidence populates the canonical and backward-compatible slices", () => {
  const cards = [artwork("MET_10"), artwork("MET_11")];
  const slices = keywordEvidenceSlices(cards);

  assert.deepEqual(slices.strongest, cards);
  assert.deepEqual(slices.randomContributors, cards);
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
