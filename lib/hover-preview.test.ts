import assert from "node:assert/strict";
import test from "node:test";
import { resolveHoverPreviewContent, sampleHoverArtwork } from "./hover-preview.ts";
import type {
  EvidenceArtwork,
  EvidenceSlices,
  SelectedEvidence,
} from "./types.ts";

function artwork(
  artworkId: string,
  overrides: Partial<EvidenceArtwork> = {},
): EvidenceArtwork {
  return {
    artworkId,
    physicalObjectId: artworkId,
    visualClusterId: artworkId,
    institution: "Museum",
    title: artworkId,
    artist: "Artist",
    dateDisplay: "1900",
    dateStart: 1900,
    dateEnd: 1900,
    dateQualifier: "exact",
    imageUrl: `/images/${artworkId}.jpg`,
    sourceRecordUrl: `https://example.test/${artworkId}`,
    metadataLicense: "CC0",
    imageRightsUri: "",
    creditLine: "",
    publicDomain: true,
    rawScore: null,
    contributionWeight: 1,
    contributor: true,
    ...overrides,
  };
}

function evidence(overrides: Partial<EvidenceSlices>): SelectedEvidence {
  return {
    queryId: "q-1",
    binKey: "decade:1900",
    slices: {
      strongest: [],
      representative: [],
      borderline: [],
      randomContributors: [],
      bestNonContributors: [],
      randomDenominator: [],
      ...overrides,
    },
  };
}

test("keeps the exact hover caption available without an image", () => {
  const content = resolveHoverPreviewContent(
    { queryId: "q-2", binKey: "period:1900" },
    [
      { id: "q-1", label: "horse", normalized: "horse" },
      { id: "q-2", label: '"ship, sea"', normalized: "ship, sea" },
    ],
    [
      {
        key: "period:1800",
        label: "1800s",
        start: 1800,
        end: 1899,
        denominator: 1,
        objectCount: 1,
        clusterCount: 1,
        belowMinimumDenominator: false,
      },
      {
        key: "period:1900",
        label: "c. 1900–1949",
        start: 1900,
        end: 1949,
        denominator: 1,
        objectCount: 1,
        clusterCount: 1,
        belowMinimumDenominator: false,
      },
    ],
    null,
  );

  assert.deepEqual(content, {
    conceptLabel: '"ship, sea"',
    periodLabel: "c. 1900–1949",
    artwork: null,
  });
});

test("attaches only an image matching the current hover", () => {
  const selection = { queryId: "q-1", binKey: "decade:1920" };
  const queries = [{ id: "q-1", label: "horse", normalized: "horse" }];
  const bins = [{
    key: "decade:1920",
    label: "1920s",
    start: 1920,
    end: 1929,
    denominator: 1,
    objectCount: 1,
    clusterCount: 1,
    belowMinimumDenominator: false,
  }];
  const matchingArtwork = artwork("matching");

  assert.equal(
    resolveHoverPreviewContent(
      selection,
      queries,
      bins,
      { selection, artwork: matchingArtwork },
    )?.artwork,
    matchingArtwork,
  );
  assert.equal(
    resolveHoverPreviewContent(
      selection,
      queries,
      bins,
      {
        selection: { queryId: "q-1", binKey: "decade:1910" },
        artwork: artwork("stale"),
      },
    )?.artwork,
    null,
  );
});

test("does not invent captions for stale hover identifiers", () => {
  const queries = [{ id: "q-1", label: "horse", normalized: "horse" }];
  const bins = [{
    key: "decade:1920",
    label: "1920s",
    start: 1920,
    end: 1929,
    denominator: 1,
    objectCount: 1,
    clusterCount: 1,
    belowMinimumDenominator: false,
  }];

  assert.equal(resolveHoverPreviewContent({ queryId: "stale", binKey: "decade:1920" }, queries, bins, null), null);
  assert.equal(resolveHoverPreviewContent({ queryId: "q-1", binKey: "stale" }, queries, bins, null), null);
});

test("prefers sampled period contributors over representative and strongest cards", () => {
  const selected = sampleHoverArtwork(evidence({
    randomContributors: [artwork("random")],
    representative: [artwork("representative")],
    strongest: [artwork("strongest")],
  }));

  assert.equal(selected?.artworkId, "random");
});

test("falls back across backend-specific evidence slices", () => {
  assert.equal(
    sampleHoverArtwork(evidence({ representative: [artwork("representative")] }))?.artworkId,
    "representative",
  );
  assert.equal(
    sampleHoverArtwork(evidence({ strongest: [artwork("strongest")] }))?.artworkId,
    "strongest",
  );
});

test("ignores missing images, non-contributors, and duplicate visual clusters", () => {
  const selected = sampleHoverArtwork(evidence({
    randomContributors: [
      artwork("missing", { imageUrl: null }),
      artwork("non-contributor", { contributor: false }),
      artwork("duplicate-a", { visualClusterId: "shared" }),
      artwork("duplicate-b", { visualClusterId: "shared" }),
    ],
  }));

  assert.ok(selected);
  assert.equal(selected.visualClusterId, "shared");
  assert.ok(["duplicate-a", "duplicate-b"].includes(selected.artworkId));
});

test("returns one stable sample for the same query and period", () => {
  const payload = evidence({
    randomContributors: [artwork("one"), artwork("two"), artwork("three")],
  });

  const first = sampleHoverArtwork(payload);
  assert.ok(first);
  for (let index = 0; index < 10; index += 1) {
    assert.equal(sampleHoverArtwork(payload)?.artworkId, first.artworkId);
  }
});

test("can advance past an image that failed to load", () => {
  const payload = evidence({
    randomContributors: [artwork("one"), artwork("two"), artwork("three")],
  });
  const first = sampleHoverArtwork(payload);
  assert.ok(first);

  const replacement = sampleHoverArtwork(payload, new Set([first.artworkId]));
  assert.ok(replacement);
  assert.notEqual(replacement.artworkId, first.artworkId);
});

test("returns null when no contributor image can be shown", () => {
  assert.equal(sampleHoverArtwork(null), null);
  assert.equal(
    sampleHoverArtwork(evidence({ strongest: [artwork("missing", { imageUrl: "" })] })),
    null,
  );
});
