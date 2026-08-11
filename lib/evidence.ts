import type {
  ChartSelection,
  EvidenceArtwork,
  EvidenceSlices,
  SearchResponse,
} from "./types";

/**
 * Build the keyword evidence contract used by the D1 Worker.
 *
 * `randomContributors` remains populated for clients that predate the frontend's
 * use of the canonical `strongest` slice.
 */
export function keywordEvidenceSlices(cards: EvidenceArtwork[]): EvidenceSlices {
  return {
    strongest: cards,
    representative: [],
    borderline: [],
    randomContributors: cards,
    bestNonContributors: [],
    randomDenominator: [],
  };
}

/** Select the canonical evidence slice, with a fallback for legacy keyword payloads. */
export function selectedEvidenceItems(
  response: Pick<SearchResponse, "selectedEvidence"> | null,
  selection: ChartSelection | null,
) {
  if (
    !response?.selectedEvidence ||
    !selection ||
    response.selectedEvidence.queryId !== selection.queryId ||
    response.selectedEvidence.binKey !== selection.binKey
  ) {
    return [];
  }

  const slices = response.selectedEvidence.slices;
  const source = slices.strongest.length
    ? slices.strongest
    : slices.randomContributors;
  const seen = new Set<string>();
  const selected: EvidenceArtwork[] = [];
  for (const artwork of source) {
    if (!artwork || seen.has(artwork.artworkId)) continue;
    seen.add(artwork.artworkId);
    selected.push(artwork);
  }
  return selected;
}
