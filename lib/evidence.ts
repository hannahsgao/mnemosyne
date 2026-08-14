import type {
  ChartSelection,
  EvidenceArtwork,
  EvidenceSlices,
  QueryDescriptor,
  SearchResponse,
} from "./types";

export const MAX_NEAREST_MATCHES = 20;

export type NearestMatchGroup = {
  query: QueryDescriptor;
  artworks: EvidenceArtwork[];
};

/**
 * Select exploratory nearest neighbors without allowing them into timeline evidence.
 *
 * The all-series `k === 0` guard prevents a partial miss from competing with
 * score-qualified results. A round-robin rank merge avoids comparing raw
 * cosine scores across queries while enforcing one page-wide result budget.
 */
export function nearestMatchGroups(
  response: Pick<SearchResponse, "queries" | "series"> | null,
): NearestMatchGroup[] {
  if (
    !response?.series.length ||
    response.series.some((series) => series.k !== 0)
  ) {
    return [];
  }
  const queryById = new Map(response.queries.map((query) => [query.id, query]));
  const candidates = response.series.flatMap((series) => {
    const query = queryById.get(series.queryId);
    if (!query || !Array.isArray(series.nearestMatches)) return [];

    const seen = new Set<string>();
    const artworks: EvidenceArtwork[] = [];
    for (const artwork of series.nearestMatches) {
      if (!artwork || seen.has(artwork.artworkId)) continue;
      seen.add(artwork.artworkId);
      artworks.push(artwork);
      if (artworks.length === MAX_NEAREST_MATCHES) break;
    }
    return artworks.length ? [{ query, artworks }] : [];
  });

  const groups = candidates.map(({ query }) => ({ query, artworks: [] as EvidenceArtwork[] }));
  const cursors = candidates.map(() => 0);
  const seenArtworkIds = new Set<string>();
  let selectedCount = 0;

  while (selectedCount < MAX_NEAREST_MATCHES) {
    let selectedThisRound = false;
    for (let index = 0; index < candidates.length; index += 1) {
      const source = candidates[index].artworks;
      while (
        cursors[index] < source.length &&
        seenArtworkIds.has(source[cursors[index]].artworkId)
      ) {
        cursors[index] += 1;
      }
      if (cursors[index] >= source.length) continue;

      const artwork = source[cursors[index]];
      cursors[index] += 1;
      seenArtworkIds.add(artwork.artworkId);
      groups[index].artworks.push(artwork);
      selectedCount += 1;
      selectedThisRound = true;
      if (selectedCount === MAX_NEAREST_MATCHES) break;
    }
    if (!selectedThisRound) break;
  }

  return groups.filter((group) => group.artworks.length > 0);
}

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
