import type {
  ChartSelection,
  EvidenceArtwork,
  EvidenceSliceName,
  QueryDescriptor,
  SelectedEvidence,
  TimeBin,
} from "./types";

const HOVER_SLICE_PRIORITY = [
  "randomContributors",
  "representative",
  "strongest",
] as const satisfies readonly EvidenceSliceName[];

type ArtworkPreview = {
  selection: ChartSelection;
  artwork: EvidenceArtwork;
};

export function resolveHoverPreviewContent(
  selection: ChartSelection | null | undefined,
  queries: readonly QueryDescriptor[],
  bins: readonly TimeBin[],
  preview: ArtworkPreview | null | undefined,
) {
  if (!selection) return null;
  const query = queries.find((candidate) => candidate.id === selection.queryId);
  const bin = bins.find((candidate) => candidate.key === selection.binKey);
  const artwork =
    preview?.selection.queryId === selection.queryId &&
    preview.selection.binKey === selection.binKey &&
    preview.artwork.imageUrl?.trim()
      ? preview.artwork
      : null;
  return query && bin
    ? { conceptLabel: query.label, periodLabel: bin.label, artwork }
    : null;
}

function stableIndex(seed: string, length: number) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0) % length;
}

/** Pick one stable, random-looking contributor so a hovered period never flickers. */
export function sampleHoverArtwork(
  evidence: SelectedEvidence | null | undefined,
  excludedArtworkIds: ReadonlySet<string> = new Set(),
): EvidenceArtwork | null {
  if (!evidence) return null;

  for (const sliceName of HOVER_SLICE_PRIORITY) {
    const seenClusters = new Set<string>();
    const candidates = evidence.slices[sliceName].filter((artwork) => {
      if (
        excludedArtworkIds.has(artwork.artworkId) ||
        !artwork.imageUrl?.trim() ||
        artwork.contributor === false
      ) return false;
      const clusterId = artwork.visualClusterId.trim() || artwork.artworkId;
      if (seenClusters.has(clusterId)) return false;
      seenClusters.add(clusterId);
      return true;
    });
    if (!candidates.length) continue;
    const seed = `${evidence.queryId}\u0000${evidence.binKey}\u0000${sliceName}`;
    return candidates[stableIndex(seed, candidates.length)];
  }

  return null;
}
