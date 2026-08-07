import type { ChartSelection } from "./types";

export const SEARCH_MODES = ["keyword", "embedding"] as const;

export type SearchMode = (typeof SEARCH_MODES)[number];

export const DEFAULT_SEARCH_MODE: SearchMode = "keyword";

export const SEARCH_MODE_LABELS: Record<SearchMode, string> = {
  keyword: "Keyword",
  embedding: "Embedding",
};

export function isSearchMode(value: string | null | undefined): value is SearchMode {
  return value === "keyword" || value === "embedding";
}

export function searchModeFromUrl(value: string | null | undefined): SearchMode {
  return isSearchMode(value) ? value : DEFAULT_SEARCH_MODE;
}

export function pageUrlForSearchMode(currentUrl: string, mode: SearchMode): string {
  const url = new URL(currentUrl, "http://localhost");
  if (mode === DEFAULT_SEARCH_MODE) url.searchParams.delete("searchMode");
  else url.searchParams.set("searchMode", mode);
  return `${url.pathname}${url.search}${url.hash}`;
}

export function buildSearchUrl(
  query: string,
  mode: SearchMode,
  selection?: ChartSelection,
): string {
  const params = new URLSearchParams({ q: query, searchMode: mode });
  if (selection) {
    params.set("evidenceQueryId", selection.queryId);
    params.set("evidenceBinKey", selection.binKey);
  }
  return `/api/search?${params.toString()}`;
}

export function buildBackendImageUrl(artworkId: string, mode: SearchMode): string {
  return `/api/backend-image?${new URLSearchParams({
    id: artworkId,
    searchMode: mode,
  })}`;
}

type SearchServiceEnvironment = Readonly<Record<string, string | undefined>>;

export function searchServiceEnvironmentName(mode: SearchMode): string {
  return mode === "embedding"
    ? "MNEMOSYNE_EMBEDDING_SEARCH_SERVICE_URL"
    : "MNEMOSYNE_KEYWORD_SEARCH_SERVICE_URL";
}

export function configuredSearchServiceUrl(
  mode: SearchMode,
  environment: SearchServiceEnvironment,
): string | null {
  const configured = environment[searchServiceEnvironmentName(mode)]?.trim();
  if (configured) return configured;

  // Preserve existing keyword deployments while keeping embedding opt-in and
  // impossible to route accidentally to a keyword-only service.
  if (mode === "keyword") {
    return environment.MNEMOSYNE_SEARCH_SERVICE_URL?.trim() || null;
  }
  return null;
}
