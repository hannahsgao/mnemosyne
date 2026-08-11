import type { ChartSelection } from "./types";

export const SEARCH_MODES = ["embedding", "keyword"] as const;

export type SearchMode = (typeof SEARCH_MODES)[number];

export const DEFAULT_SEARCH_MODE: SearchMode = "embedding";

export const SEARCH_MODE_LABELS: Record<SearchMode, string> = {
  keyword: "Metadata keywords",
  embedding: "Visual concepts",
};

export type SearchPageState = {
  query: string;
  mode: SearchMode;
  selection: ChartSelection | null;
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

/** Read the shareable search state, falling back cleanly for old or partial URLs. */
export function searchPageStateFromUrl(
  currentUrl: string,
  fallbackQuery: string,
): SearchPageState {
  const url = new URL(currentUrl, "http://localhost");
  const query = url.searchParams.get("q")?.trim() || fallbackQuery;
  const queryId = url.searchParams.get("concept")?.trim();
  const binKey = url.searchParams.get("period")?.trim();
  return {
    query,
    mode: searchModeFromUrl(url.searchParams.get("searchMode")),
    selection: queryId && binKey ? { queryId, binKey } : null,
  };
}

/** Preserve unrelated parameters while keeping the complete explorer state shareable. */
export function pageUrlForSearchState(
  currentUrl: string,
  state: SearchPageState,
): string {
  const url = new URL(currentUrl, "http://localhost");
  const query = state.query.trim();
  if (query) url.searchParams.set("q", query);
  else url.searchParams.delete("q");
  if (state.mode === DEFAULT_SEARCH_MODE) url.searchParams.delete("searchMode");
  else url.searchParams.set("searchMode", state.mode);
  if (state.selection) {
    url.searchParams.set("concept", state.selection.queryId);
    url.searchParams.set("period", state.selection.binKey);
  } else {
    url.searchParams.delete("concept");
    url.searchParams.delete("period");
  }
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

export function buildEvidenceUrl(
  query: string,
  mode: SearchMode,
  selection: ChartSelection,
): string {
  const params = new URLSearchParams({
    q: query,
    searchMode: mode,
    evidenceQueryId: selection.queryId,
    evidenceBinKey: selection.binKey,
  });
  return `/api/evidence?${params.toString()}`;
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
