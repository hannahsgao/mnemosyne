import type { SearchMode } from "./search-mode.ts";
import type { EvidenceArtwork, SearchResponse, SelectedEvidence } from "./types.ts";

export const ANALYTICS_SCHEMA_VERSION = 1 as const;

export const ANALYTICS_EVENT_NAMES = [
  "page_view",
  "search_attempt",
  "search_result",
  "point_select",
  "evidence_result",
  "artwork_open",
  "series_activate",
  "series_toggle",
  "help_open",
  "gallery_depth",
] as const;

export type AnalyticsEventName = (typeof ANALYTICS_EVENT_NAMES)[number];
export type SearchScopedAnalyticsEventName = Exclude<AnalyticsEventName, "page_view">;
export type AnalyticsSearchSource =
  | "initial_default"
  | "url_restore"
  | "form"
  | "example"
  | "mode_switch";
export type AnalyticsInputMethod = "pointer" | "keyboard";
export type AnalyticsPlacement = "evidence" | "nearest";
export type AnalyticsEvidenceSource =
  | "point"
  | "auto_peak"
  | "url_restore"
  | "series_activate"
  | "series_replacement";

export type AnalyticsEventProperties = {
  page_view: {
    search_mode: SearchMode;
    source: "default" | "shared_query" | "shared_selection";
    referrer_kind: "direct" | "internal" | "search" | "social" | "other";
    viewport_bucket: "small" | "medium" | "large";
    query_count: number;
  };
  search_attempt: {
    search_mode: SearchMode;
    source: AnalyticsSearchSource;
    outcome: "accepted" | "invalid";
    query_count: number;
    query_length_bucket: string;
    example_index?: number;
    error_code?: string;
  };
  search_result: {
    search_mode: SearchMode;
    source: AnalyticsSearchSource;
    outcome: "timeline" | "empty" | "nearest_only" | "error" | "aborted";
    duration_ms: number;
    query_count: number;
    series_count?: number;
    matched_series_count?: number;
    bin_count?: number;
    result_count?: number;
    nearest_count?: number;
    status_code?: number;
    error_code?: string;
    transport?: "direct" | "proxy";
    cache_status?: string;
    corpus_id?: string;
    corpus_version?: string;
    model_id?: string;
    model_version?: string;
    metric_id?: string;
  };
  point_select: {
    search_mode: SearchMode;
    input_method: AnalyticsInputMethod;
    bin_key: string;
    series_index: number;
  };
  evidence_result: {
    search_mode: SearchMode;
    source: AnalyticsEvidenceSource;
    outcome: "success" | "empty" | "error" | "aborted";
    duration_ms: number;
    bin_key: string;
    series_index: number;
    cached: boolean;
    result_count?: number;
    status_code?: number;
    error_code?: string;
  };
  artwork_open: {
    search_mode: SearchMode;
    placement: AnalyticsPlacement;
    input_method: AnalyticsInputMethod;
    institution: "met" | "nga" | "aic" | "cma" | "smk" | "other";
    rank: number;
    series_index: number;
    bin_key?: string;
    has_image: boolean;
    contributor: boolean;
  };
  series_activate: {
    search_mode: SearchMode;
    source: "legend" | "endpoint";
    input_method: AnalyticsInputMethod;
    series_index: number;
  };
  series_toggle: {
    search_mode: SearchMode;
    action: "show" | "hide";
    input_method: AnalyticsInputMethod;
    series_index: number;
    visible_count: number;
  };
  help_open: {
    search_mode: SearchMode;
    target: "calculation" | "chart_reading";
  };
  gallery_depth: {
    search_mode: SearchMode;
    placement: AnalyticsPlacement;
    visible_count: number;
    series_index: number;
    bin_key?: string;
  };
};

export type AnalyticsEvent<K extends AnalyticsEventName = AnalyticsEventName> = {
  id: string;
  sequence: number;
  name: K;
  properties: AnalyticsEventProperties[K];
} & (K extends "page_view" ? { searchId?: never } : { searchId: string });

export type AnalyticsEnvelope = {
  schemaVersion: typeof ANALYTICS_SCHEMA_VERSION;
  sessionId: string;
  viewId: string;
  events: AnalyticsEvent[];
};

export function analyticsQueryLengthBucket(length: number) {
  if (length <= 0) return "0";
  if (length <= 25) return "1-25";
  if (length <= 75) return "26-75";
  if (length <= 150) return "76-150";
  if (length <= 300) return "151-300";
  if (length <= 500) return "301-500";
  return "501+";
}

export function analyticsCacheStatus(value: string | null) {
  const normalized = value?.normalize("NFKC").trim().toLowerCase();
  return normalized && /^[a-z0-9._+-]{1,40}$/.test(normalized)
    ? normalized
    : undefined;
}

export function analyticsSafeToken(value: string) {
  const normalized = value.normalize("NFKC").trim();
  return normalized && /^[\p{L}\p{N}._:/+@-]{1,100}$/u.test(normalized)
    ? normalized
    : undefined;
}

export function analyticsDuration(startedAt: number, endedAt: number) {
  if (!Number.isFinite(startedAt) || !Number.isFinite(endedAt)) return 0;
  return Math.max(0, Math.min(86_400_000, Math.round(endedAt - startedAt)));
}

export function analyticsSeriesIndex(
  response: Pick<SearchResponse, "queries"> | null,
  queryId: string,
) {
  const index = response?.queries.findIndex((query) => query.id === queryId) ?? -1;
  return Math.max(0, Math.min(4, index));
}

export function analyticsEvidenceCount(evidence: SelectedEvidence | null) {
  if (!evidence) return 0;
  const source = evidence.slices.strongest.length
    ? evidence.slices.strongest
    : evidence.slices.randomContributors;
  return new Set(source.map((artwork) => artwork.artworkId)).size;
}

export function analyticsInstitution(
  value: EvidenceArtwork["institution"],
): AnalyticsEventProperties["artwork_open"]["institution"] {
  const normalized = value.normalize("NFKC").trim().toLowerCase();
  if (normalized === "met" || normalized.includes("metropolitan museum")) return "met";
  if (normalized === "nga" || normalized.includes("national gallery of art")) return "nga";
  if (normalized === "aic" || normalized.includes("art institute of chicago")) return "aic";
  if (normalized === "cma" || normalized.includes("cleveland museum")) return "cma";
  if (normalized === "smk" || normalized.includes("statens museum")) return "smk";
  return "other";
}

export function analyticsSearchSummary(
  response: SearchResponse,
): Pick<
  AnalyticsEventProperties["search_result"],
  | "outcome"
  | "series_count"
  | "matched_series_count"
  | "bin_count"
  | "result_count"
  | "nearest_count"
  | "corpus_id"
  | "corpus_version"
  | "model_id"
  | "model_version"
  | "metric_id"
> {
  const matchedSeries = response.series.filter((series) => series.k > 0);
  const hasTimeline = response.series.some((series) => series.points.length > 0);
  const resultCount = response.series.reduce((total, series) => total + Math.max(0, series.k), 0);
  const nearestCount = response.series.reduce(
    (total, series) => total + (series.nearestMatches?.length ?? 0),
    0,
  );
  const outcome = hasTimeline
    ? "timeline"
    : nearestCount > 0
      ? "nearest_only"
      : "empty";
  const corpusId = analyticsSafeToken(response.corpus.id);
  const corpusVersion = analyticsSafeToken(response.corpus.version);
  const modelId = analyticsSafeToken(response.model.id);
  const modelVersion = analyticsSafeToken(response.model.version);
  const metricId = analyticsSafeToken(response.metric.id);
  return {
    outcome,
    series_count: response.series.length,
    matched_series_count: matchedSeries.length,
    bin_count: response.bins.length,
    result_count: Math.min(10_000_000, resultCount),
    nearest_count: Math.min(10_000, nearestCount),
    ...(corpusId ? { corpus_id: corpusId } : {}),
    ...(corpusVersion ? { corpus_version: corpusVersion } : {}),
    ...(modelId ? { model_id: modelId } : {}),
    ...(modelVersion ? { model_version: modelVersion } : {}),
    ...(metricId ? { metric_id: metricId } : {}),
  };
}

export function analyticsFailureCode(error: unknown, statusCode?: number) {
  if (error instanceof Error && /unsupported response/i.test(error.message)) {
    return "invalid_response";
  }
  if (statusCode !== undefined) return `http_${statusCode}`;
  if (error instanceof TypeError) return "network";
  return "service_error";
}

export function analyticsReferrerKind(referrer: string, currentHost: string) {
  if (!referrer) return "direct" as const;
  try {
    const hostname = new URL(referrer).hostname.toLowerCase();
    if (hostname === currentHost.toLowerCase()) return "internal" as const;
    if (/^(?:www\.)?(?:google|bing|duckduckgo|yahoo)\./.test(hostname)) return "search" as const;
    if (/(?:facebook|instagram|linkedin|pinterest|reddit|tiktok|twitter|x)\./.test(hostname)) {
      return "social" as const;
    }
  } catch {
    // Treat malformed or non-HTTP referrers as an unclassified external source.
  }
  return "other" as const;
}

export function analyticsViewportBucket(width: number) {
  if (width < 720) return "small" as const;
  if (width < 1200) return "medium" as const;
  return "large" as const;
}
