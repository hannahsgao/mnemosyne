"use client";

import {
  type Dispatch,
  type FormEvent,
  type SetStateAction,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Timeline,
  type TimelineHoverPreview,
  type TimelineInputMethod,
  type TimelineSeriesInteraction,
} from "../components/Timeline";
import {
  createAnalyticsId,
  trackPageAnalyticsOnce,
  trackSearchAnalytics,
  trackSearchAnalyticsOnce,
} from "../lib/analytics-client";
import {
  analyticsCacheStatus,
  analyticsDuration,
  analyticsEvidenceCount,
  analyticsFailureCode,
  analyticsInstitution,
  analyticsQueryLengthBucket,
  analyticsReferrerKind,
  analyticsSearchSummary,
  analyticsSeriesIndex,
  analyticsViewportBucket,
  type AnalyticsEvidenceSource,
  type AnalyticsEventProperties,
  type AnalyticsPlacement,
  type AnalyticsSearchSource,
} from "../lib/analytics-events";
import {
  evidencePreviewLabel,
  nearestMatchGroups,
  selectedEvidenceItems,
} from "../lib/evidence";
import {
  evidenceMatchesSelection,
  invalidSearchStatus,
  invalidateExplorerRequests,
  prepareEvidenceRequest,
  searchErrorPlacement,
} from "../lib/explorer-state";
import { hoverPreviewImageUrl, sampleHoverArtwork } from "../lib/hover-preview";
import { MAX_QUERY_LENGTH, parseConceptQuery, QuerySyntaxError } from "../lib/query";
import { requestKeywordEvidence, requestKeywordSearch } from "../lib/keyword-transport";
import { requestVisualEvidence, requestVisualSearch } from "../lib/visual-transport";
import {
  DEFAULT_SEARCH_MODE,
  pageUrlForSearchState,
  searchPageStateFromUrl,
  type SearchMode,
} from "../lib/search-mode";
import {
  describeTimelineMetric,
  formatTimelineYear,
  peakSelection,
  pointForBin,
  timelineWindow,
} from "../lib/timeline";
import type {
  ChartSelection,
  CorpusMetadata,
  EvidenceArtwork,
  MetricMetadata,
  SearchResponse,
  SelectedEvidence,
} from "../lib/types";

const INITIAL_QUERY = "manuscript page, newspaper, comic strip";
const EXAMPLE_QUERIES = [
  "mirror, portrait, self-portrait",
  "clock, chair, table, lamp",
  "crown, bonnet, top hat, bowler hat",
  "manuscript page, newspaper, comic strip",
  "crucifixion, public execution",
  "sailing ship, steamship",
  "palace interior, church interior, domestic interior, factory interior",
  "Last Supper, banquet, tea party, café",
  "powdered wig, bonnet, crinoline dress, flapper dress",
];
const METADATA_EXAMPLE_QUERIES = [
  "manuscript, printed book, newspaper",
  "bronze, marble, porcelain, plastic",
  "carriage, automobile, airplane",
  "venice, paris, new york, los angeles",
  "chariot, carriage, train, airplane",
];
const INITIAL_VISIBLE_WORKS = 5;
const VISIBLE_WORK_BATCH = 5;
const HOVER_PREVIEW_DELAY_MS = 50;
const HOVER_PREVIEW_CACHE_LIMIT = 80;
const preloadedHoverImageUrls = new Set<string>();
const pendingHoverImagePreloads = new Map<string, HTMLImageElement>();

function preloadHoverArtwork(artwork: EvidenceArtwork | null) {
  const imageUrl = hoverPreviewImageUrl(artwork?.imageUrl);
  if (!imageUrl || typeof Image === "undefined" || preloadedHoverImageUrls.has(imageUrl)) {
    return;
  }

  const image = new Image();
  preloadedHoverImageUrls.add(imageUrl);
  pendingHoverImagePreloads.set(imageUrl, image);
  while (preloadedHoverImageUrls.size > HOVER_PREVIEW_CACHE_LIMIT) {
    const oldest = preloadedHoverImageUrls.values().next().value;
    if (oldest === undefined) break;
    preloadedHoverImageUrls.delete(oldest);
  }

  image.decoding = "async";
  image.fetchPriority = "low";
  image.onload = () => pendingHoverImagePreloads.delete(imageUrl);
  image.onerror = () => {
    pendingHoverImagePreloads.delete(imageUrl);
    preloadedHoverImageUrls.delete(imageUrl);
  };
  image.src = imageUrl;
}

function hoverPreviewCacheKey(
  query: string,
  mode: SearchMode,
  selection: ChartSelection,
) {
  return `${mode}\u0000${query}\u0000${selection.queryId}\u0000${selection.binKey}`;
}

function writeHoverPreviewCache(
  cache: Map<string, EvidenceArtwork | null>,
  key: string,
  artwork: EvidenceArtwork | null,
) {
  cache.delete(key);
  cache.set(key, artwork);
  while (cache.size > HOVER_PREVIEW_CACHE_LIMIT) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}

function cacheSelectedEvidencePreview(
  cache: Map<string, EvidenceArtwork | null>,
  failedArtworkIds: Map<string, Set<string>>,
  query: string,
  mode: SearchMode,
  evidence: SelectedEvidence | null,
) {
  if (!evidence) return;
  const selection = { queryId: evidence.queryId, binKey: evidence.binKey };
  const key = hoverPreviewCacheKey(query, mode, selection);
  if (!cache.has(key)) {
    const artwork = sampleHoverArtwork(evidence, failedArtworkIds.get(key));
    writeHoverPreviewCache(
      cache,
      key,
      artwork,
    );
    preloadHoverArtwork(artwork);
  }
}

const SEARCH_INPUT_LABELS: Record<SearchMode, string> = {
  embedding: "Search artworks by visual content",
  keyword: "Search artwork catalogue metadata",
};

const SEARCH_PLACEHOLDERS: Record<SearchMode, string> = {
  embedding: "mirror, portrait, self-portrait",
  keyword: "carriage, automobile, airplane",
};

const SEARCH_MODE_HELP: Record<SearchMode, string> = {
  embedding: "Describe what you want to see.",
  keyword: "Search words in the catalogue record.",
};

const CHART_HELP: Record<SearchMode, string> = {
  embedding: "Higher means visual matches are more concentrated in that period. Gaps mean limited evidence.",
  keyword: "Higher means more dated records match in that period. Gaps mean limited evidence.",
};
type SearchOptions = {
  syncInput?: boolean;
  requestedSelection?: ChartSelection | null;
  analyticsSource?: AnalyticsSearchSource;
  exampleIndex?: number;
};

type EvidenceEnvelope = {
  schemaVersion: "mnemosyne.evidence.v1";
  selectedEvidence: SelectedEvidence | null;
  generatedAt: string;
};

type EvidenceContext = {
  baseResult?: SearchResponse;
  query?: string;
  mode?: SearchMode;
  analyticsSource?: AnalyticsEvidenceSource;
  searchId?: string;
};

type ArtworkAnalyticsContext = {
  placement: AnalyticsPlacement;
  searchMode: SearchMode;
  searchId?: string;
  seriesIndex: number;
  binKey?: string;
  rank: number;
};

function isSearchResponse(value: unknown): value is SearchResponse {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<SearchResponse>;
  return (
    candidate.schemaVersion === "mnemosyne.search.v1" &&
    Array.isArray(candidate.queries) &&
    Array.isArray(candidate.bins) &&
    Array.isArray(candidate.series) &&
    Boolean(candidate.corpus) &&
    Boolean(candidate.metric)
  );
}

function isEvidenceEnvelope(value: unknown): value is EvidenceEnvelope {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<EvidenceEnvelope>;
  return (
    candidate.schemaVersion === "mnemosyne.evidence.v1" &&
    (candidate.selectedEvidence === null || typeof candidate.selectedEvidence === "object")
  );
}

function errorMessage(payload: unknown, fallback: string) {
  return payload && typeof payload === "object" && "error" in payload
    ? String(payload.error)
    : fallback;
}

function ChartCalculationTooltip({
  metric,
  onOpen,
}: {
  metric: MetricMetadata;
  onOpen?: () => void;
}) {
  const tooltipId = "chart-calculation-tooltip";
  return (
    <span className="chart-info">
      <button
        className="chart-info-trigger"
        type="button"
        aria-label="How this graph is calculated"
        aria-describedby={tooltipId}
        onPointerEnter={onOpen}
        onFocus={onOpen}
        onClick={onOpen}
      >
        <svg viewBox="0 0 18 18" aria-hidden="true">
          <circle cx="9" cy="9" r="7" />
          <path d="M9 8v4.25" />
          <circle className="chart-info-dot" cx="9" cy="5.4" r=".8" />
        </svg>
      </button>
      <span className="chart-info-tooltip" id={tooltipId} role="tooltip">
        <strong>How this graph is calculated</strong>
        <span>{describeTimelineMetric(metric)}</span>
      </span>
    </span>
  );
}
function institutionLabel(value: string) {
  const normalized = value.trim().toLowerCase();
  if (normalized === "met" || normalized === "the met") return "The Met";
  if (normalized === "nga" || normalized === "national gallery of art") {
    return "National Gallery of Art";
  }
  if (normalized === "aic" || normalized === "art institute of chicago") {
    return "Art Institute of Chicago";
  }
  if (normalized === "cma" || normalized === "cleveland museum of art") {
    return "Cleveland Museum of Art";
  }
  return value || "Museum source unavailable";
}

function itemNoun(count: number, countingUnit: CorpusMetadata["countingUnit"]) {
  if (countingUnit === "catalog-record") {
    return `catalog record${count === 1 ? "" : "s"}`;
  }
  return `work${count === 1 ? "" : "s"}`;
}

function corpusSummary(corpus: CorpusMetadata) {
  const label = corpus.label === corpus.id
    ? "Open-access museum image catalog"
    : corpus.label;
  return corpus.count === null
    ? label
    : `${corpus.count.toLocaleString()} ${itemNoun(corpus.count, corpus.countingUnit)} · ${label}`;
}

function ArtworkCard({
  artwork,
  analytics,
}: {
  artwork: EvidenceArtwork;
  analytics: ArtworkAnalyticsContext;
}) {
  const institution = institutionLabel(artwork.institution);
  const metadata = artwork.dateDisplay
    ? `${institution} · ${artwork.dateDisplay}`
    : institution;
  const trackOpen = (inputMethod: TimelineInputMethod) => trackSearchAnalytics(
    "artwork_open",
    {
      search_mode: analytics.searchMode,
      placement: analytics.placement,
      input_method: inputMethod,
      institution: analyticsInstitution(artwork.institution),
      rank: analytics.rank,
      series_index: analytics.seriesIndex,
      ...(analytics.binKey ? { bin_key: analytics.binKey } : {}),
      has_image: Boolean(artwork.imageUrl),
      contributor: artwork.contributor,
    },
    analytics.searchId,
  );

  return (
    <a
      className="artwork-card"
      href={artwork.sourceRecordUrl}
      target="_blank"
      rel="noreferrer"
      onClick={(event) => trackOpen(event.detail === 0 ? "keyboard" : "pointer")}
      onAuxClick={(event) => {
        if (event.button === 1) trackOpen("pointer");
      }}
    >
      <div className="artwork-image-wrap">
        {artwork.imageUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            className="artwork-image"
            src={artwork.imageUrl}
            alt=""
            loading="lazy"
            decoding="async"
          />
        ) : (
          <div className="image-placeholder">No image</div>
        )}
      </div>
      <div className="artwork-copy">
        <strong title={artwork.title || "Untitled"}>{artwork.title || "Untitled"}</strong>
        <span className="artwork-artist" title={artwork.artist || "Unknown artist"}>
          {artwork.artist || "Unknown artist"}
        </span>
        <small className="artwork-meta" title={metadata}>
          <span className="artwork-meta-text">{metadata}</span>
          <span className="artwork-external" aria-hidden="true">↗</span>
        </small>
        <span className="sr-only">Opens the museum record in a new tab.</span>
      </div>
    </a>
  );
}

function ArtworkCardSkeleton() {
  return (
    <div className="artwork-card artwork-card-loading" aria-hidden="true">
      <div className="artwork-image-wrap" />
      <div className="artwork-copy">
        <span className="artwork-loading-line artwork-loading-title" />
        <span className="artwork-loading-line artwork-loading-artist" />
        <span className="artwork-loading-line artwork-loading-meta" />
      </div>
    </div>
  );
}

function ProgressiveArtworkGrid({
  artworks,
  emptyMessage,
  isLoading,
  onVisibleCountChange,
  onDepthReached,
  analytics,
  showEmpty,
  visibleCount,
}: {
  artworks: EvidenceArtwork[];
  emptyMessage: string;
  isLoading: boolean;
  onVisibleCountChange: Dispatch<SetStateAction<number>>;
  onDepthReached?: (visibleCount: number) => void;
  analytics: Omit<ArtworkAnalyticsContext, "rank">;
  showEmpty: boolean;
  visibleCount: number;
}) {
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const visibleArtworks = artworks.slice(0, visibleCount);
  const hasMore = visibleArtworks.length < artworks.length;

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (isLoading || !sentinel || !hasMore) return;

    if (!("IntersectionObserver" in window)) {
      onVisibleCountChange(artworks.length);
      return;
    }

    const observer = new IntersectionObserver(([entry]) => {
      if (!entry?.isIntersecting) return;
      observer.disconnect();
      onVisibleCountChange((current) => Math.min(
        artworks.length,
        current + VISIBLE_WORK_BATCH,
      ));
    }, { rootMargin: "0px 0px 240px" });

    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [artworks.length, hasMore, isLoading, onVisibleCountChange, visibleCount]);

  useEffect(() => {
    if (isLoading || visibleCount <= INITIAL_VISIBLE_WORKS) return;
    onDepthReached?.(Math.min(visibleCount, artworks.length));
  }, [artworks.length, isLoading, onDepthReached, visibleCount]);

  return (
    <>
      <div className="artwork-grid" aria-busy={isLoading}>
        {isLoading && Array.from({ length: INITIAL_VISIBLE_WORKS }, (_, index) => (
          <ArtworkCardSkeleton key={index} />
        ))}
        {!isLoading && visibleArtworks.map((artwork, index) => (
          <ArtworkCard
            key={artwork.artworkId}
            artwork={artwork}
            analytics={{ ...analytics, rank: index + 1 }}
          />
        ))}
        {!isLoading && showEmpty && <p className="no-works">{emptyMessage}</p>}
      </div>
      {!isLoading && artworks.length > INITIAL_VISIBLE_WORKS && (
        <span className="sr-only" role="status" aria-atomic="true">
          Showing {visibleArtworks.length} of {artworks.length} artworks.
        </span>
      )}
      {!isLoading && hasMore && (
        <div className="artwork-load-sentinel" ref={sentinelRef} aria-hidden="true" />
      )}
    </>
  );
}
function selectionFromRequestedState(
  response: SearchResponse,
  requested: ChartSelection | null | undefined,
) {
  if (!requested) return null;
  const series = response.series.find((item) => item.queryId === requested.queryId);
  if (!series?.points.some((point) => point.binKey === requested.binKey)) return null;
  if (response.bins.find((bin) => bin.key === requested.binKey)?.belowMinimumDenominator === true) {
    return null;
  }
  return requested;
}

export default function Home() {
  const [input, setInput] = useState(INITIAL_QUERY);
  const [submittedQuery, setSubmittedQuery] = useState(INITIAL_QUERY);
  const [searchMode, setSearchMode] = useState<SearchMode>(DEFAULT_SEARCH_MODE);
  const [submittedSearchMode, setSubmittedSearchMode] = useState<SearchMode>(DEFAULT_SEARCH_MODE);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [selection, setSelection] = useState<ChartSelection | null>(null);
  const [hiddenQueryIds, setHiddenQueryIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [visibleEvidenceCount, setVisibleEvidenceCount] = useState(INITIAL_VISIBLE_WORKS);
  const [hoverPreview, setHoverPreview] = useState<TimelineHoverPreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const requestId = useRef(0);
  const evidenceRequestId = useRef(0);
  const hoverPreviewRequestId = useRef(0);
  const searchAbort = useRef<AbortController | null>(null);
  const evidenceAbort = useRef<AbortController | null>(null);
  const hoverPreviewAbort = useRef<AbortController | null>(null);
  const hoverPreviewTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hoverPreviewCache = useRef(new Map<string, EvidenceArtwork | null>());
  const failedHoverArtworkIds = useRef(new Map<string, Set<string>>());
  const resultSearchId = useRef<string | undefined>(undefined);

  function resetHoverPreview(clearCache = false) {
    hoverPreviewRequestId.current += 1;
    if (hoverPreviewTimer.current !== null) {
      clearTimeout(hoverPreviewTimer.current);
      hoverPreviewTimer.current = null;
    }
    hoverPreviewAbort.current?.abort();
    hoverPreviewAbort.current = null;
    setHoverPreview(null);
    if (clearCache) {
      hoverPreviewCache.current.clear();
      failedHoverArtworkIds.current.clear();
    }
  }

  const handleHoverSelection = useCallback((nextSelection: ChartSelection | null) => {
    const currentRequest = ++hoverPreviewRequestId.current;
    if (hoverPreviewTimer.current !== null) {
      clearTimeout(hoverPreviewTimer.current);
      hoverPreviewTimer.current = null;
    }
    hoverPreviewAbort.current?.abort();
    hoverPreviewAbort.current = null;
    setHoverPreview(null);
    if (!nextSelection) return;

    const key = hoverPreviewCacheKey(submittedQuery, submittedSearchMode, nextSelection);
    const cache = hoverPreviewCache.current;
    if (cache.has(key)) {
      const artwork = cache.get(key) ?? null;
      cache.delete(key);
      cache.set(key, artwork);
      if (artwork) setHoverPreview({ selection: nextSelection, artwork });
      return;
    }

    hoverPreviewTimer.current = setTimeout(() => {
      hoverPreviewTimer.current = null;
      const controller = new AbortController();
      hoverPreviewAbort.current = controller;
      void (async () => {
        try {
          const { response, payload } = submittedSearchMode === "keyword"
            ? await requestKeywordEvidence(submittedQuery, nextSelection, {
                signal: controller.signal,
              })
            : await requestVisualEvidence(submittedQuery, nextSelection, {
                signal: controller.signal,
              });
          if (!response.ok || !isEvidenceEnvelope(payload)) return;
          if (hoverPreviewRequestId.current !== currentRequest) return;
          const evidence = payload.selectedEvidence;
          const artwork = evidenceMatchesSelection(evidence, nextSelection)
            ? sampleHoverArtwork(evidence, failedHoverArtworkIds.current.get(key))
            : null;
          writeHoverPreviewCache(cache, key, artwork);
          if (artwork) setHoverPreview({ selection: nextSelection, artwork });
        } catch {
          // Image failures are decorative; the chart caption remains available.
        } finally {
          if (hoverPreviewAbort.current === controller) hoverPreviewAbort.current = null;
        }
      })();
    }, HOVER_PREVIEW_DELAY_MS);
  }, [submittedQuery, submittedSearchMode]);

  const handleHoverPreviewError = useCallback((failedPreview: TimelineHoverPreview) => {
    const key = hoverPreviewCacheKey(
      submittedQuery,
      submittedSearchMode,
      failedPreview.selection,
    );
    const failures = failedHoverArtworkIds.current;
    const artworkIds = failures.get(key) ?? new Set<string>();
    artworkIds.add(failedPreview.artwork.artworkId);
    failures.delete(key);
    failures.set(key, artworkIds);
    while (failures.size > HOVER_PREVIEW_CACHE_LIMIT) {
      const oldest = failures.keys().next().value;
      if (oldest === undefined) break;
      failures.delete(oldest);
    }
    hoverPreviewCache.current.delete(key);
    setHoverPreview((current) =>
      current?.selection.queryId === failedPreview.selection.queryId &&
      current.selection.binKey === failedPreview.selection.binKey &&
      current.artwork.artworkId === failedPreview.artwork.artworkId
        ? null
        : current,
    );
  }, [submittedQuery, submittedSearchMode]);

  function replacePageState(query: string, mode: SearchMode, nextSelection: ChartSelection | null) {
    const nextUrl = pageUrlForSearchState(window.location.href, {
      query,
      mode,
      selection: nextSelection,
    });
    window.history.replaceState(window.history.state, "", nextUrl);
  }

  async function search(
    nextQuery: string,
    nextMode: SearchMode = searchMode,
    options: SearchOptions = {},
  ) {
    const analyticsSource = options.analyticsSource ?? "form";
    const searchId = createAnalyticsId(window.crypto);
    const startedAt = performance.now();
    const invalidated = invalidateExplorerRequests(
      requestId.current,
      evidenceRequestId.current,
      searchAbort.current,
      evidenceAbort.current,
    );
    requestId.current = invalidated.searchRequestId;
    evidenceRequestId.current = invalidated.evidenceRequestId;
    searchAbort.current = null;
    evidenceAbort.current = null;
    const currentRequest = invalidated.searchRequestId;

    let parsedQuery: ReturnType<typeof parseConceptQuery>;
    try {
      parsedQuery = parseConceptQuery(nextQuery);
    } catch (caught) {
      trackSearchAnalytics("search_attempt", {
        search_mode: nextMode,
        source: analyticsSource,
        outcome: "invalid",
        query_count: 0,
        query_length_bucket: analyticsQueryLengthBucket(nextQuery.length),
        ...(options.exampleIndex === undefined ? {} : { example_index: options.exampleIndex }),
        error_code: caught instanceof QuerySyntaxError ? caught.code : "invalid_query",
      }, searchId);
      const status = invalidSearchStatus(
        caught instanceof QuerySyntaxError ? caught.message : "Check the query and try again.",
      );
      setError(status.error);
      setLoading(status.loading);
      setEvidenceLoading(status.evidenceLoading);
      setEvidenceError(null);
      return;
    }

    const trimmedQuery = nextQuery.trim();
    trackSearchAnalytics("search_attempt", {
      search_mode: nextMode,
      source: analyticsSource,
      outcome: "accepted",
      query_count: parsedQuery.length,
      query_length_bucket: analyticsQueryLengthBucket(nextQuery.length),
      ...(options.exampleIndex === undefined ? {} : { example_index: options.exampleIndex }),
    }, searchId);
    resetHoverPreview(true);
    const controller = new AbortController();
    searchAbort.current = controller;
    replacePageState(trimmedQuery, nextMode, null);
    setSearchMode(nextMode);
    setSubmittedSearchMode(nextMode);
    if (options.syncInput !== false) setInput(trimmedQuery);
    setSubmittedQuery(trimmedQuery);
    setLoading(true);
    resultSearchId.current = undefined;
    setResult(null);
    setEvidenceLoading(false);
    setError(null);
    setEvidenceError(null);
    setSelection(null);
    setVisibleEvidenceCount(INITIAL_VISIBLE_WORKS);
    setHiddenQueryIds(new Set());

    let analyticsFinished = false;
    let statusCode: number | undefined;
    let transport: "direct" | "proxy" | undefined;
    let cacheStatus: string | undefined;
    const finishSearchAnalytics = (
      properties: Omit<
        AnalyticsEventProperties["search_result"],
        "search_mode" | "source" | "duration_ms" | "query_count"
      >,
    ) => {
      if (analyticsFinished) return;
      analyticsFinished = true;
      trackSearchAnalytics("search_result", {
        search_mode: nextMode,
        source: analyticsSource,
        duration_ms: analyticsDuration(startedAt, performance.now()),
        query_count: parsedQuery.length,
        ...properties,
      }, searchId);
    };

    try {
      let payload: SearchResponse;

      if (nextMode === "keyword") {
        const { response, payload: body, via } = await requestKeywordSearch(trimmedQuery, {
          signal: controller.signal,
        });
        statusCode = response.status;
        transport = via;
        cacheStatus = analyticsCacheStatus(response.headers.get("CF-Cache-Status"));
        if (!response.ok) throw new Error(errorMessage(body, "Search failed."));
        if (!isSearchResponse(body)) throw new Error("The search service returned an unsupported response.");
        payload = body;
      } else {
        const { response, payload: body, via } = await requestVisualSearch(trimmedQuery, {
          signal: controller.signal,
        });
        statusCode = response.status;
        transport = via;
        cacheStatus = analyticsCacheStatus(response.headers.get("CF-Cache-Status"));
        if (!response.ok) {
          throw new Error(errorMessage(
            body,
            "Visual search is unavailable.",
          ));
        }
        if (!isSearchResponse(body)) {
          throw new Error("Visual search returned an unsupported response.");
        }
        payload = body;
      }

      if (requestId.current !== currentRequest) {
        finishSearchAnalytics({
          outcome: "aborted",
          ...(statusCode ? { status_code: statusCode } : {}),
          ...(transport ? { transport } : {}),
          ...(cacheStatus ? { cache_status: cacheStatus } : {}),
        });
        return;
      }
      finishSearchAnalytics({
        ...analyticsSearchSummary(payload),
        ...(statusCode ? { status_code: statusCode } : {}),
        ...(transport ? { transport } : {}),
        ...(cacheStatus ? { cache_status: cacheStatus } : {}),
      });
      const requestedSelection = selectionFromRequestedState(
        payload,
        options.requestedSelection,
      );
      const nextSelection = requestedSelection ?? (
        payload.selectedEvidence
          ? { queryId: payload.selectedEvidence.queryId, binKey: payload.selectedEvidence.binKey }
          : peakSelection(payload)
      );
      cacheSelectedEvidencePreview(
        hoverPreviewCache.current,
        failedHoverArtworkIds.current,
        trimmedQuery,
        nextMode,
        payload.selectedEvidence,
      );
      resultSearchId.current = searchId;
      setResult(payload);
      setSelection(nextSelection);
      replacePageState(trimmedQuery, nextMode, nextSelection);
      if (nextSelection && !evidenceMatchesSelection(payload.selectedEvidence, nextSelection)) {
        void Promise.resolve().then(() => {
          if (requestId.current !== currentRequest) return;
          void loadEvidence(nextSelection, {
            baseResult: payload,
            query: trimmedQuery,
            mode: nextMode,
            analyticsSource: options.requestedSelection ? "url_restore" : "auto_peak",
            searchId,
          });
        });
      } else if (nextSelection && payload.selectedEvidence) {
        const resultCount = analyticsEvidenceCount(payload.selectedEvidence);
        trackSearchAnalytics("evidence_result", {
          search_mode: nextMode,
          source: options.requestedSelection ? "url_restore" : "auto_peak",
          outcome: resultCount > 0 ? "success" : "empty",
          duration_ms: 0,
          bin_key: nextSelection.binKey,
          series_index: analyticsSeriesIndex(payload, nextSelection.queryId),
          cached: true,
          result_count: resultCount,
        }, searchId);
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") {
        finishSearchAnalytics({
          outcome: "aborted",
          ...(transport ? { transport } : {}),
        });
        return;
      }
      finishSearchAnalytics({
        outcome: "error",
        ...(statusCode ? { status_code: statusCode } : {}),
        error_code: analyticsFailureCode(caught, statusCode),
        ...(transport ? { transport } : {}),
        ...(cacheStatus ? { cache_status: cacheStatus } : {}),
      });
      if (requestId.current !== currentRequest) return;
      const message = caught instanceof Error ? caught.message : "Search failed.";
      setError(nextMode === "embedding" && caught instanceof TypeError
        ? "Visual search is unavailable."
        : message);
      setResult(null);
      setSelection(null);
    } finally {
      if (requestId.current === currentRequest) setLoading(false);
    }
  }

  async function loadEvidence(nextSelection: ChartSelection, context: EvidenceContext = {}) {
    if (selection?.queryId !== nextSelection.queryId || selection?.binKey !== nextSelection.binKey) {
      setVisibleEvidenceCount(INITIAL_VISIBLE_WORKS);
    }
    setSelection(nextSelection);
    const activeQuery = context.query ?? submittedQuery;
    const activeMode = context.mode ?? submittedSearchMode;
    const analyticsSource = context.analyticsSource ?? "point";
    const searchId = context.searchId ?? resultSearchId.current;
    const startedAt = performance.now();
    replacePageState(activeQuery, activeMode, nextSelection);
    const activeResult = context.baseResult ?? result;
    const seriesIndex = analyticsSeriesIndex(activeResult, nextSelection.queryId);
    const cached = evidenceMatchesSelection(activeResult?.selectedEvidence, nextSelection);
    const prepared = prepareEvidenceRequest(
      evidenceRequestId.current,
      evidenceAbort.current,
      cached,
    );
    evidenceRequestId.current = prepared.requestId;
    evidenceAbort.current = prepared.controller;
    setEvidenceLoading(prepared.loading);
    setEvidenceError(null);
    if (cached) {
      const resultCount = analyticsEvidenceCount(activeResult?.selectedEvidence ?? null);
      trackSearchAnalytics("evidence_result", {
        search_mode: activeMode,
        source: analyticsSource,
        outcome: resultCount > 0 ? "success" : "empty",
        duration_ms: 0,
        bin_key: nextSelection.binKey,
        series_index: seriesIndex,
        cached: true,
        result_count: resultCount,
      }, searchId);
      return;
    }

    const currentRequest = prepared.requestId;
    const controller = prepared.controller!;
    let statusCode: number | undefined;
    let analyticsFinished = false;
    const finishEvidenceAnalytics = (
      outcome: "success" | "empty" | "error" | "aborted",
      extras: { result_count?: number; error_code?: string } = {},
    ) => {
      if (analyticsFinished) return;
      analyticsFinished = true;
      trackSearchAnalytics("evidence_result", {
        search_mode: activeMode,
        source: analyticsSource,
        outcome,
        duration_ms: analyticsDuration(startedAt, performance.now()),
        bin_key: nextSelection.binKey,
        series_index: seriesIndex,
        cached: false,
        ...(statusCode ? { status_code: statusCode } : {}),
        ...extras,
      }, searchId);
    };
    try {
      let selectedEvidence: SelectedEvidence | null;
      if (activeMode === "keyword") {
        const { response, payload } = await requestKeywordEvidence(
          activeQuery,
          nextSelection,
          { signal: controller.signal },
        );
        statusCode = response.status;
        if (!response.ok) throw new Error(errorMessage(payload, "Evidence could not be loaded."));
        if (!isEvidenceEnvelope(payload)) {
          throw new Error("The evidence service returned unsupported evidence.");
        }
        selectedEvidence = payload.selectedEvidence;
      } else {
        const { response, payload } = await requestVisualEvidence(
          activeQuery,
          nextSelection,
          { signal: controller.signal },
        );
        statusCode = response.status;
        if (!response.ok) {
          throw new Error(errorMessage(
            payload,
            "Visual evidence is unavailable.",
          ));
        }
        if (!isEvidenceEnvelope(payload)) {
          throw new Error("Visual search returned unsupported evidence.");
        }
        selectedEvidence = payload.selectedEvidence;
      }
      if (evidenceRequestId.current !== currentRequest) {
        finishEvidenceAnalytics("aborted");
        return;
      }
      const resultCount = analyticsEvidenceCount(selectedEvidence);
      finishEvidenceAnalytics(resultCount > 0 ? "success" : "empty", {
        result_count: resultCount,
      });
      cacheSelectedEvidencePreview(
        hoverPreviewCache.current,
        failedHoverArtworkIds.current,
        activeQuery,
        activeMode,
        selectedEvidence,
      );
      setResult((current) => current ? { ...current, selectedEvidence } : current);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") {
        finishEvidenceAnalytics("aborted");
        return;
      }
      finishEvidenceAnalytics("error", {
        error_code: analyticsFailureCode(caught, statusCode),
      });
      if (evidenceRequestId.current !== currentRequest) return;
      setEvidenceError(caught instanceof Error ? caught.message : "Evidence could not be loaded.");
    } finally {
      if (evidenceRequestId.current === currentRequest) setEvidenceLoading(false);
    }
  }

  useEffect(() => {
    const initial = searchPageStateFromUrl(window.location.href, INITIAL_QUERY);
    const initialUrl = new URL(window.location.href);
    const pageSource = initial.selection
      ? "shared_selection"
      : initialUrl.searchParams.has("q") || initialUrl.searchParams.has("searchMode")
        ? "shared_query"
        : "default";
    let queryCount = 1;
    try {
      queryCount = parseConceptQuery(initial.query).length;
    } catch {
      // The search flow will report any invalid shared query in more detail.
    }
    trackPageAnalyticsOnce("page_view", {
      search_mode: initial.mode,
      source: pageSource,
      referrer_kind: analyticsReferrerKind(document.referrer, window.location.hostname),
      viewport_bucket: analyticsViewportBucket(window.innerWidth),
      query_count: queryCount,
    });
    setInput(initial.query);
    setSearchMode(initial.mode);
    void search(initial.query, initial.mode, {
      requestedSelection: initial.selection,
      analyticsSource: pageSource === "default" ? "initial_default" : "url_restore",
    });
    return () => {
      searchAbort.current?.abort();
      evidenceAbort.current?.abort();
      hoverPreviewRequestId.current += 1;
      if (hoverPreviewTimer.current !== null) clearTimeout(hoverPreviewTimer.current);
      hoverPreviewAbort.current?.abort();
    };
    // Initial state comes from the shareable URL; subsequent searches are user-driven.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const evidenceItems = useMemo(
    () => selectedEvidenceItems(result, selection),
    [result, selection],
  );
  const nearestGroups = useMemo(
    () => submittedSearchMode === "embedding" ? nearestMatchGroups(result) : [],
    [result, submittedSearchMode],
  );
  const selectedQuery = result?.queries.find((query) => query.id === selection?.queryId) ?? null;
  const selectedBin = result?.bins.find((bin) => bin.key === selection?.binKey) ?? null;
  const selectedSeries = result?.series.find((series) => series.queryId === selection?.queryId) ?? null;
  const selectedPoint = selectedSeries && selection ? pointForBin(selectedSeries, selection.binKey) : null;
  const displayedBins = result?.bins.length ? timelineWindow(result.bins) : [];
  const hasChartPoints = result?.series.some((series) => series.points.length > 0) ?? false;
  const allTermsUnmatched = Boolean(
    result?.series.length && result.series.every((series) => series.k === 0),
  );
  const yearRange = displayedBins.length
    ? `${formatTimelineYear(displayedBins[0].start)}–${formatTimelineYear(displayedBins[displayedBins.length - 1].end)}`
    : "";
  const errorPlacement = searchErrorPlacement(error, result !== null);
  const exampleQueries = searchMode === "embedding" ? EXAMPLE_QUERIES : METADATA_EXAMPLE_QUERIES;
  const resultsTitle = submittedSearchMode === "embedding"
    ? "Visual matches over time"
    : "Metadata matches over time";

  const handleEvidenceDepthReached = useCallback((nextVisibleCount: number) => {
    if (!selection) return;
    const searchId = resultSearchId.current;
    trackSearchAnalyticsOnce(
      `gallery_depth:${searchId ?? "none"}:${selection.queryId}:${selection.binKey}:${nextVisibleCount}`,
      "gallery_depth",
      {
        search_mode: submittedSearchMode,
        placement: "evidence",
        visible_count: nextVisibleCount,
        series_index: analyticsSeriesIndex(result, selection.queryId),
        bin_key: selection.binKey,
      },
      searchId,
    );
  }, [result, selection, submittedSearchMode]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void search(input, searchMode, { analyticsSource: "form" });
  }

  function activateSeries(queryId: string, interaction: TimelineSeriesInteraction) {
    if (!result || hiddenQueryIds.has(queryId)) return;
    trackSearchAnalytics("series_activate", {
      search_mode: submittedSearchMode,
      source: interaction.source,
      input_method: interaction.inputMethod,
      series_index: analyticsSeriesIndex(result, queryId),
    }, resultSearchId.current);
    const candidate = selection && result.series.find((series) => series.queryId === queryId)?.points.some(
      (point) => point.binKey === selection.binKey,
    )
      ? { queryId, binKey: selection.binKey }
      : peakSelection(result, queryId);
    if (candidate) void loadEvidence(candidate, { analyticsSource: "series_activate" });
    else setSelection(null);
  }

  function toggleSeries(queryId: string, inputMethod: TimelineInputMethod) {
    const nextHidden = new Set(hiddenQueryIds);
    const isHidden = nextHidden.has(queryId);
    if (isHidden) nextHidden.delete(queryId);
    else nextHidden.add(queryId);
    setHiddenQueryIds(nextHidden);
    trackSearchAnalytics("series_toggle", {
      search_mode: submittedSearchMode,
      action: isHidden ? "show" : "hide",
      input_method: inputMethod,
      series_index: analyticsSeriesIndex(result, queryId),
      visible_count: Math.max(0, (result?.queries.length ?? 0) - nextHidden.size),
    }, resultSearchId.current);

    if (!isHidden && selection?.queryId === queryId && result) {
      const replacement = result.queries.find((query) => !nextHidden.has(query.id));
      const candidate = replacement ? peakSelection(result, replacement.id) : null;
      if (candidate) void loadEvidence(candidate, { analyticsSource: "series_replacement" });
      else setSelection(null);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar" id="top">
        <a className="wordmark" href="#top" aria-label="Mnemosyne home">Mnemosyne</a>
      </header>

      <div className="workspace">
        <section className="search-area" aria-label="Search museum collections">
          <div className="search-controls">
            <div className="search-toolbar">
              <form className="search-form" onSubmit={submit}>
                <label className="sr-only" htmlFor="concept-search">
                  {SEARCH_INPUT_LABELS[searchMode]}
                </label>
                <div className="search-input-wrap">
                  <input
                    id="concept-search"
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    placeholder={SEARCH_PLACEHOLDERS[searchMode]}
                    maxLength={MAX_QUERY_LENGTH}
                    aria-describedby="search-mode-help"
                  />
                </div>
                <button type="submit" disabled={loading}>
                  {loading ? "Searching…" : "Search"}
                </button>
              </form>
            </div>

            <div className="query-row">
              <span className="search-mode-help" id="search-mode-help">
                {SEARCH_MODE_HELP[searchMode]}
              </span>
              <span className="example-query-label">Try:</span>
              <ul className="example-query-list">
                {exampleQueries.map((example, index) => (
                  <li key={example}>
                    <button
                      type="button"
                      onClick={() => void search(example, searchMode, {
                        analyticsSource: "example",
                        exampleIndex: index,
                      })}
                    >
                      {example}
                    </button>
                  </li>
                ))}
              </ul>
            </div>

            {errorPlacement === "inline" && <div className="search-error" role="alert">{error}</div>}
          </div>
        </section>

        <section className="results" aria-live="polite" aria-busy={loading}>
          <div className="results-heading">
            <div>
              <h2>
                {loading
                  ? submittedSearchMode === "embedding"
                    ? "Searching artworks…"
                    : "Searching the catalogue…"
                  : resultsTitle}
              </h2>
              {!loading && result && (
                <ChartCalculationTooltip
                  metric={result.metric}
                  onOpen={() => trackSearchAnalyticsOnce(
                    `help:calculation:${resultSearchId.current ?? "none"}`,
                    "help_open",
                    {
                      search_mode: submittedSearchMode,
                      target: "calculation",
                    },
                    resultSearchId.current,
                  )}
                />
              )}
              {!loading && yearRange && <span>{yearRange}</span>}
            </div>
            {result && !loading && (
              <p>
                {result.queries.length} {result.queries.length === 1 ? "term" : "terms"} · {corpusSummary(result.corpus)}
              </p>
            )}
          </div>

          {errorPlacement === "empty" && (
            <div className="message-state" role="alert">
              <span>{error}</span>
            </div>
          )}
          {loading && submittedSearchMode === "embedding" && (
            <p className="cold-start-note">A new visual search can take a moment.</p>
          )}
          {loading && <div className="chart-skeleton" />}
          {!loading && result && result.bins.length > 0 && hasChartPoints && (
            <Timeline
              bins={result.bins}
              series={result.series}
              queries={result.queries}
              metric={result.metric}
              countingUnit={result.corpus.countingUnit}
              label={resultsTitle}
              description={CHART_HELP[submittedSearchMode]}
              selection={selection}
              hiddenQueryIds={hiddenQueryIds}
              hoverPreview={hoverPreview}
              onSelect={(nextSelection, inputMethod) => {
                if (
                  selection?.queryId === nextSelection.queryId &&
                  selection.binKey === nextSelection.binKey
                ) return;
                trackSearchAnalytics("point_select", {
                  search_mode: submittedSearchMode,
                  input_method: inputMethod,
                  bin_key: nextSelection.binKey,
                  series_index: analyticsSeriesIndex(result, nextSelection.queryId),
                }, resultSearchId.current);
                void loadEvidence(nextSelection, { analyticsSource: "point" });
              }}
              onActivateSeries={activateSeries}
              onToggleSeries={toggleSeries}
              onHelpOpen={() => trackSearchAnalyticsOnce(
                `help:chart_reading:${resultSearchId.current ?? "none"}`,
                "help_open",
                {
                  search_mode: submittedSearchMode,
                  target: "chart_reading",
                },
                resultSearchId.current,
              )}
              onHoverSelection={handleHoverSelection}
              onHoverPreviewError={handleHoverPreviewError}
            />
          )}
          {!loading && !error && result && !result.bins.length && (
            <div className="message-state">No dated artworks were found for this search.</div>
          )}
          {!loading && !error && result && result.bins.length > 0 && !hasChartPoints && (
            <div className="message-state">
              {submittedSearchMode === "keyword"
                ? "No dated artworks matched these metadata keywords."
                : allTermsUnmatched && nearestGroups.length
                  ? "None of the search terms produced a strong visual match. The closest artworks are shown below."
                  : "No strong visual matches were found in the dated collection."}
            </div>
          )}
        </section>

        {!loading && !error && nearestGroups.length > 0 && (
          <section className="nearest-results" aria-labelledby="nearest-results-heading" aria-live="polite">
            <div className="evidence-heading">
              <h2 id="nearest-results-heading">Closest visual results</h2>
              <p>Below the strong-match cutoff</p>
            </div>
            <p className="nearest-results-note">
              These are the most visually similar artworks the search found, but none met the cutoff for timeline evidence.
            </p>
            {nearestGroups.map(({ query, artworks }) => (
              <div className="nearest-result-group" key={query.id}>
                <div className="nearest-result-heading">
                  <h3>No strong visual matches for “{query.label}”</h3>
                  <p>Showing {artworks.length} closest result{artworks.length === 1 ? "" : "s"}</p>
                </div>
                <div className="artwork-grid">
                  {artworks.map((artwork, index) => (
                    <ArtworkCard
                      key={artwork.artworkId}
                      artwork={artwork}
                      analytics={{
                        placement: "nearest",
                        searchMode: submittedSearchMode,
                        searchId: resultSearchId.current,
                        seriesIndex: analyticsSeriesIndex(result, query.id),
                        rank: index + 1,
                      }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </section>
        )}

        {(hasChartPoints || !nearestGroups.length) && (
          <section className="evidence" aria-label="Artworks for the selected chart point">
          <div className="evidence-heading">
            <h2>
              {!selection || !selectedQuery || !selectedBin
                ? "Select a point on the chart to see artworks"
                : `${selectedQuery.label} · ${selectedBin.label}`}
            </h2>
            {selectedPoint && (
              <p>
                {evidenceLoading
                  ? "Loading artworks…"
                  : evidenceError
                    ? "Artworks unavailable"
                    : evidencePreviewLabel(
                        Math.min(visibleEvidenceCount, evidenceItems.length),
                        evidenceItems.length,
                        selectedPoint.objectCount,
                        result!.corpus.countingUnit,
                      )}
              </p>
            )}
          </div>

          {evidenceError && <p className="evidence-error" role="alert">{evidenceError}</p>}
          <ProgressiveArtworkGrid
            key={selection ? `${selection.queryId}:${selection.binKey}` : "unselected"}
            artworks={evidenceItems}
            emptyMessage={submittedSearchMode === "keyword"
              ? "No keyword matches in this period."
              : "No strong visual matches in this period."}
            isLoading={evidenceLoading}
            analytics={{
              placement: "evidence",
              searchMode: submittedSearchMode,
              searchId: resultSearchId.current,
              seriesIndex: analyticsSeriesIndex(result, selection?.queryId ?? ""),
              ...(selection ? { binKey: selection.binKey } : {}),
            }}
            onDepthReached={handleEvidenceDepthReached}
            onVisibleCountChange={setVisibleEvidenceCount}
            showEmpty={Boolean(!evidenceError && selection && evidenceItems.length === 0 && !loading)}
            visibleCount={visibleEvidenceCount}
          />
          </section>
        )}

        <footer className="source-footer">
          {submittedSearchMode === "embedding"
            ? "Visual search uses public-domain artwork images and CC0 catalog data from The Metropolitan Museum of Art, the National Gallery of Art, and the Cleveland Museum of Art."
            : "Metadata search uses catalog data from The Metropolitan Museum of Art Open Access collection."}
        </footer>
      </div>
    </main>
  );
}
