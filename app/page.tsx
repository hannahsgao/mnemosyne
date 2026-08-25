"use client";

import {
  type FormEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Timeline } from "../components/Timeline";
import { nearestMatchGroups, selectedEvidenceItems } from "../lib/evidence";
import {
  evidenceMatchesSelection,
  invalidSearchStatus,
  invalidateExplorerRequests,
  prepareEvidenceRequest,
  searchErrorPlacement,
} from "../lib/explorer-state";
import { MAX_QUERY_LENGTH, parseConceptQuery, QuerySyntaxError } from "../lib/query";
import { requestKeywordEvidence, requestKeywordSearch } from "../lib/keyword-transport";
import { requestVisualEvidence, requestVisualSearch } from "../lib/visual-transport";
import {
  DEFAULT_SEARCH_MODE,
  pageUrlForSearchState,
  SEARCH_MODE_LABELS,
  SEARCH_MODES,
  searchPageStateFromUrl,
  type SearchMode,
} from "../lib/search-mode";
import { formatTimelineYear, peakSelection, pointForBin, timelineWindow } from "../lib/timeline";
import type {
  ChartSelection,
  CorpusMetadata,
  EvidenceArtwork,
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

const SEARCH_INPUT_LABELS: Record<SearchMode, string> = {
  embedding: "Search artworks by visual content",
  keyword: "Search artwork catalogue metadata",
};

const SEARCH_PLACEHOLDERS: Record<SearchMode, string> = {
  embedding: "mirror, portrait, self-portrait",
  keyword: "carriage, automobile, airplane",
};

const SEARCH_MODE_TITLES: Record<SearchMode, string> = {
  embedding: "Search for what appears in the artwork",
  keyword: "Search titles, artists, tags, and catalogue text",
};

const SEARCH_MODE_HELP: Record<SearchMode, string> = {
  embedding: "Describe what you want to see.",
  keyword: "Search words in the catalogue record.",
};

const CHART_HELP: Record<SearchMode, string> = {
  embedding: "Higher values mean visual matches are more concentrated in that period than across the collection overall. Gaps mark periods with too little evidence.",
  keyword: "Higher values mean a larger share of dated catalogue records match in that period. Gaps mark periods with too little evidence.",
};

type SearchOptions = {
  syncInput?: boolean;
  requestedSelection?: ChartSelection | null;
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

function institutionLabel(value: string) {
  const normalized = value.trim().toLowerCase();
  if (normalized === "met" || normalized === "the met") return "The Met";
  if (normalized === "nga" || normalized === "national gallery of art") {
    return "National Gallery of Art";
  }
  if (normalized === "aic" || normalized === "art institute of chicago") {
    return "Art Institute of Chicago";
  }
  return value || "Museum source unavailable";
}

function itemNoun(count: number, countingUnit: CorpusMetadata["countingUnit"]) {
  if (countingUnit === "catalog-record") {
    return `catalog record${count === 1 ? "" : "s"}`;
  }
  return `work${count === 1 ? "" : "s"}`;
}

function matchingItemLabel(
  count: number,
  countingUnit: CorpusMetadata["countingUnit"],
) {
  return `${count} matching ${itemNoun(count, countingUnit)}`;
}

function corpusSummary(corpus: CorpusMetadata) {
  const label = corpus.label === corpus.id
    ? "Open-access museum image catalog"
    : corpus.label;
  return corpus.count === null
    ? label
    : `${corpus.count.toLocaleString()} ${itemNoun(corpus.count, corpus.countingUnit)} · ${label}`;
}

function ArtworkCard({ artwork }: { artwork: EvidenceArtwork }) {
  return (
    <a className="artwork-card" href={artwork.sourceRecordUrl} target="_blank" rel="noreferrer">
      <div className="artwork-image-wrap">
        {artwork.imageUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img className="artwork-image" src={artwork.imageUrl} alt="" loading="lazy" />
        ) : (
          <div className="image-placeholder">No image</div>
        )}
      </div>
      <div className="artwork-copy">
        <strong title={artwork.title || "Untitled"}>{artwork.title || "Untitled"}</strong>
        <span title={artwork.artist}>{artwork.artist || "Unknown artist"}</span>
        <em title={institutionLabel(artwork.institution)}>{institutionLabel(artwork.institution)}</em>
        <small>{artwork.dateDisplay} ↗</small>
      </div>
    </a>
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
  const [showAllExamples, setShowAllExamples] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const requestId = useRef(0);
  const evidenceRequestId = useRef(0);
  const searchAbort = useRef<AbortController | null>(null);
  const evidenceAbort = useRef<AbortController | null>(null);

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

    try {
      parseConceptQuery(nextQuery);
    } catch (caught) {
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
    const controller = new AbortController();
    searchAbort.current = controller;
    replacePageState(trimmedQuery, nextMode, null);
    setSearchMode(nextMode);
    setSubmittedSearchMode(nextMode);
    if (options.syncInput !== false) setInput(trimmedQuery);
    setSubmittedQuery(trimmedQuery);
    setLoading(true);
    setEvidenceLoading(false);
    setError(null);
    setEvidenceError(null);
    setSelection(null);
    setShowAllExamples(false);
    setHiddenQueryIds(new Set());

    try {
      let payload: SearchResponse;

      if (nextMode === "keyword") {
        const { response, payload: body } = await requestKeywordSearch(trimmedQuery, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(errorMessage(body, "Search failed."));
        if (!isSearchResponse(body)) throw new Error("The search service returned an unsupported response.");
        payload = body;
      } else {
        const { response, payload: body } = await requestVisualSearch(trimmedQuery, {
          signal: controller.signal,
        });
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

      if (requestId.current !== currentRequest) return;
      const requestedSelection = selectionFromRequestedState(
        payload,
        options.requestedSelection,
      );
      const nextSelection = requestedSelection ?? (
        payload.selectedEvidence
          ? { queryId: payload.selectedEvidence.queryId, binKey: payload.selectedEvidence.binKey }
          : peakSelection(payload)
      );
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
          });
        });
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
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
      setShowAllExamples(false);
    }
    setSelection(nextSelection);
    const activeQuery = context.query ?? submittedQuery;
    const activeMode = context.mode ?? submittedSearchMode;
    replacePageState(activeQuery, activeMode, nextSelection);
    const activeResult = context.baseResult ?? result;
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
    if (cached) return;

    const currentRequest = prepared.requestId;
    const controller = prepared.controller!;
    try {
      let selectedEvidence: SelectedEvidence | null;
      if (activeMode === "keyword") {
        const { response, payload } = await requestKeywordEvidence(
          activeQuery,
          nextSelection,
          { signal: controller.signal },
        );
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
      if (evidenceRequestId.current !== currentRequest) return;
      setResult((current) => current ? { ...current, selectedEvidence } : current);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      if (evidenceRequestId.current !== currentRequest) return;
      setEvidenceError(caught instanceof Error ? caught.message : "Evidence could not be loaded.");
    } finally {
      if (evidenceRequestId.current === currentRequest) setEvidenceLoading(false);
    }
  }

  useEffect(() => {
    const initial = searchPageStateFromUrl(window.location.href, INITIAL_QUERY);
    setInput(initial.query);
    setSearchMode(initial.mode);
    void search(initial.query, initial.mode, { requestedSelection: initial.selection });
    return () => {
      searchAbort.current?.abort();
      evidenceAbort.current?.abort();
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
  const visibleItems = evidenceItems.slice(0, showAllExamples ? evidenceItems.length : INITIAL_VISIBLE_WORKS);
  const hiddenExampleCount = Math.max(0, evidenceItems.length - INITIAL_VISIBLE_WORKS);
  const selectedQuery = result?.queries.find((query) => query.id === selection?.queryId) ?? null;
  const selectedBin = result?.bins.find((bin) => bin.key === selection?.binKey) ?? null;
  const selectedSeries = result?.series.find((series) => series.queryId === selection?.queryId) ?? null;
  const selectedPoint = selectedSeries && selection ? pointForBin(selectedSeries, selection.binKey) : null;
  const currentEvidence = result?.selectedEvidence ?? null;
  const selectedEvidence = currentEvidence &&
    currentEvidence.queryId === selection?.queryId &&
    currentEvidence.binKey === selection?.binKey
    ? currentEvidence
    : null;
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

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void search(input, searchMode);
  }

  function changeSearchMode(nextMode: SearchMode) {
    if (nextMode === searchMode) return;
    void search(input, nextMode, { syncInput: false });
  }

  function activateSeries(queryId: string) {
    if (!result || hiddenQueryIds.has(queryId)) return;
    const candidate = selection && result.series.find((series) => series.queryId === queryId)?.points.some(
      (point) => point.binKey === selection.binKey,
    )
      ? { queryId, binKey: selection.binKey }
      : peakSelection(result, queryId);
    if (candidate) void loadEvidence(candidate);
    else setSelection(null);
  }

  function toggleSeries(queryId: string) {
    const nextHidden = new Set(hiddenQueryIds);
    const isHidden = nextHidden.has(queryId);
    if (isHidden) nextHidden.delete(queryId);
    else nextHidden.add(queryId);
    setHiddenQueryIds(nextHidden);

    if (!isHidden && selection?.queryId === queryId && result) {
      const replacement = result.queries.find((query) => !nextHidden.has(query.id));
      const candidate = replacement ? peakSelection(result, replacement.id) : null;
      if (candidate) void loadEvidence(candidate);
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
              <span className="search-mode-toggle" role="group" aria-label="Search method">
                {SEARCH_MODES.map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    aria-pressed={searchMode === mode}
                    title={SEARCH_MODE_TITLES[mode]}
                    onClick={() => changeSearchMode(mode)}
                  >
                    {SEARCH_MODE_LABELS[mode]}
                  </button>
                ))}
              </span>

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
                    aria-describedby="query-help"
                  />
                </div>
                <button type="submit" disabled={loading}>
                  {loading ? "Searching…" : "Search"}
                </button>
              </form>
            </div>

            <div className="query-row" id="query-help">
              <span className="search-mode-help">{SEARCH_MODE_HELP[searchMode]}</span>
              <span>Try:</span>
              {exampleQueries.map((example) => (
                <button key={example} type="button" onClick={() => void search(example, searchMode)}>{example}</button>
              ))}
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
              onSelect={(nextSelection) => void loadEvidence(nextSelection)}
              onActivateSeries={activateSeries}
              onToggleSeries={toggleSeries}
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
                  {artworks.map((artwork) => (
                    <ArtworkCard key={artwork.artworkId} artwork={artwork} />
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
            {selectedPoint && submittedSearchMode === "keyword" && (
              <p>{matchingItemLabel(selectedPoint.objectCount, result!.corpus.countingUnit)}</p>
            )}
            {selectedPoint && submittedSearchMode !== "keyword" && (
              <p>
                {evidenceLoading
                  ? "Loading artworks…"
                  : evidenceError
                    ? "Artworks unavailable"
                    : matchingItemLabel(selectedEvidence?.contributorCount ?? 0, result!.corpus.countingUnit)}
              </p>
            )}
          </div>

          {evidenceError && <p className="evidence-error" role="alert">{evidenceError}</p>}
          <div className="artwork-grid" aria-busy={evidenceLoading}>
            {evidenceLoading && <p className="no-works">Loading artworks…</p>}
            {!evidenceLoading && visibleItems.map((artwork) => (
              <ArtworkCard key={artwork.artworkId} artwork={artwork} />
            ))}
            {!evidenceLoading && !evidenceError && selection && evidenceItems.length === 0 && !loading && (
              <p className="no-works">
                {submittedSearchMode === "keyword"
                  ? "No keyword matches in this period."
                  : "No strong visual matches in this period."}
              </p>
            )}
          </div>
          {!evidenceLoading && hiddenExampleCount > 0 && (
            <button
              className="more-examples"
              type="button"
              aria-expanded={showAllExamples}
              onClick={() => setShowAllExamples((current) => !current)}
            >
              {showAllExamples
                ? `Show fewer ${itemNoun(2, result!.corpus.countingUnit)}`
                : `Show ${hiddenExampleCount} more ${itemNoun(hiddenExampleCount, result!.corpus.countingUnit)}`}
            </button>
          )}
          </section>
        )}

        <footer className="source-footer">
          {submittedSearchMode === "embedding"
            ? "Visual search uses public-domain artwork images and CC0 catalog data from The Metropolitan Museum of Art and the National Gallery of Art."
            : "Metadata search uses catalog data from The Metropolitan Museum of Art Open Access collection."}
        </footer>
      </div>
    </main>
  );
}
