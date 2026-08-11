"use client";

import {
  type FormEvent,
  type KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Timeline } from "../components/Timeline";
import {
  defaultConceptSelection,
  loadConceptCatalog,
  loadConceptEvidence,
  loadConceptSearch,
  resolveConcept,
  suggestConcepts,
  UnknownConceptError,
  type ConceptCatalog,
  type ConceptResolution,
  type ConceptSuggestion,
} from "../lib/concept-catalog";
import { selectedEvidenceItems } from "../lib/evidence";
import {
  activeQueryFragment,
  evidenceMatchesSelection,
  formatQueryTerm,
  invalidSearchStatus,
  invalidateExplorerRequests,
  prepareEvidenceRequest,
  replaceActiveQueryFragment,
  retryablePromise,
  searchErrorPlacement,
} from "../lib/explorer-state";
import { MAX_QUERY_LENGTH, normalizeQueryTerm, parseConceptQuery, QuerySyntaxError } from "../lib/query";
import {
  buildEvidenceUrl,
  buildSearchUrl,
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
  EvidenceArtwork,
  SearchResponse,
  SelectedEvidence,
} from "../lib/types";

const INITIAL_QUERY = "Horse, Ship";
const EXAMPLE_QUERIES = [
  "Flowers, Mountain, Moon",
  "Portrait, Mother and child",
  "Armor, Sword, Crown",
  "Landscape, Geometric ornament",
  "Battle, Procession",
];
const INITIAL_VISIBLE_WORKS = 5;

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
  catalog?: ConceptCatalog;
  query?: string;
  mode?: SearchMode;
};

type UnknownPrompt = {
  input: string;
  suggestions: ConceptSuggestion[];
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
        <strong title={artwork.title}>{artwork.title}</strong>
        <span title={artwork.artist}>{artwork.artist || "Unknown artist"}</span>
        <small>{artwork.dateDisplay} ↗</small>
      </div>
    </a>
  );
}

function urlSelection(selection: ChartSelection | null, mode: SearchMode) {
  if (!selection || mode !== "embedding") return selection;
  return { ...selection, queryId: selection.queryId.replace(/^concept:/, "") };
}

function selectionFromRequestedState(
  response: SearchResponse,
  requested: ChartSelection | null | undefined,
  mode: SearchMode,
) {
  if (!requested) return null;
  const queryId = mode === "embedding" && !requested.queryId.startsWith("concept:")
    ? `concept:${requested.queryId}`
    : requested.queryId;
  const series = response.series.find((item) => item.queryId === queryId);
  if (!series?.points.some((point) => point.binKey === requested.binKey)) return null;
  if (response.bins.find((bin) => bin.key === requested.binKey)?.belowMinimumDenominator === true) {
    return null;
  }
  return { queryId, binKey: requested.binKey };
}

function replaceUnknownQueryTerm(value: string, unknown: string, replacement: string) {
  try {
    const unknownNormalized = normalizeQueryTerm(unknown);
    return parseConceptQuery(value)
      .map((term) => term.normalized === unknownNormalized ? replacement : term.label)
      .map(formatQueryTerm)
      .join(", ");
  } catch {
    return replacement;
  }
}

export default function Home() {
  const [input, setInput] = useState(INITIAL_QUERY);
  const [submittedQuery, setSubmittedQuery] = useState(INITIAL_QUERY);
  const [searchMode, setSearchMode] = useState<SearchMode>(DEFAULT_SEARCH_MODE);
  const [submittedSearchMode, setSubmittedSearchMode] = useState<SearchMode>(DEFAULT_SEARCH_MODE);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [selection, setSelection] = useState<ChartSelection | null>(null);
  const [resolutions, setResolutions] = useState<ConceptResolution[]>([]);
  const [unknownPrompt, setUnknownPrompt] = useState<UnknownPrompt | null>(null);
  const [hiddenQueryIds, setHiddenQueryIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [showAllExamples, setShowAllExamples] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<ConceptCatalog | null>(null);
  const [autocompleteOpen, setAutocompleteOpen] = useState(false);
  const [activeSuggestionIndex, setActiveSuggestionIndex] = useState(0);
  const requestId = useRef(0);
  const evidenceRequestId = useRef(0);
  const searchAbort = useRef<AbortController | null>(null);
  const evidenceAbort = useRef<AbortController | null>(null);
  const catalogPromise = useRef<Promise<ConceptCatalog> | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  async function ensureCatalog() {
    const loaded = await retryablePromise(catalogPromise, () => loadConceptCatalog());
    setCatalog((current) => current ?? loaded);
    return loaded;
  }

  function replacePageState(query: string, mode: SearchMode, nextSelection: ChartSelection | null) {
    const nextUrl = pageUrlForSearchState(window.location.href, {
      query,
      mode,
      selection: urlSelection(nextSelection, mode),
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

    let parsed;
    try {
      parsed = parseConceptQuery(nextQuery);
    } catch (caught) {
      const status = invalidSearchStatus(
        caught instanceof QuerySyntaxError ? caught.message : "Check the query and try again.",
      );
      setError(status.error);
      setLoading(status.loading);
      setEvidenceLoading(status.evidenceLoading);
      setEvidenceError(null);
      setUnknownPrompt(null);
      setAutocompleteOpen(false);
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
    setUnknownPrompt(null);
    setResolutions([]);
    setSelection(null);
    setShowAllExamples(false);
    setHiddenQueryIds(new Set());
    setAutocompleteOpen(false);

    try {
      let payload: SearchResponse;
      let loadedCatalog: ConceptCatalog | undefined;
      let nextResolutions: ConceptResolution[] = [];

      if (nextMode === "embedding") {
        loadedCatalog = await ensureCatalog();
        nextResolutions = parsed.map((term) => {
          const resolved = resolveConcept(loadedCatalog!, term.label);
          if (!resolved) {
            throw new UnknownConceptError(term.label, suggestConcepts(loadedCatalog!, term.label, 4));
          }
          return resolved;
        });
        payload = await loadConceptSearch(loadedCatalog, nextResolutions);
      } else {
        const response = await fetch(buildSearchUrl(trimmedQuery, nextMode), {
          signal: controller.signal,
        });
        const body: unknown = await response.json();
        if (!response.ok) throw new Error(errorMessage(body, "Search failed."));
        if (!isSearchResponse(body)) throw new Error("The search service returned an unsupported response.");
        payload = body;
      }

      if (requestId.current !== currentRequest) return;
      const requestedSelection = selectionFromRequestedState(
        payload,
        options.requestedSelection,
        nextMode,
      );
      const nextSelection = requestedSelection ?? (
        nextMode === "embedding" && loadedCatalog
          ? defaultConceptSelection(loadedCatalog, payload)
          : payload.selectedEvidence
            ? { queryId: payload.selectedEvidence.queryId, binKey: payload.selectedEvidence.binKey }
            : peakSelection(payload)
      );
      setResolutions(nextResolutions);
      setResult(payload);
      setSelection(nextSelection);
      replacePageState(trimmedQuery, nextMode, nextSelection);
      if (nextSelection && !evidenceMatchesSelection(payload.selectedEvidence, nextSelection)) {
        void Promise.resolve().then(() => {
          if (requestId.current !== currentRequest) return;
          void loadEvidence(nextSelection, {
            baseResult: payload,
            catalog: loadedCatalog,
            query: trimmedQuery,
            mode: nextMode,
          });
        });
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      if (requestId.current !== currentRequest) return;
      if (caught instanceof UnknownConceptError) {
        setUnknownPrompt({ input: caught.input, suggestions: caught.suggestions });
      }
      setError(caught instanceof Error ? caught.message : "Search failed.");
      setResult(null);
      setSelection(null);
      setResolutions([]);
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
      if (activeMode === "embedding") {
        const activeCatalog = context.catalog ?? catalog ?? await ensureCatalog();
        selectedEvidence = await loadConceptEvidence(
          activeCatalog,
          nextSelection.queryId,
          nextSelection.binKey,
        );
      } else {
        const response = await fetch(
          buildEvidenceUrl(activeQuery, activeMode, nextSelection),
          { signal: controller.signal },
        );
        const payload: unknown = await response.json();
        if (!response.ok) throw new Error(errorMessage(payload, "Evidence could not be loaded."));
        if (!isEvidenceEnvelope(payload)) {
          throw new Error("The evidence service returned unsupported evidence.");
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

  const queryFragment = activeQueryFragment(input);
  const autocompleteSuggestions = useMemo(
    () => searchMode === "embedding" && catalog && autocompleteOpen
      ? suggestConcepts(catalog, queryFragment, 7)
      : [],
    [autocompleteOpen, catalog, queryFragment, searchMode],
  );
  useEffect(() => setActiveSuggestionIndex(0), [queryFragment, autocompleteOpen]);

  const aliasResolutions = resolutions.filter((item) =>
    item.matchedBy === "alias" || item.requestedNormalized !== item.concept.normalized
  );
  const evidenceItems = useMemo(
    () => selectedEvidenceItems(result, selection),
    [result, selection],
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
  const yearRange = displayedBins.length
    ? `${formatTimelineYear(displayedBins[0].start)}–${formatTimelineYear(displayedBins[displayedBins.length - 1].end)}`
    : "";
  const errorPlacement = searchErrorPlacement(error, result !== null);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void search(input, searchMode);
  }

  function changeSearchMode(nextMode: SearchMode) {
    if (nextMode === searchMode) return;
    void search(input, nextMode, { syncInput: false });
  }

  function chooseSuggestion(suggestion: ConceptSuggestion) {
    const next = replaceActiveQueryFragment(input, suggestion.concept.label);
    setInput(next);
    setAutocompleteOpen(false);
    inputRef.current?.focus();
  }

  function handleSearchKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (!autocompleteOpen || !autocompleteSuggestions.length) {
      if (event.key === "ArrowDown" && searchMode === "embedding") setAutocompleteOpen(true);
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveSuggestionIndex((current) => (current + 1) % autocompleteSuggestions.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveSuggestionIndex((current) =>
        (current - 1 + autocompleteSuggestions.length) % autocompleteSuggestions.length
      );
    } else if (event.key === "Enter") {
      event.preventDefault();
      chooseSuggestion(autocompleteSuggestions[activeSuggestionIndex] ?? autocompleteSuggestions[0]);
    } else if (event.key === "Escape") {
      event.preventDefault();
      setAutocompleteOpen(false);
    }
  }

  function applyUnknownSuggestion(suggestion: ConceptSuggestion) {
    if (!unknownPrompt) return;
    const next = replaceUnknownQueryTerm(input, unknownPrompt.input, suggestion.concept.label);
    setInput(next);
    void search(next, searchMode);
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
      <header className="topbar">
        <a className="wordmark" href="#top" aria-label="Mnemosyne home">Mnemosyne</a>
      </header>

      <div className="workspace" id="top">
        <section className="search-area" aria-labelledby="page-title">
          <div className="intro">
            <h1 id="page-title">Trace a visual idea through art history</h1>
            <p>Compare up to five precomputed visual concepts across The Met’s public-domain image collection, or switch to catalogue metadata keywords.</p>
          </div>

          <div className="search-controls">
            <form className="search-form" onSubmit={submit}>
              <label className="sr-only" htmlFor="concept-search">
                {searchMode === "embedding" ? "Compare visual concepts" : "Search metadata keywords"}
              </label>
              <div className="search-input-wrap">
                <input
                  ref={inputRef}
                  id="concept-search"
                  value={input}
                  onChange={(event) => {
                    setInput(event.target.value);
                    setAutocompleteOpen(searchMode === "embedding");
                  }}
                  onFocus={() => {
                    if (searchMode !== "embedding") return;
                    setAutocompleteOpen(true);
                    void ensureCatalog().catch(() => undefined);
                  }}
                  onBlur={() => setAutocompleteOpen(false)}
                  onKeyDown={handleSearchKeyDown}
                  placeholder={searchMode === "embedding" ? "Horse, Ship" : "industry, machine, skyscraper"}
                  maxLength={MAX_QUERY_LENGTH}
                  aria-describedby="query-help"
                  role="combobox"
                  aria-autocomplete="list"
                  aria-controls="concept-suggestions"
                  aria-expanded={autocompleteOpen && autocompleteSuggestions.length > 0}
                  aria-activedescendant={autocompleteOpen && autocompleteSuggestions.length
                    ? `concept-suggestion-${activeSuggestionIndex}`
                    : undefined}
                />
                {autocompleteOpen && autocompleteSuggestions.length > 0 && (
                  <ul className="autocomplete-list" id="concept-suggestions" role="listbox">
                    {autocompleteSuggestions.map((suggestion, index) => (
                      <li
                        className="autocomplete-option"
                        id={`concept-suggestion-${index}`}
                        key={suggestion.concept.id}
                        role="option"
                        aria-selected={index === activeSuggestionIndex}
                        onMouseDown={(event) => event.preventDefault()}
                        onMouseEnter={() => setActiveSuggestionIndex(index)}
                        onClick={() => chooseSuggestion(suggestion)}
                      >
                        <strong>{suggestion.concept.label}</strong>
                        <span>
                          {suggestion.matchedBy === "alias"
                            ? `${suggestion.matchedText} → ${suggestion.concept.label}`
                            : suggestion.concept.category ?? "Visual concept"}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              <button type="submit" disabled={loading}>{loading ? "Searching…" : "Search"}</button>
            </form>

            <div className="search-mode-row">
              <span>Search by</span>
              <span className="search-mode-toggle" role="group" aria-label="Search method">
                {SEARCH_MODES.map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    aria-pressed={searchMode === mode}
                    title={mode === "keyword"
                      ? "Match words in catalogue metadata"
                      : "Use precomputed image-embedding timelines for curated concepts"}
                    onClick={() => changeSearchMode(mode)}
                  >
                    {SEARCH_MODE_LABELS[mode]}
                  </button>
                ))}
              </span>
            </div>

            <div className="query-row" id="query-help">
              <span>{searchMode === "embedding" ? "Curated concepts:" : "Try:"}</span>
              {EXAMPLE_QUERIES.map((example) => (
                <button key={example} type="button" onClick={() => void search(example, searchMode)}>{example}</button>
              ))}
            </div>

            {submittedSearchMode === "embedding" && aliasResolutions.length > 0 && (
              <div className="resolution-row" aria-label="Resolved visual concept aliases">
                <strong>Resolved:</strong>
                {aliasResolutions.map((item) => (
                  <span className="resolution-pill" key={`${item.requestedNormalized}:${item.concept.id}`}>
                    {item.requested} → <strong>{item.concept.label}</strong>
                  </span>
                ))}
              </div>
            )}
            {errorPlacement === "inline" && <div className="search-error" role="alert">{error}</div>}
          </div>
        </section>

        <section className="results" aria-live="polite" aria-busy={loading}>
          <div className="results-heading">
            <div>
              <h2>{loading ? "Searching the collection…" : result?.metric.label ?? "Results over time"}</h2>
              {!loading && yearRange && <span>{yearRange}</span>}
            </div>
            {result && !loading && <p>{result.queries.length} series · {result.corpus.label}</p>}
          </div>

          {errorPlacement === "empty" && (
            <div className="message-state" role="alert">
              <span>{error}</span>
              {unknownPrompt && unknownPrompt.suggestions.length > 0 && (
                <span className="unknown-suggestions">
                  <span>Did you mean</span>
                  {unknownPrompt.suggestions.map((suggestion) => (
                    <button key={suggestion.concept.id} type="button" onClick={() => applyUnknownSuggestion(suggestion)}>
                      {suggestion.concept.label}
                    </button>
                  ))}
                </span>
              )}
            </div>
          )}
          {loading && <div className="chart-skeleton" />}
          {!loading && result && result.bins.length > 0 && hasChartPoints && (
            <Timeline
              bins={result.bins}
              series={result.series}
              queries={result.queries}
              metric={result.metric}
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
              {submittedSearchMode === "embedding"
                ? "No dated visual matches passed the score cutoff. Empty periods mean insufficient evidence, not zero historical prevalence."
                : "No dated artworks matched these metadata keywords."}
            </div>
          )}
        </section>

        <section className="evidence" aria-label="Evidence for the selected chart point">
          <div className="evidence-heading">
            <h2>
              {!selection || !selectedQuery || !selectedBin
                ? "Select a line and period"
                : `${selectedQuery.label} · ${selectedBin.label}`}
            </h2>
            {selectedPoint && submittedSearchMode !== "embedding" && (
              <p>{selectedPoint.objectCount} contributing work{selectedPoint.objectCount === 1 ? "" : "s"}</p>
            )}
            {selectedPoint && submittedSearchMode === "embedding" && (
              <p>
                {evidenceLoading
                  ? "Loading visual matches…"
                  : evidenceError
                    ? "Evidence unavailable"
                    : `${selectedEvidence?.contributorCount ?? 0} visual match${(selectedEvidence?.contributorCount ?? 0) === 1 ? "" : "es"}`}
              </p>
            )}
          </div>

          {evidenceError && <p className="evidence-error" role="alert">{evidenceError}</p>}
          <div className="artwork-grid" aria-busy={evidenceLoading}>
            {evidenceLoading && <p className="no-works">Loading evidence…</p>}
            {!evidenceLoading && visibleItems.map((artwork) => (
              <ArtworkCard key={artwork.artworkId} artwork={artwork} />
            ))}
            {!evidenceLoading && !evidenceError && selection && evidenceItems.length === 0 && !loading && (
              <p className="no-works">
                {submittedSearchMode === "embedding"
                  ? "No visual matches passed the score cutoff in this period."
                  : "No keyword matches in this period."}
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
                ? "Show fewer matches"
                : `Show ${hiddenExampleCount} more match${hiddenExampleCount === 1 ? "" : "es"}`}
            </button>
          )}
        </section>
      </div>
    </main>
  );
}
