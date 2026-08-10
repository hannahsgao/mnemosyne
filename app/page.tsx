"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { Timeline } from "../components/Timeline";
import { MAX_QUERY_LENGTH, parseConceptQuery, QuerySyntaxError } from "../lib/query";
import {
  buildSearchUrl,
  DEFAULT_SEARCH_MODE,
  pageUrlForSearchMode,
  SEARCH_MODE_LABELS,
  SEARCH_MODES,
  searchModeFromUrl,
  type SearchMode,
} from "../lib/search-mode";
import { formatTimelineYear, peakSelection, pointForBin, timelineWindow } from "../lib/timeline";
import type {
  ChartSelection,
  EvidenceArtwork,
  SearchResponse,
} from "../lib/types";

const INITIAL_QUERY = "horse, ship";
const EXAMPLE_QUERIES = [
  "industry, machine, skyscraper",
  "railroad, automobile, airplane",
  "photography, poster, newspaper",
  "plastic, radio, television",
  "war, revolution",
];
const INITIAL_VISIBLE_WORKS = 5;

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

function selectedEvidenceItems(response: SearchResponse | null, selection: ChartSelection | null) {
  if (
    !response?.selectedEvidence ||
    !selection ||
    response.selectedEvidence.queryId !== selection.queryId ||
    response.selectedEvidence.binKey !== selection.binKey
  ) {
    return [];
  }

  const seen = new Set<string>();
  const selected: EvidenceArtwork[] = [];
  for (const artwork of response.selectedEvidence.slices.strongest) {
    if (!artwork || seen.has(artwork.artworkId)) continue;
    seen.add(artwork.artworkId);
    selected.push(artwork);
  }
  return selected;
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
        <span title={artwork.artist}>{artwork.artist}</span>
        <small>{artwork.dateDisplay} ↗</small>
      </div>
    </a>
  );
}

function replacePageSearchMode(mode: SearchMode) {
  const nextUrl = pageUrlForSearchMode(window.location.href, mode);
  window.history.replaceState(window.history.state, "", nextUrl);
}

export default function Home() {
  const [input, setInput] = useState(INITIAL_QUERY);
  const [submittedQuery, setSubmittedQuery] = useState(INITIAL_QUERY);
  const [searchMode, setSearchMode] = useState<SearchMode>(DEFAULT_SEARCH_MODE);
  const [submittedSearchMode, setSubmittedSearchMode] =
    useState<SearchMode>(DEFAULT_SEARCH_MODE);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [selection, setSelection] = useState<ChartSelection | null>(null);
  const [hiddenQueryIds, setHiddenQueryIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [showAllExamples, setShowAllExamples] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);
  const evidenceRequestId = useRef(0);
  const searchAbort = useRef<AbortController | null>(null);
  const evidenceAbort = useRef<AbortController | null>(null);

  async function search(
    nextQuery: string,
    nextMode: SearchMode = searchMode,
    options: { syncInput?: boolean } = {},
  ) {
    try {
      parseConceptQuery(nextQuery);
    } catch (caught) {
      setError(caught instanceof QuerySyntaxError ? caught.message : "Check the query and try again.");
      return;
    }

    const currentRequest = ++requestId.current;
    evidenceRequestId.current += 1;
    searchAbort.current?.abort();
    evidenceAbort.current?.abort();
    const controller = new AbortController();
    searchAbort.current = controller;
    replacePageSearchMode(nextMode);
    setSearchMode(nextMode);
    setSubmittedSearchMode(nextMode);
    if (options.syncInput !== false) setInput(nextQuery.trim());
    setSubmittedQuery(nextQuery.trim());
    setLoading(true);
    setEvidenceLoading(false);
    setError(null);
    setSelection(null);
    setShowAllExamples(false);
    setHiddenQueryIds(new Set());

    try {
      const response = await fetch(buildSearchUrl(nextQuery.trim(), nextMode), {
        signal: controller.signal,
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        const message =
          payload && typeof payload === "object" && "error" in payload
            ? String(payload.error)
            : "Search failed.";
        throw new Error(message);
      }
      if (!isSearchResponse(payload)) throw new Error("The search service returned an unsupported response.");
      if (requestId.current !== currentRequest) return;

      const nextSelection = payload.selectedEvidence
        ? {
            queryId: payload.selectedEvidence.queryId,
            binKey: payload.selectedEvidence.binKey,
          }
        : peakSelection(payload);
      setResult(payload);
      setSelection(nextSelection);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      if (requestId.current !== currentRequest) return;
      setError(caught instanceof Error ? caught.message : "Search failed.");
      setResult(null);
      setSelection(null);
    } finally {
      if (requestId.current === currentRequest) setLoading(false);
    }
  }

  async function loadEvidence(nextSelection: ChartSelection) {
    if (
      selection?.queryId !== nextSelection.queryId ||
      selection?.binKey !== nextSelection.binKey
    ) {
      setShowAllExamples(false);
    }
    setSelection(nextSelection);
    if (
      result?.selectedEvidence?.queryId === nextSelection.queryId &&
      result.selectedEvidence.binKey === nextSelection.binKey
    ) {
      return;
    }

    const currentRequest = ++evidenceRequestId.current;
    evidenceAbort.current?.abort();
    const controller = new AbortController();
    evidenceAbort.current = controller;
    setEvidenceLoading(true);
    setError(null);
    try {
      const response = await fetch(
        buildSearchUrl(submittedQuery, submittedSearchMode, nextSelection),
        {
          signal: controller.signal,
        },
      );
      const payload: unknown = await response.json();
      if (!response.ok) {
        const message =
          payload && typeof payload === "object" && "error" in payload
            ? String(payload.error)
            : "Evidence could not be loaded.";
        throw new Error(message);
      }
      if (!isSearchResponse(payload)) throw new Error("The search service returned unsupported evidence.");
      if (evidenceRequestId.current !== currentRequest) return;
      setResult((current) =>
        current
          ? {
              ...current,
              selectedEvidence: payload.selectedEvidence,
            }
          : current,
      );
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      if (evidenceRequestId.current !== currentRequest) return;
      setError(caught instanceof Error ? caught.message : "Evidence could not be loaded.");
    } finally {
      if (evidenceRequestId.current === currentRequest) setEvidenceLoading(false);
    }
  }

  useEffect(() => {
    const initialMode = searchModeFromUrl(
      new URLSearchParams(window.location.search).get("searchMode"),
    );
    void search(INITIAL_QUERY, initialMode);
    return () => {
      searchAbort.current?.abort();
      evidenceAbort.current?.abort();
    };
    // The first query is intentional; subsequent searches are user-driven.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const strongestEvidenceItems = useMemo(
    () => selectedEvidenceItems(result, selection),
    [result, selection],
  );
  const visibleItems = strongestEvidenceItems.slice(
    0,
    showAllExamples ? strongestEvidenceItems.length : INITIAL_VISIBLE_WORKS,
  );
  const hiddenExampleCount = Math.max(
    0,
    strongestEvidenceItems.length - INITIAL_VISIBLE_WORKS,
  );
  const selectedQuery = result?.queries.find((query) => query.id === selection?.queryId) ?? null;
  const selectedBin = result?.bins.find((bin) => bin.key === selection?.binKey) ?? null;
  const selectedSeries = result?.series.find((series) => series.queryId === selection?.queryId) ?? null;
  const selectedPoint =
    selectedSeries && selection ? pointForBin(selectedSeries, selection.binKey) : null;
  const currentEvidence = result?.selectedEvidence ?? null;
  const selectedEvidence =
    currentEvidence &&
    currentEvidence.queryId === selection?.queryId &&
    currentEvidence.binKey === selection?.binKey
      ? currentEvidence
      : null;
  const displayedBins = result?.bins.length ? timelineWindow(result.bins) : [];
  const hasChartPoints = result?.series.some((series) => series.points.length > 0) ?? false;
  const yearRange = displayedBins.length
    ? `${formatTimelineYear(displayedBins[0].start)}–${formatTimelineYear(displayedBins[displayedBins.length - 1].end)}`
    : "";

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void search(input, searchMode);
  }

  function changeSearchMode(nextMode: SearchMode) {
    if (nextMode === searchMode) return;
    void search(submittedQuery, nextMode, { syncInput: false });
  }

  function activateSeries(queryId: string) {
    if (!result || hiddenQueryIds.has(queryId)) return;
    const candidate =
      selection && result.series.find((series) => series.queryId === queryId)?.points.some(
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
            <h1 id="page-title">Trace an idea through art history</h1>
            <p>Compare up to five concepts using catalogue keywords or image embeddings. Separate lines with commas; quote a literal comma.</p>
          </div>

          <div className="search-controls">
            <form className="search-form" onSubmit={submit}>
              <label className="sr-only" htmlFor="concept-search">Compare visual concepts</label>
              <input
                id="concept-search"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="industry, machine, skyscraper"
                maxLength={MAX_QUERY_LENGTH}
                aria-describedby="query-help"
              />
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
                    title={
                      mode === "keyword"
                        ? "Match words in catalogue metadata"
                        : "Match visual concepts using image embeddings"
                    }
                    onClick={() => changeSearchMode(mode)}
                  >
                    {SEARCH_MODE_LABELS[mode]}
                  </button>
                ))}
              </span>
            </div>

            <div className="query-row" id="query-help">
              <span>Try:</span>
              {EXAMPLE_QUERIES.map((example) => (
                <button key={example} type="button" onClick={() => void search(example)}>{example}</button>
              ))}
            </div>
          </div>
        </section>

        <section className="results" aria-live="polite" aria-busy={loading}>
          <div className="results-heading">
            <div>
              <h2>{loading ? "Searching the collection…" : result?.metric.label ?? "Results over time"}</h2>
              {!loading && yearRange && <span>{yearRange}</span>}
            </div>
            {result && !loading && (
              <p>{result.queries.length} series · {result.corpus.label}</p>
            )}
          </div>

          {error && !result && <div className="message-state">{error} Try another search.</div>}
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
                : "No dated artworks matched these keywords."}
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
                {selectedEvidence?.contributorCount ?? 0} visual match
                {(selectedEvidence?.contributorCount ?? 0) === 1 ? "" : "es"}
              </p>
            )}
          </div>

          <div className="artwork-grid" aria-busy={evidenceLoading}>
            {evidenceLoading && <p className="no-works">Loading evidence…</p>}
            {!evidenceLoading && visibleItems.map((artwork) => (
              <ArtworkCard key={artwork.artworkId} artwork={artwork} />
            ))}
            {!evidenceLoading && selection && strongestEvidenceItems.length === 0 && !loading && (
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
