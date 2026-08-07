"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { Timeline } from "../components/Timeline";
import { MAX_QUERY_LENGTH, parseConceptQuery, QuerySyntaxError } from "../lib/query";
import { formatTimelineYear, peakSelection, pointForBin, timelineWindow } from "../lib/timeline";
import type {
  ChartSelection,
  EvidenceArtwork,
  EvidenceSliceName,
  SearchResponse,
} from "../lib/types";

const INITIAL_QUERY = "horse, ship";
const EXAMPLE_QUERIES = ["horse, ship", '"still life, fruit", flowers', "loneliness, joy"];
const INITIAL_VISIBLE_WORKS = 5;
const EVIDENCE_ORDER: EvidenceSliceName[] = [
  "strongest",
  "representative",
  "borderline",
  "randomContributors",
  "bestNonContributors",
  "randomDenominator",
];

const EVIDENCE_LABELS: Partial<Record<EvidenceSliceName, string>> = {
  representative: "Representative",
  borderline: "Near threshold",
  bestNonContributors: "Best available match",
  randomDenominator: "Corpus sample",
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

function selectedEvidenceItems(response: SearchResponse | null, selection: ChartSelection | null) {
  if (
    !response?.selectedEvidence ||
    !selection ||
    response.selectedEvidence.queryId !== selection.queryId ||
    response.selectedEvidence.binKey !== selection.binKey
  ) {
    return [];
  }

  const slices = response.selectedEvidence.slices;
  const seen = new Set<string>();
  const selected: { artwork: EvidenceArtwork; slice: EvidenceSliceName }[] = [];
  const largestSlice = Math.max(...EVIDENCE_ORDER.map((slice) => slices[slice].length));
  for (let index = 0; index < largestSlice; index += 1) {
    for (const slice of EVIDENCE_ORDER) {
      const artwork = slices[slice][index];
      if (!artwork || seen.has(artwork.artworkId)) continue;
      seen.add(artwork.artworkId);
      selected.push({ artwork, slice });
    }
  }
  return selected;
}

function ArtworkCard({ artwork, slice }: { artwork: EvidenceArtwork; slice: EvidenceSliceName }) {
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
        {EVIDENCE_LABELS[slice] && <em>{EVIDENCE_LABELS[slice]}</em>}
      </div>
    </a>
  );
}

function buildSearchUrl(query: string, selection?: ChartSelection) {
  const params = new URLSearchParams({ q: query });
  if (selection) {
    params.set("evidenceQueryId", selection.queryId);
    params.set("evidenceBinKey", selection.binKey);
  }
  return `/api/search?${params.toString()}`;
}

export default function Home() {
  const [input, setInput] = useState(INITIAL_QUERY);
  const [submittedQuery, setSubmittedQuery] = useState(INITIAL_QUERY);
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

  async function search(nextQuery: string) {
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
    setInput(nextQuery.trim());
    setSubmittedQuery(nextQuery.trim());
    setLoading(true);
    setEvidenceLoading(false);
    setError(null);
    setSelection(null);
    setShowAllExamples(false);
    setHiddenQueryIds(new Set());

    try {
      const response = await fetch(buildSearchUrl(nextQuery.trim()), {
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
      const response = await fetch(buildSearchUrl(submittedQuery, nextSelection), {
        signal: controller.signal,
      });
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
      setResult((current) => (current ? { ...current, selectedEvidence: payload.selectedEvidence } : current));
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      if (evidenceRequestId.current !== currentRequest) return;
      setError(caught instanceof Error ? caught.message : "Evidence could not be loaded.");
    } finally {
      if (evidenceRequestId.current === currentRequest) setEvidenceLoading(false);
    }
  }

  useEffect(() => {
    void search(INITIAL_QUERY);
    return () => {
      searchAbort.current?.abort();
      evidenceAbort.current?.abort();
    };
    // The first query is intentional; subsequent searches are user-driven.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedItems = useMemo(
    () => selectedEvidenceItems(result, selection),
    [result, selection],
  );
  const visibleItems = showAllExamples
    ? selectedItems
    : selectedItems.slice(0, INITIAL_VISIBLE_WORKS);
  const hiddenExampleCount = Math.max(0, selectedItems.length - INITIAL_VISIBLE_WORKS);
  const selectedQuery = result?.queries.find((query) => query.id === selection?.queryId) ?? null;
  const selectedBin = result?.bins.find((bin) => bin.key === selection?.binKey) ?? null;
  const selectedSeries = result?.series.find((series) => series.queryId === selection?.queryId) ?? null;
  const selectedPoint =
    selectedSeries && selection ? pointForBin(selectedSeries, selection.binKey) : null;
  const displayedBins = result?.bins.length ? timelineWindow(result.bins) : [];
  const yearRange = displayedBins.length
    ? `${formatTimelineYear(displayedBins[0].start)}–${formatTimelineYear(displayedBins[displayedBins.length - 1].end)}`
    : "";

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void search(input);
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
            <p>Compare up to five visual concepts. Separate lines with commas; quote a literal comma.</p>
          </div>

          <form className="search-form" onSubmit={submit}>
            <label className="sr-only" htmlFor="concept-search">Compare visual concepts</label>
            <input
              id="concept-search"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder='horse, ship, "still life, fruit"'
              maxLength={MAX_QUERY_LENGTH}
              aria-describedby="query-help"
            />
            <button type="submit" disabled={loading}>{loading ? "Searching…" : "Search"}</button>
          </form>

          <div className="query-row" id="query-help">
            <span>Try:</span>
            {EXAMPLE_QUERIES.map((example) => (
              <button key={example} type="button" onClick={() => void search(example)}>{example}</button>
            ))}
          </div>
        </section>

        <section className="results" aria-live="polite" aria-busy={loading}>
          <div className="results-heading">
            <div>
              <h2>{loading ? "Searching the collection…" : result?.metric.label ?? "Results over time"}</h2>
              {!loading && yearRange && <span>{yearRange}</span>}
            </div>
            {result && !loading && (
              <p>{result.queries.length} line{result.queries.length === 1 ? "" : "s"} · {result.corpus.label}</p>
            )}
          </div>

          {error && !result && <div className="message-state">{error} Try another search.</div>}
          {loading && <div className="chart-skeleton" />}
          {!loading && result && result.bins.length > 0 && (
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
        </section>

        <section className="evidence" aria-label="Evidence for the selected chart point">
          <div className="evidence-heading">
            <h2>
              {!selection || !selectedQuery || !selectedBin
                ? "Select a line and period"
                : `${selectedQuery.label} · ${selectedBin.label}`}
            </h2>
            {selectedPoint && (
              <p>{selectedPoint.objectCount} contributing work{selectedPoint.objectCount === 1 ? "" : "s"}</p>
            )}
          </div>

          <div className="artwork-grid" aria-busy={evidenceLoading}>
            {evidenceLoading && <p className="no-works">Loading evidence…</p>}
            {!evidenceLoading && visibleItems.map(({ artwork, slice }) => (
              <ArtworkCard key={artwork.artworkId} artwork={artwork} slice={slice} />
            ))}
            {!evidenceLoading && selection && selectedItems.length === 0 && !loading && (
              <p className="no-works">No contributing works in this period. Context samples may be unavailable.</p>
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
                ? "Show fewer examples"
                : `View ${hiddenExampleCount} more example${hiddenExampleCount === 1 ? "" : "s"}`}
            </button>
          )}
        </section>
      </div>
    </main>
  );
}
