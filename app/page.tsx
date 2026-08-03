"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { Timeline } from "../components/Timeline";
import type { Artwork, DecadePoint, SearchResponse } from "../lib/types";

const EXAMPLE_QUERIES = ["horse", "mother and child", "loneliness"];

function decadeFor(work: Artwork) {
  return work.year === null ? null : Math.floor(work.year / 10) * 10;
}

function buildTimeline(artworks: Artwork[]): DecadePoint[] {
  const counts = new Map<number, number>();
  for (const work of artworks) {
    const decade = decadeFor(work);
    if (decade !== null) counts.set(decade, (counts.get(decade) ?? 0) + 1);
  }

  const populated = [...counts.keys()].sort((a, b) => a - b);
  if (!populated.length) return [];

  const start = populated[0];
  const end = populated[populated.length - 1];
  const maxCount = Math.max(...counts.values());
  const points: DecadePoint[] = [];

  for (let decade = start; decade <= end; decade += 10) {
    const count = counts.get(decade) ?? 0;
    points.push({ decade, count, value: count / maxCount });
  }
  return points;
}

function ArtworkCard({ artwork, index }: { artwork: Artwork; index: number }) {
  return (
    <a className="artwork-card" href={artwork.sourceUrl} target="_blank" rel="noreferrer">
      <div className="artwork-image-wrap">
        {artwork.imageUrl ? (
          // Museum IIIF images are deliberately rendered directly so the source remains inspectable.
          // eslint-disable-next-line @next/next/no-img-element
          <img className="artwork-image" src={artwork.imageUrl} alt="" loading="lazy" />
        ) : (
          <div className="image-placeholder">Image unavailable</div>
        )}
        <span className="result-rank">{String(index + 1).padStart(2, "0")}</span>
      </div>
      <div className="artwork-copy">
        <h3>{artwork.title}</h3>
        <p>{artwork.artist}</p>
        <div className="artwork-meta">
          <span>{artwork.dateLabel}</span>
          <span>{artwork.publicDomain ? "Public domain" : "View rights"} ↗</span>
        </div>
      </div>
    </a>
  );
}

export default function Home() {
  const [input, setInput] = useState("horse");
  const [query, setQuery] = useState("horse");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [selectedDecade, setSelectedDecade] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function search(nextQuery: string) {
    const cleanQuery = nextQuery.trim();
    if (!cleanQuery) return;

    setInput(cleanQuery);
    setQuery(cleanQuery);
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(cleanQuery)}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "Search failed.");
      const nextResult = payload as SearchResponse;
      const points = buildTimeline(nextResult.artworks);
      const peak = [...points].sort((a, b) => b.count - a.count)[0];
      setResult(nextResult);
      setSelectedDecade(peak?.decade ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Search failed.");
      setResult(null);
      setSelectedDecade(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void search("horse");
    // The first query is intentional; subsequent searches are user-driven.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const points = useMemo(() => buildTimeline(result?.artworks ?? []), [result]);
  const selectedWorks = useMemo(
    () =>
      (result?.artworks ?? [])
        .filter((work) => decadeFor(work) === selectedDecade)
        .sort((a, b) => Number(Boolean(b.imageUrl)) - Number(Boolean(a.imageUrl))),
    [result, selectedDecade],
  );

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void search(input);
  }

  return (
    <main>
      <nav className="topbar">
        <a className="wordmark" href="#top" aria-label="Mnemosyne home">
          <span className="mark">M</span>
          <span>MNEMOSYNE</span>
        </a>
        <div className="nav-meta">
          <span>Visual culture, indexed</span>
          <a href="#method">Method</a>
        </div>
      </nav>

      <section className="hero" id="top">
        <div className="eyebrow"><span /> PROTOTYPE 01 · MUSEUM COLLECTION SEARCH</div>
        <h1>See an idea move<br />through art history.</h1>
        <p className="hero-deck">
          Search a visual concept, follow its trace across time, and open the artworks behind every point.
        </p>

        <form className="search-form" onSubmit={submit}>
          <label className="sr-only" htmlFor="concept-search">Search visual culture</label>
          <input
            id="concept-search"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Try ‘horse’, ‘grief’, or ‘blue sky’"
            maxLength={120}
          />
          <button type="submit" disabled={loading}>
            {loading ? "Searching…" : "Trace concept"}<span aria-hidden>→</span>
          </button>
        </form>

        <div className="query-row">
          <span>Try</span>
          {EXAMPLE_QUERIES.map((example) => (
            <button key={example} onClick={() => void search(example)}>{example}</button>
          ))}
        </div>
      </section>

      <section className="workspace" aria-live="polite">
        <div className="section-heading">
          <div>
            <span className="kicker">TEMPORAL TRACE</span>
            <h2>{loading ? "Reading the collection…" : `“${query}” across time`}</h2>
          </div>
          {result && (
            <div className="result-stats">
              <span><strong>{result.retrieved}</strong> dated results sampled</span>
              <span><strong>{result.totalMatches.toLocaleString()}</strong> metadata matches</span>
            </div>
          )}
        </div>

        {error && <div className="error-state">{error} Try another query or refresh the page.</div>}
        {loading && <div className="chart-skeleton"><div /><div /><div /></div>}
        {!loading && !error && points.length > 0 && (
          <Timeline points={points} selectedDecade={selectedDecade} onSelect={setSelectedDecade} />
        )}
        {!loading && !error && !points.length && (
          <div className="empty-state">No dated artworks were returned for this query.</div>
        )}

        {selectedDecade !== null && (
          <div className="evidence">
            <div className="evidence-heading">
              <div>
                <span className="kicker">EVIDENCE TRAIL · {selectedDecade}–{selectedDecade + 9}</span>
                <h2>Artworks behind this point</h2>
              </div>
              <p>{selectedWorks.length} retrieved work{selectedWorks.length === 1 ? "" : "s"} in this decade</p>
            </div>
            <div className="artwork-grid">
              {selectedWorks.map((artwork, index) => (
                <ArtworkCard key={artwork.id} artwork={artwork} index={index} />
              ))}
            </div>
          </div>
        )}
      </section>

      <section className="method" id="method">
        <div className="method-intro">
          <span className="kicker">HOW THIS BECOMES THE REAL THING</span>
          <h2>One interface. A progressively better retrieval engine.</h2>
          <p>
            This vertical slice uses museum metadata so the interaction can be tested now. The production engine
            swaps that retrieval layer for image–text embeddings while preserving the timeline and evidence trail.
          </p>
        </div>
        <div className="architecture" aria-label="High-level system architecture">
          <div><span>01</span><strong>Museum APIs + dumps</strong><small>Images, dates, rights, provenance</small></div>
          <i>→</i>
          <div><span>02</span><strong>Canonical corpus</strong><small>Deduplicate, normalize, date-weight</small></div>
          <i>→</i>
          <div><span>03</span><strong>SigLIP 2 index</strong><small>Image embeddings, no fixed labels</small></div>
          <i>→</i>
          <div><span>04</span><strong>Trace + evidence</strong><small>Aggregate, compare, inspect works</small></div>
        </div>
        <div className="prototype-note">
          <strong>Important:</strong> the current line is the distribution of the API’s top metadata results, not a
          historical prevalence claim. It exists to validate the product loop before corpus-scale embedding work.
        </div>
      </section>

      <footer>
        <span>MNEMOSYNE · A MEMORY OF IMAGES</span>
        <span>Prototype collection data via the Art Institute of Chicago</span>
      </footer>
    </main>
  );
}
