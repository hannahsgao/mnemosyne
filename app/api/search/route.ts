import { NextRequest, NextResponse } from "next/server";
import { parseConceptQuery, QuerySyntaxError } from "../../../lib/query";
import {
  buildBackendImageUrl,
  configuredSearchServiceUrl,
  DEFAULT_SEARCH_MODE,
  isSearchMode,
  searchServiceEnvironmentName,
  type SearchMode,
} from "../../../lib/search-mode";
import { buildRelativeDensityPoints, buildSharedDecadeBins, decadeKey, decadeStart } from "../../../lib/timeline";
import type {
  EvidenceArtwork,
  EvidenceSlices,
  QueryDescriptor,
  SearchResponse,
  SearchSeries,
} from "../../../lib/types";

type AicArtwork = {
  id: number;
  title: string;
  artist_display: string | null;
  date_display: string | null;
  date_start: number | null;
  image_id: string | null;
  is_public_domain: boolean;
};

type AicResponse = {
  data: AicArtwork[];
  pagination?: { total?: number };
  config?: { website_url?: string };
};

type PrototypeQueryResult = {
  query: QueryDescriptor;
  artworks: EvidenceArtwork[];
  totalMatches: number;
};

const FIELDS = [
  "id",
  "title",
  "artist_display",
  "date_display",
  "date_start",
  "image_id",
  "is_public_domain",
].join(",");
const UPSTREAM_TIMEOUT_MS = 15_000;

function rewriteBackendImageUrls(payload: unknown, searchMode: SearchMode) {
  if (!payload || typeof payload !== "object") return payload;
  const rewriteItems = (items: unknown) => {
    if (!Array.isArray(items)) return;
    for (const item of items) {
      if (!item || typeof item !== "object") continue;
      const artwork = item as { imageUrl?: unknown };
      if (typeof artwork.imageUrl !== "string" || !artwork.imageUrl.startsWith("/v1/images/")) {
        continue;
      }
      try {
        const artworkId = decodeURIComponent(artwork.imageUrl.slice("/v1/images/".length));
        artwork.imageUrl = buildBackendImageUrl(artworkId, searchMode);
      } catch {
        artwork.imageUrl = "";
      }
    }
  };

  const selected = (payload as { selectedEvidence?: unknown }).selectedEvidence;
  if (selected && typeof selected === "object") {
    const slices = (selected as { slices?: unknown }).slices;
    if (slices && typeof slices === "object") {
      for (const items of Object.values(slices)) rewriteItems(items);
    }
  }
  return payload;
}

function normalizeArtist(value: string | null) {
  if (!value) return "Unknown artist";
  return value.split("\n")[0]?.trim() || "Unknown artist";
}

function emptyEvidenceSlices(): EvidenceSlices {
  return {
    strongest: [],
    representative: [],
    borderline: [],
    randomContributors: [],
    bestNonContributors: [],
    randomDenominator: [],
  };
}

async function proxySearchService(
  request: NextRequest,
  serviceUrl: string,
  serviceEnvironmentName: string,
  query: string,
  searchMode: SearchMode,
) {
  let target: URL;
  try {
    target = new URL(serviceUrl);
    if (target.protocol !== "http:" && target.protocol !== "https:") throw new Error();
    if (target.pathname === "/") target.pathname = "/v1/search";
  } catch {
    return NextResponse.json(
      { error: `${serviceEnvironmentName} is not a valid URL.` },
      { status: 500 },
    );
  }

  try {
    const response = await fetch(target, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mnemosyne web app",
      },
      body: JSON.stringify({
        query,
        selectedQueryId: request.nextUrl.searchParams.get("evidenceQueryId"),
        selectedBinKey: request.nextUrl.searchParams.get("evidenceBinKey"),
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const body: unknown = await response.json();
      return NextResponse.json(rewriteBackendImageUrls(body, searchMode), {
        status: response.status,
        headers: { "Cache-Control": response.headers.get("cache-control") ?? "no-store" },
      });
    }
    return new NextResponse(await response.text(), {
      status: response.status,
      headers: { "Cache-Control": "no-store", "Content-Type": contentType || "text/plain" },
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "The local search service could not be reached.",
        detail: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 502 },
    );
  }
}

async function searchAic(query: QueryDescriptor): Promise<PrototypeQueryResult> {
  const url = new URL("https://api.artic.edu/api/v1/artworks/search");
  url.searchParams.set("q", query.label);
  url.searchParams.set("limit", "100");
  url.searchParams.set("fields", FIELDS);

  const response = await fetch(url, {
    headers: { "User-Agent": "Mnemosyne prototype (research interface)" },
    next: { revalidate: 60 * 60 * 12 },
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(`Museum API returned ${response.status}`);

  const payload = (await response.json()) as AicResponse;
  const websiteBase = (payload.config?.website_url ?? "https://www.artic.edu").replace(
    "http://",
    "https://",
  );

  const artworks = payload.data
    .filter((work) => work.date_start && work.date_start >= 1000 && work.date_start <= 2030)
    .map<EvidenceArtwork>((work) => ({
      artworkId: `aic:${work.id}`,
      physicalObjectId: `aic:${work.id}`,
      visualClusterId: `aic:${work.id}`,
      institution: "Art Institute of Chicago",
      title: work.title,
      artist: normalizeArtist(work.artist_display),
      dateDisplay: work.date_display ?? String(work.date_start),
      dateStart: work.date_start,
      dateEnd: work.date_start,
      dateQualifier: "catalogue display date",
      imageUrl: work.image_id ? `/api/image/${work.image_id}?v=proxy-1` : null,
      sourceRecordUrl: `${websiteBase}/artworks/${work.id}`,
      metadataLicense: "CC0 metadata",
      imageRightsUri: "",
      creditLine: "",
      publicDomain: work.is_public_domain,
      rawScore: null,
      contributionWeight: 1,
      contributor: true,
    }));

  return {
    query,
    artworks,
    totalMatches: payload.pagination?.total ?? artworks.length,
  };
}

function buildPrototypeResponse(
  results: PrototypeQueryResult[],
  evidenceQueryId: string | null,
  evidenceBinKey: string | null,
): SearchResponse {
  const bins = buildSharedDecadeBins(
    results.flatMap((result) => result.artworks.map((artwork) => artwork.dateStart)),
  );
  const series: SearchSeries[] = results.map((result) => ({
    queryId: result.query.id,
    k: result.artworks.length,
    threshold: null,
    lowSignal: null,
    diagnostics: {
      standardizedSeparation: null,
      controlMean: null,
      controlStdDev: null,
      promptTopKJaccard: null,
      reasons: ["This line reflects a live catalogue metadata sample, not embedding retrieval."],
    },
    points: buildRelativeDensityPoints(
      bins,
      result.artworks.map((artwork) => artwork.dateStart),
    ),
    totalMatches: result.totalMatches,
  }));

  const selectedSeries =
    series.find((candidate) => candidate.queryId === evidenceQueryId) ?? series[0] ?? null;
  const selectedBin =
    (evidenceBinKey && bins.some((bin) => bin.key === evidenceBinKey)
      ? evidenceBinKey
      : selectedSeries?.points.length
        ? selectedSeries.points.reduce((best, point) =>
            point.value > best.value ? point : best,
          ).binKey
        : null) ?? null;

  let selectedEvidence: SearchResponse["selectedEvidence"] = null;
  if (selectedSeries && selectedBin) {
    const source = results.find((result) => result.query.id === selectedSeries.queryId);
    const slices = emptyEvidenceSlices();
    slices.strongest = (source?.artworks ?? [])
      .filter(
        (artwork) =>
          artwork.dateStart !== null &&
          decadeKey(decadeStart(artwork.dateStart)) === selectedBin,
      )
      .sort((left, right) => Number(Boolean(right.imageUrl)) - Number(Boolean(left.imageUrl)))
      .slice(0, 12);
    selectedEvidence = { queryId: selectedSeries.queryId, binKey: selectedBin, slices };
  }

  return {
    schemaVersion: "mnemosyne.search.v1",
    queries: results.map((result) => result.query),
    corpus: {
      id: "aic-live-metadata",
      version: "v1",
      label: "Art Institute of Chicago live metadata search",
      count: null,
      countingUnit: "physical-object",
      view: "dated search results",
      filters: { date: ["1000–2030"] },
    },
    model: {
      id: "aic-catalogue-search",
      version: "live",
      promptTemplateVersion: "none",
    },
    metric: {
      id: "prototype-relative-result-density",
      version: "1",
      label: "Relative result density",
      percentile: null,
      unit: "relative-density",
      description: "Result count per decade divided by the largest decade count for that query.",
    },
    bins,
    series,
    selectedEvidence,
    warnings: [
      "Prototype mode: lines show the distribution of top catalogue metadata results, not historical prevalence or 1% concentration lift.",
    ],
    generatedAt: new Date().toISOString(),
  };
}

export async function GET(request: NextRequest) {
  const input = request.nextUrl.searchParams.get("q") ?? "";
  const requestedSearchMode = request.nextUrl.searchParams.get("searchMode");
  if (requestedSearchMode !== null && !isSearchMode(requestedSearchMode)) {
    return NextResponse.json(
      { error: "searchMode must be keyword or embedding." },
      { status: 400 },
    );
  }
  const searchMode = requestedSearchMode ?? DEFAULT_SEARCH_MODE;
  let parsed;
  try {
    parsed = parseConceptQuery(input);
  } catch (error) {
    if (error instanceof QuerySyntaxError) {
      return NextResponse.json({ error: error.message, code: error.code }, { status: 400 });
    }
    throw error;
  }

  const mode = process.env.MNEMOSYNE_SEARCH_MODE?.trim();
  if (mode === "artifact") {
    const serviceUrl = configuredSearchServiceUrl(searchMode, process.env);
    const environmentName = searchServiceEnvironmentName(searchMode);
    if (!serviceUrl) {
      return NextResponse.json(
        {
          error: `${searchMode === "embedding" ? "Embedding" : "Keyword"} search is not configured. Set ${environmentName}.`,
        },
        { status: 503 },
      );
    }
    return proxySearchService(request, serviceUrl, environmentName, input, searchMode);
  }
  if (mode !== "catalogue-demo") {
    return NextResponse.json(
      {
        error:
          "Search mode is not configured. Set MNEMOSYNE_SEARCH_MODE to artifact or catalogue-demo.",
      },
      { status: 503 },
    );
  }
  if (searchMode === "embedding") {
    return NextResponse.json(
      {
        error:
          "Embedding search is unavailable in catalogue-demo mode. Configure both artifact search services to use the toggle.",
      },
      { status: 503 },
    );
  }

  const queries: QueryDescriptor[] = parsed.map((query, index) => ({
    id: `q-${index + 1}`,
    ...query,
  }));

  try {
    const results = await Promise.all(queries.map(searchAic));
    const body = buildPrototypeResponse(
      results,
      request.nextUrl.searchParams.get("evidenceQueryId"),
      request.nextUrl.searchParams.get("evidenceBinKey"),
    );

    return NextResponse.json(body, {
      headers: { "Cache-Control": "public, s-maxage=43200, stale-while-revalidate=86400" },
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "The museum collection could not be reached. Please try again shortly.",
        detail: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 502 },
    );
  }
}
