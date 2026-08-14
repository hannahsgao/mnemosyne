import { NextRequest, NextResponse } from "next/server";
import { parseConceptQuery, QuerySyntaxError } from "../../../lib/query";
import {
  isTimeoutError,
  NO_STORE_CACHE_CONTROL,
  outerCacheControl,
  safeRetryAfter,
  upstreamHeaders,
  upstreamTimeoutMs,
  visualProxyError,
} from "../../../lib/embedding-proxy";
import {
  buildBackendImageUrl,
  configuredSearchServiceUrl,
  DEFAULT_SEARCH_MODE,
  isSearchMode,
  searchServiceEnvironmentName,
  type SearchMode,
} from "../../../lib/search-mode";
import type { SelectedEvidence } from "../../../lib/types";
import { GET as search } from "../search/route";

type EvidenceEnvelope = {
  schemaVersion: "mnemosyne.evidence.v1";
  selectedEvidence: SelectedEvidence | null;
  generatedAt: string;
};

function evidenceServiceUrl(searchServiceUrl: string) {
  const target = new URL(searchServiceUrl);
  if (target.protocol !== "http:" && target.protocol !== "https:") throw new Error();
  if (target.pathname === "/") target.pathname = "/v1/evidence";
  else if (target.pathname.endsWith("/v1/search")) {
    target.pathname = `${target.pathname.slice(0, -"/v1/search".length)}/v1/evidence`;
  } else if (target.pathname.endsWith("/search")) {
    target.pathname = `${target.pathname.slice(0, -"/search".length)}/evidence`;
  } else {
    target.pathname = `${target.pathname.replace(/\/$/, "")}/v1/evidence`;
  }
  target.search = "";
  target.hash = "";
  return target;
}

function rewriteImageUrls(selectedEvidence: SelectedEvidence | null, searchMode: SearchMode) {
  if (!selectedEvidence) return selectedEvidence;
  for (const artworks of Object.values(selectedEvidence.slices)) {
    for (const artwork of artworks) {
      if (!artwork.imageUrl?.startsWith("/v1/images/")) continue;
      try {
        const artworkId = decodeURIComponent(artwork.imageUrl.slice("/v1/images/".length));
        artwork.imageUrl = buildBackendImageUrl(artworkId, searchMode);
      } catch {
        artwork.imageUrl = null;
      }
    }
  }
  return selectedEvidence;
}

function errorMessage(payload: unknown, fallback: string) {
  return payload && typeof payload === "object" && "error" in payload
    ? String(payload.error)
    : fallback;
}

function noStoreJson(body: unknown, status: number) {
  return NextResponse.json(body, {
    status,
    headers: { "Cache-Control": NO_STORE_CACHE_CONTROL },
  });
}

function visualErrorResponse(status: number, retryAfter: string | null = null, timedOut = false) {
  const failure = visualProxyError(status, timedOut);
  const headers: Record<string, string> = { "Cache-Control": NO_STORE_CACHE_CONTROL };
  const safeRetry = safeRetryAfter(retryAfter);
  if (safeRetry) headers["Retry-After"] = safeRetry;
  return NextResponse.json(failure.body, { status: failure.status, headers });
}

export async function GET(request: NextRequest) {
  const input = request.nextUrl.searchParams.get("q") ?? "";
  const requestedMode = request.nextUrl.searchParams.get("searchMode");
  if (requestedMode !== null && !isSearchMode(requestedMode)) {
    return noStoreJson({ error: "searchMode must be keyword or embedding." }, 400);
  }
  const searchMode = requestedMode ?? DEFAULT_SEARCH_MODE;
  try {
    parseConceptQuery(input);
  } catch (error) {
    if (error instanceof QuerySyntaxError) {
      return noStoreJson({ error: error.message, code: error.code }, 400);
    }
    throw error;
  }

  const mode = process.env.MNEMOSYNE_SEARCH_MODE?.trim();
  if (mode === "artifact") {
    const configured = configuredSearchServiceUrl(searchMode, process.env);
    const environmentName = searchServiceEnvironmentName(searchMode);
    if (!configured) {
      return NextResponse.json(
        { error: `${SEARCH_MODE_NAME[searchMode]} search is not configured. Set ${environmentName}.` },
        { status: 503, headers: { "Cache-Control": NO_STORE_CACHE_CONTROL } },
      );
    }
    let target: URL;
    try {
      target = evidenceServiceUrl(configured);
    } catch {
      return noStoreJson({ error: `${environmentName} is not a valid URL.` }, 500);
    }
    try {
      const response = await fetch(target, {
        method: "POST",
        headers: upstreamHeaders(searchMode, process.env),
        body: JSON.stringify({
          query: input,
          selectedQueryId: request.nextUrl.searchParams.get("evidenceQueryId"),
          selectedBinKey: request.nextUrl.searchParams.get("evidenceBinKey"),
        }),
        cache: "no-store",
        signal: AbortSignal.timeout(upstreamTimeoutMs(searchMode, process.env)),
      });
      let payload: unknown = null;
      try {
        payload = await response.json();
      } catch {
        // Handled as a safe unsupported/unavailable response below.
      }
      if (!response.ok) {
        if (searchMode === "embedding") {
          return visualErrorResponse(response.status, response.headers.get("retry-after"));
        }
        return noStoreJson(
          { error: errorMessage(payload, "Evidence could not be loaded.") },
          response.status,
        );
      }
      if (
        !payload ||
        typeof payload !== "object" ||
        (payload as Partial<EvidenceEnvelope>).schemaVersion !== "mnemosyne.evidence.v1"
      ) {
        return searchMode === "embedding"
          ? visualErrorResponse(502)
          : noStoreJson({ error: "The evidence service returned an unsupported response." }, 502);
      }
      const envelope = payload as EvidenceEnvelope;
      return NextResponse.json({
        ...envelope,
        selectedEvidence: rewriteImageUrls(envelope.selectedEvidence, searchMode),
      }, {
        headers: {
          "Cache-Control": outerCacheControl(
            searchMode,
            response.status,
            response.headers.get("cache-control"),
          ),
        },
      });
    } catch (error) {
      if (searchMode === "embedding") return visualErrorResponse(502, null, isTimeoutError(error));
      return noStoreJson({ error: "The metadata evidence service could not be reached." }, 502);
    }
  }

  if (mode === "catalogue-demo" && searchMode === "keyword") {
    // The demo has no evidence-only upstream; preserve it by adapting the
    // existing response while keeping the public page on one envelope shape.
    const response = await search(request);
    const payload: unknown = await response.json();
    if (!response.ok) {
      return NextResponse.json(
        { error: errorMessage(payload, "Evidence could not be loaded.") },
        { status: response.status, headers: { "Cache-Control": NO_STORE_CACHE_CONTROL } },
      );
    }
    const selectedEvidence =
      payload && typeof payload === "object" && "selectedEvidence" in payload
        ? (payload.selectedEvidence as SelectedEvidence | null)
        : null;
    return NextResponse.json<EvidenceEnvelope>({
      schemaVersion: "mnemosyne.evidence.v1",
      selectedEvidence,
      generatedAt: new Date().toISOString(),
    });
  }

  return NextResponse.json(
    { error: "Evidence search is not configured for this mode." },
    { status: 503, headers: { "Cache-Control": NO_STORE_CACHE_CONTROL } },
  );
}

const SEARCH_MODE_NAME: Record<SearchMode, string> = {
  embedding: "Visual",
  keyword: "Metadata",
};
