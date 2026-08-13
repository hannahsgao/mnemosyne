import { NextRequest, NextResponse } from "next/server";
import {
  isTimeoutError,
  NO_STORE_CACHE_CONTROL,
  upstreamHeaders,
  upstreamTimeoutMs,
  visualProxyError,
} from "../../../lib/embedding-proxy";
import {
  configuredSearchServiceUrl,
  DEFAULT_SEARCH_MODE,
  isSearchMode,
  searchServiceEnvironmentName,
  type SearchMode,
} from "../../../lib/search-mode";

const MAX_ARTWORK_ID_LENGTH = 256;
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;

function noStoreJson(body: unknown, status: number) {
  return NextResponse.json(body, {
    status,
    headers: { "Cache-Control": NO_STORE_CACHE_CONTROL },
  });
}

function visualErrorResponse(status: number, timedOut = false) {
  const failure = visualProxyError(status, timedOut);
  return noStoreJson(failure.body, failure.status);
}

function searchServiceOrigin(searchMode: SearchMode) {
  const raw = configuredSearchServiceUrl(searchMode, process.env);
  if (!raw) return null;
  try {
    const serviceUrl = new URL(raw);
    if (serviceUrl.protocol !== "http:" && serviceUrl.protocol !== "https:") return null;
    return serviceUrl.origin;
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest) {
  if (process.env.MNEMOSYNE_SEARCH_MODE?.trim() !== "artifact") {
    return noStoreJson({ error: "Artifact image serving is disabled." }, 404);
  }
  const requestedSearchMode = request.nextUrl.searchParams.get("searchMode");
  if (requestedSearchMode !== null && !isSearchMode(requestedSearchMode)) {
    return NextResponse.json(
      { error: "searchMode must be keyword or embedding." },
      { status: 400, headers: { "Cache-Control": NO_STORE_CACHE_CONTROL } },
    );
  }
  const searchMode = requestedSearchMode ?? DEFAULT_SEARCH_MODE;
  const artworkId = request.nextUrl.searchParams.get("id") ?? "";
  if (!artworkId || artworkId.length > MAX_ARTWORK_ID_LENGTH || /[\u0000-\u001f]/.test(artworkId)) {
    return noStoreJson({ error: "Invalid artwork identifier." }, 400);
  }
  const origin = searchServiceOrigin(searchMode);
  if (!origin) {
    return NextResponse.json(
      {
        error: `Artifact image service is not configured. Set ${searchServiceEnvironmentName(searchMode)}.`,
      },
      { status: 503, headers: { "Cache-Control": NO_STORE_CACHE_CONTROL } },
    );
  }

  try {
    const response = await fetch(`${origin}/v1/images/${encodeURIComponent(artworkId)}`, {
      cache: "no-store",
      headers: upstreamHeaders(searchMode, process.env, "image/*"),
      signal: AbortSignal.timeout(upstreamTimeoutMs(searchMode, process.env)),
    });
    if (!response.ok || !response.body) {
      if (searchMode === "embedding" && response.status !== 404) {
        return visualErrorResponse(response.status);
      }
      return noStoreJson({ error: "Artwork image was not found." }, 404);
    }
    const contentType = response.headers.get("content-type") ?? "";
    const contentLength = Number(response.headers.get("content-length"));
    if (
      !contentType.startsWith("image/") ||
      !Number.isSafeInteger(contentLength) ||
      contentLength <= 0 ||
      contentLength > MAX_IMAGE_BYTES
    ) {
      return noStoreJson({ error: "Artwork image response was invalid." }, 502);
    }
    return new NextResponse(response.body, {
      headers: {
        "Cache-Control": "public, max-age=31536000, immutable",
        "Content-Length": String(contentLength),
        "Content-Type": contentType,
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch (error) {
    if (searchMode === "embedding") return visualErrorResponse(502, isTimeoutError(error));
    return noStoreJson({ error: "The metadata artwork image service could not be reached." }, 502);
  }
}
