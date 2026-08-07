import { NextRequest, NextResponse } from "next/server";
import {
  configuredSearchServiceUrl,
  DEFAULT_SEARCH_MODE,
  isSearchMode,
  searchServiceEnvironmentName,
  type SearchMode,
} from "../../../lib/search-mode";

const MAX_ARTWORK_ID_LENGTH = 256;
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 10_000;

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
    return NextResponse.json({ error: "Artifact image serving is disabled." }, { status: 404 });
  }
  const requestedSearchMode = request.nextUrl.searchParams.get("searchMode");
  if (requestedSearchMode !== null && !isSearchMode(requestedSearchMode)) {
    return NextResponse.json(
      { error: "searchMode must be keyword or embedding." },
      { status: 400 },
    );
  }
  const searchMode = requestedSearchMode ?? DEFAULT_SEARCH_MODE;
  const artworkId = request.nextUrl.searchParams.get("id") ?? "";
  if (!artworkId || artworkId.length > MAX_ARTWORK_ID_LENGTH || /[\u0000-\u001f]/.test(artworkId)) {
    return NextResponse.json({ error: "Invalid artwork identifier." }, { status: 400 });
  }
  const origin = searchServiceOrigin(searchMode);
  if (!origin) {
    return NextResponse.json(
      {
        error: `Artifact image service is not configured. Set ${searchServiceEnvironmentName(searchMode)}.`,
      },
      { status: 503 },
    );
  }

  try {
    const response = await fetch(`${origin}/v1/images/${encodeURIComponent(artworkId)}`, {
      cache: "no-store",
      headers: { Accept: "image/*", "User-Agent": "Mnemosyne web app" },
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    if (!response.ok || !response.body) {
      return NextResponse.json({ error: "Artwork image was not found." }, { status: 404 });
    }
    const contentType = response.headers.get("content-type") ?? "";
    const contentLength = Number(response.headers.get("content-length"));
    if (
      !contentType.startsWith("image/") ||
      !Number.isSafeInteger(contentLength) ||
      contentLength <= 0 ||
      contentLength > MAX_IMAGE_BYTES
    ) {
      return NextResponse.json({ error: "Artwork image response was invalid." }, { status: 502 });
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
    return NextResponse.json(
      {
        error: "The local artwork image service could not be reached.",
        detail: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 502 },
    );
  }
}
