import { NextRequest, NextResponse } from "next/server";

const MAX_ARTWORK_ID_LENGTH = 256;
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 10_000;

function searchServiceOrigin() {
  const raw = process.env.MNEMOSYNE_SEARCH_SERVICE_URL?.trim();
  if (!raw) return null;
  try {
    return new URL(raw).origin;
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest) {
  if (process.env.MNEMOSYNE_SEARCH_MODE?.trim() !== "artifact") {
    return NextResponse.json({ error: "Artifact image serving is disabled." }, { status: 404 });
  }
  const artworkId = request.nextUrl.searchParams.get("id") ?? "";
  if (!artworkId || artworkId.length > MAX_ARTWORK_ID_LENGTH || /[\u0000-\u001f]/.test(artworkId)) {
    return NextResponse.json({ error: "Invalid artwork identifier." }, { status: 400 });
  }
  const origin = searchServiceOrigin();
  if (!origin) {
    return NextResponse.json({ error: "Artifact search service is not configured." }, { status: 503 });
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
