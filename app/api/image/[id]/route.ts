import { NextRequest, NextResponse } from "next/server";

const IMAGE_ID = /^[a-z0-9-]{20,80}$/i;
const CACHE_SECONDS = 60 * 60 * 24 * 30;
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 10_000;

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;

  if (!IMAGE_ID.test(id)) {
    return NextResponse.json({ error: "Invalid image identifier." }, { status: 400 });
  }

  const sourceUrl = new URL(
    `https://www.artic.edu/iiif/2/${id}/full/600,/0/default.jpg`,
  );

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  try {
    const response = await fetch(sourceUrl, {
      headers: {
        Accept: "image/*",
        "User-Agent": "Mnemosyne web app",
      },
      next: { revalidate: CACHE_SECONDS },
      signal: controller.signal,
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "Artwork image was not available." },
        { status: response.status === 404 ? 404 : 502 },
      );
    }

    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.toLowerCase().startsWith("image/")) {
      return NextResponse.json(
        { error: "Artwork image response was invalid." },
        { status: 502 },
      );
    }

    const image = await response.arrayBuffer();
    if (image.byteLength === 0 || image.byteLength > MAX_IMAGE_BYTES) {
      return NextResponse.json(
        { error: "Artwork image response had an invalid size." },
        { status: 502 },
      );
    }

    return new NextResponse(image, {
      headers: {
        "Cache-Control": `public, max-age=${CACHE_SECONDS}, immutable`,
        "Content-Length": String(image.byteLength),
        "Content-Type": contentType,
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch {
    return NextResponse.json(
      { error: "Artwork image could not be loaded." },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
