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

  try {
    let image: ArrayBuffer | null = null;
    let contentType = "image/jpeg";

    // Most cards only need 600px. The museum occasionally returns an empty
    // first response for that derivative, so fall back to its standard 843px size.
    for (const width of [600, 843]) {
      const response = await fetch(
        `https://www.artic.edu/iiif/2/${id}/full/${width},/0/default.jpg`,
        {
          next: { revalidate: CACHE_SECONDS },
          headers: {
            Accept: "image/jpeg",
            "User-Agent": "Mnemosyne prototype (research interface)",
          },
          signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
        },
      );

      if (!response.ok) continue;
      const candidateType = response.headers.get("content-type") ?? "";
      const declaredLength = Number(response.headers.get("content-length") ?? 0);
      if (
        !candidateType.startsWith("image/") ||
        (declaredLength > 0 && declaredLength > MAX_IMAGE_BYTES)
      ) {
        continue;
      }

      const candidate = await response.arrayBuffer();
      if (!candidate.byteLength || candidate.byteLength > MAX_IMAGE_BYTES) continue;

      image = candidate;
      contentType = candidateType;
      break;
    }

    if (!image) throw new Error("Museum image service did not return a usable image");

    return new NextResponse(image, {
      headers: {
        "Cache-Control": `public, max-age=${CACHE_SECONDS}, immutable`,
        "Content-Type": contentType,
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "The artwork image could not be loaded.",
        detail: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 502 },
    );
  }
}
