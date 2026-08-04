import { NextRequest, NextResponse } from "next/server";

const IMAGE_ID = /^[a-z0-9-]{20,80}$/i;
const CACHE_SECONDS = 60 * 60 * 24 * 30;

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
          cache: "no-store",
          headers: {
            Accept: "image/jpeg",
            "User-Agent": "Mnemosyne prototype (research interface)",
          },
        },
      );

      if (!response.ok) continue;

      const candidate = await response.arrayBuffer();
      if (!candidate.byteLength) continue;

      image = candidate;
      contentType = response.headers.get("content-type") ?? contentType;
      break;
    }

    if (!image) throw new Error("Museum image service did not return a usable image");

    return new NextResponse(image, {
      headers: {
        "Cache-Control": `public, max-age=${CACHE_SECONDS}, immutable`,
        "Content-Type": contentType,
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
