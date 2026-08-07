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

  const sourceUrl = new URL(
    `https://www.artic.edu/iiif/2/${id}/full/600,/0/default.jpg`,
  );

  return NextResponse.redirect(sourceUrl, {
    status: 307,
    headers: { "Cache-Control": `public, max-age=${CACHE_SECONDS}, immutable` },
  });
}
