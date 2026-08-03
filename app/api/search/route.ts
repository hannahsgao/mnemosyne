import { NextRequest, NextResponse } from "next/server";
import type { Artwork, SearchResponse } from "../../../lib/types";

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
  config?: { iiif_url?: string; website_url?: string };
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

function normalizeArtist(value: string | null) {
  if (!value) return "Unknown artist";
  return value.split("\n")[0]?.trim() || "Unknown artist";
}

export async function GET(request: NextRequest) {
  const query = request.nextUrl.searchParams.get("q")?.trim().slice(0, 120);

  if (!query) {
    return NextResponse.json({ error: "Add a search query." }, { status: 400 });
  }

  const url = new URL("https://api.artic.edu/api/v1/artworks/search");
  url.searchParams.set("q", query);
  url.searchParams.set("limit", "100");
  url.searchParams.set("fields", FIELDS);

  try {
    const response = await fetch(url, {
      headers: { "User-Agent": "Mnemosyne prototype (research interface)" },
      next: { revalidate: 60 * 60 * 12 },
    });

    if (!response.ok) throw new Error(`Museum API returned ${response.status}`);

    const payload = (await response.json()) as AicResponse;
    const iiifBase = payload.config?.iiif_url ?? "https://www.artic.edu/iiif/2";
    const websiteBase = (payload.config?.website_url ?? "https://www.artic.edu").replace(
      "http://",
      "https://",
    );

    const artworks: Artwork[] = payload.data
      .filter((work) => work.date_start && work.date_start >= 1000 && work.date_start <= 2030)
      .map((work) => ({
        id: work.id,
        title: work.title,
        artist: normalizeArtist(work.artist_display),
        dateLabel: work.date_display ?? String(work.date_start),
        year: work.date_start,
        imageUrl: work.image_id
          ? `${iiifBase}/${work.image_id}/full/600,/0/default.jpg`
          : null,
        sourceUrl: `${websiteBase}/artworks/${work.id}`,
        publicDomain: work.is_public_domain,
      }));

    const body: SearchResponse = {
      query,
      source: "Art Institute of Chicago public API",
      retrieved: artworks.length,
      totalMatches: payload.pagination?.total ?? artworks.length,
      artworks,
      generatedAt: new Date().toISOString(),
    };

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
