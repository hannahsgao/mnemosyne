export type Artwork = {
  id: number;
  title: string;
  artist: string;
  dateLabel: string;
  year: number | null;
  imageUrl: string | null;
  sourceUrl: string;
  publicDomain: boolean;
};

export type SearchResponse = {
  query: string;
  source: string;
  retrieved: number;
  totalMatches: number;
  artworks: Artwork[];
  generatedAt: string;
};

export type DecadePoint = {
  decade: number;
  count: number;
  value: number;
};
