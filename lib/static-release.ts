export const IMMUTABLE_RELEASE_PREFIX = "/data/v1/releases/";
export const IMMUTABLE_RELEASE_CACHE_CONTROL =
  "public, max-age=31536000, immutable";

type AssetFetcher = (request: Request) => Promise<Response>;

export async function serveImmutableReleaseAsset(
  request: Request,
  fetchAsset: AssetFetcher,
): Promise<Response | null> {
  if (request.method !== "GET" && request.method !== "HEAD") return null;

  const pathname = new URL(request.url).pathname;
  if (!pathname.startsWith(IMMUTABLE_RELEASE_PREFIX)) return null;

  const response = await fetchAsset(request);
  if (!response.ok && response.status !== 304) return response;

  const headers = new Headers(response.headers);
  headers.set("Cache-Control", IMMUTABLE_RELEASE_CACHE_CONTROL);
  headers.set("X-Content-Type-Options", "nosniff");
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
