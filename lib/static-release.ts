export const CATALOG_ROUTE_PREFIX = "/catalog-data/v1/";
export const CATALOG_ASSET_PREFIX = "/data/v1/";
export const IMMUTABLE_RELEASE_PREFIX = `${CATALOG_ROUTE_PREFIX}releases/`;
export const IMMUTABLE_RELEASE_CACHE_CONTROL =
  "public, max-age=31536000, immutable";

type AssetFetcher = (request: Request) => Promise<Response>;

export async function serveCatalogAsset(
  request: Request,
  fetchAsset: AssetFetcher,
): Promise<Response | null> {
  if (request.method !== "GET" && request.method !== "HEAD") return null;

  const pathname = new URL(request.url).pathname;
  if (!pathname.startsWith(CATALOG_ROUTE_PREFIX)) return null;

  const assetUrl = new URL(request.url);
  assetUrl.pathname = `${CATALOG_ASSET_PREFIX}${pathname.slice(
    CATALOG_ROUTE_PREFIX.length,
  )}`;
  const response = await fetchAsset(
    new Request(assetUrl, {
      method: request.method,
      headers: request.headers,
    }),
  );
  if (!pathname.startsWith(IMMUTABLE_RELEASE_PREFIX)) return response;
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
