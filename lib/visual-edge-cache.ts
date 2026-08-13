import { EMBEDDING_CACHE_CONTROL } from "./embedding-proxy.ts";

export const VISUAL_EDGE_CACHE_HEADER = "X-Mnemosyne-Edge-Cache";

const VISUAL_CACHE_PATHS = new Set(["/api/search", "/api/evidence"]);
const VISUAL_CACHE_VERSION = "met142482-siglip2-75de-v1";

export function visualEdgeCacheKey(
  request: Pick<Request, "method" | "url">,
): Request | null {
  if (request.method !== "GET") return null;
  const url = new URL(request.url);
  if (
    !VISUAL_CACHE_PATHS.has(url.pathname) ||
    url.searchParams.get("searchMode") !== "embedding"
  ) {
    return null;
  }

  // The complete public query string remains in the key. The private suffix
  // prevents a model/corpus release from inheriting a previous deployment's
  // cached response when this constant is bumped alongside the release.
  url.searchParams.set("__mnemosyne_edge_cache", VISUAL_CACHE_VERSION);
  return new Request(url, { method: "GET" });
}

export function shouldStoreVisualEdgeResponse(
  response: Pick<Response, "headers" | "ok">,
): boolean {
  return (
    response.ok &&
    response.headers.get("Cache-Control") === EMBEDDING_CACHE_CONTROL &&
    !response.headers.has("Set-Cookie")
  );
}

export function withVisualEdgeCacheStatus(
  response: Response,
  status: "BYPASS" | "HIT" | "MISS",
): Response {
  const headers = new Headers(response.headers);
  headers.set(VISUAL_EDGE_CACHE_HEADER, status);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
