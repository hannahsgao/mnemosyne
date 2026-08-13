const VISUAL_API_PATHS = new Set(["/api/search", "/api/evidence"]);

export interface RateLimitBinding {
  limit(options: { key: string }): Promise<{ success: boolean }>;
}

export function isVisualApiRequest(
  request: Pick<Request, "method" | "url">,
): boolean {
  if (request.method !== "GET") return false;
  const url = new URL(request.url);
  return (
    VISUAL_API_PATHS.has(url.pathname) &&
    url.searchParams.get("searchMode") === "embedding"
  );
}

export async function checkVisualRateLimit(
  request: Request,
  limiter?: RateLimitBinding,
): Promise<Response | null> {
  if (!limiter || !isVisualApiRequest(request)) return null;

  // Cloudflare sets this header at its edge. Do not trust a browser-controlled
  // forwarded-for header when selecting the limiter key.
  const clientIp = request.headers.get("CF-Connecting-IP")?.trim();
  if (!clientIp) return null;

  try {
    const { success } = await limiter.limit({
      key: `visual:${clientIp}`,
    });
    if (success) return null;
  } catch {
    // Search remains available if a hosting environment omits or temporarily
    // cannot reach its optional rate-limit binding.
    return null;
  }

  return Response.json(
    { error: "Too many Visual requests. Please try again in a minute." },
    {
      status: 429,
      headers: {
        "Cache-Control": "no-store",
        "Retry-After": "60",
      },
    },
  );
}
