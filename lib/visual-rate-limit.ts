const VISUAL_API_PATHS = new Set(["/api/search", "/api/evidence"]);
const VISUAL_RATE_LIMIT = 20;
const VISUAL_RATE_LIMIT_WINDOW_SECONDS = 60;

interface VisualRateLimitStatement {
  bind(...values: unknown[]): VisualRateLimitStatement;
  first<T>(): Promise<T | null>;
}

export interface VisualRateLimitDatabase {
  prepare(query: string): VisualRateLimitStatement;
}

const UPSERT_VISUAL_RATE_LIMIT = `
INSERT INTO visual_rate_limits (client_key, window_start, request_count, updated_at)
VALUES (?, ?, 1, ?)
ON CONFLICT(client_key) DO UPDATE SET
  window_start = excluded.window_start,
  request_count = CASE
    WHEN visual_rate_limits.window_start = excluded.window_start
      THEN visual_rate_limits.request_count + 1
    ELSE 1
  END,
  updated_at = excluded.updated_at
RETURNING request_count
`;

export function isVisualApiRequest(
  request: Pick<Request, "method" | "url">,
): boolean {
  if (request.method !== "GET") return false;
  const url = new URL(request.url);
  const searchMode = url.searchParams.get("searchMode");
  return (
    VISUAL_API_PATHS.has(url.pathname) &&
    (searchMode === null || searchMode === "embedding")
  );
}

async function visualClientKey(clientIp: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(`mnemosyne:visual:v1:${clientIp}`),
  );
  return Array.from(new Uint8Array(digest).slice(0, 16), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

export async function checkVisualRateLimit(
  request: Request,
  database?: VisualRateLimitDatabase,
  nowMs = Date.now(),
): Promise<Response | null> {
  if (!database || !isVisualApiRequest(request)) return null;

  // Cloudflare sets this header at its edge. Do not trust a browser-controlled
  // forwarded-for header when selecting the limiter key, and never persist the
  // address itself: D1 receives only a one-way, application-scoped digest.
  const clientIp = request.headers.get("CF-Connecting-IP")?.trim();
  if (
    !clientIp ||
    clientIp.length > 64 ||
    !/^[0-9a-f:.]+$/i.test(clientIp)
  ) {
    return null;
  }

  try {
    const nowSeconds = Math.floor(nowMs / 1_000);
    const windowStart =
      nowSeconds - (nowSeconds % VISUAL_RATE_LIMIT_WINDOW_SECONDS);
    const row = await database
      .prepare(UPSERT_VISUAL_RATE_LIMIT)
      .bind(await visualClientKey(clientIp), windowStart, nowSeconds)
      .first<{ request_count: number }>();
    if (Number(row?.request_count ?? 0) <= VISUAL_RATE_LIMIT) return null;
  } catch {
    // Rate limiting mitigates overload; private-Space authentication remains
    // the security boundary. Fail open if D1 is temporarily unavailable.
    return null;
  }

  return Response.json(
    {
      error: "Visual search is busy. Please try again shortly.",
      code: "visual-busy",
    },
    {
      status: 429,
      headers: {
        "Cache-Control": "no-store",
        "Retry-After": "60",
      },
    },
  );
}
