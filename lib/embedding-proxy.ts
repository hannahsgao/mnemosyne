import type { SearchMode } from "./search-mode.ts";

export const NO_STORE_CACHE_CONTROL = "no-store";
export const EMBEDDING_CACHE_CONTROL =
  "public, max-age=0, s-maxage=3600, stale-while-revalidate=86400";

const DEFAULT_KEYWORD_TIMEOUT_MS = 15_000;
const DEFAULT_EMBEDDING_TIMEOUT_MS = 60_000;
const MAX_EMBEDDING_TIMEOUT_MS = 120_000;

type Environment = Readonly<Record<string, string | undefined>>;

export function upstreamHeaders(
  searchMode: SearchMode,
  environment: Environment,
  accept = "application/json",
): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: accept,
    "User-Agent": "Mnemosyne web app",
  };
  if (accept === "application/json") headers["Content-Type"] = "application/json";

  if (searchMode === "embedding") {
    const token = environment.MNEMOSYNE_EMBEDDING_SEARCH_SERVICE_TOKEN?.trim();
    if (token) headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

export function upstreamTimeoutMs(
  searchMode: SearchMode,
  environment: Environment,
): number {
  if (searchMode === "keyword") return DEFAULT_KEYWORD_TIMEOUT_MS;
  const configured = Number(environment.MNEMOSYNE_EMBEDDING_SEARCH_TIMEOUT_MS?.trim());
  if (!Number.isFinite(configured) || configured <= 0) return DEFAULT_EMBEDDING_TIMEOUT_MS;
  return Math.min(Math.floor(configured), MAX_EMBEDDING_TIMEOUT_MS);
}

export function outerCacheControl(
  searchMode: SearchMode,
  status: number,
  upstreamCacheControl: string | null = null,
): string {
  if (status < 200 || status >= 300) return NO_STORE_CACHE_CONTROL;
  if (searchMode === "embedding") return EMBEDDING_CACHE_CONTROL;
  return upstreamCacheControl?.trim() || NO_STORE_CACHE_CONTROL;
}

export function safeRetryAfter(value: string | null): string | null {
  if (!value || !/^\d+$/.test(value)) return null;
  const seconds = Number(value);
  return Number.isSafeInteger(seconds) && seconds > 0 && seconds <= 300
    ? String(seconds)
    : null;
}

export function isTimeoutError(error: unknown): boolean {
  return error instanceof Error && error.name === "TimeoutError";
}

export type VisualProxyError = {
  status: number;
  body: { error: string; code: "visual-busy" | "visual-warming" | "visual-unavailable" };
};

export function visualProxyError(status: number, timedOut = false): VisualProxyError {
  if (status === 429) {
    return {
      status,
      body: {
        error: "Visual search is busy. Please try again shortly.",
        code: "visual-busy",
      },
    };
  }
  if (timedOut || status === 503 || status === 504) {
    return {
      status: 503,
      body: {
        error: "Visual search is warming up. Please try again shortly.",
        code: "visual-warming",
      },
    };
  }
  return {
    status: 502,
    body: {
      error: "Visual search is unavailable.",
      code: "visual-unavailable",
    },
  };
}
