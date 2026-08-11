import { buildEvidenceUrl, buildSearchUrl } from "./search-mode.ts";
import type { ChartSelection } from "./types";

export const KEYWORD_DIRECT_FALLBACK_STATUSES = [404, 405, 501] as const;

type TransportOptions = {
  fetch?: typeof fetch;
  signal?: AbortSignal;
};

export type KeywordTransportResult = {
  response: Response;
  payload: unknown;
  via: "direct" | "proxy";
};

function fallbackStatus(status: number) {
  return (KEYWORD_DIRECT_FALLBACK_STATUSES as readonly number[]).includes(status);
}

async function decoded(response: Response) {
  try {
    return { payload: await response.json() as unknown, validJson: true };
  } catch {
    return {
      payload: {
        error: response.ok
          ? "The search endpoint returned an unsupported response."
          : `The search endpoint returned ${response.status}.`,
      },
      validJson: false,
    };
  }
}

async function proxyRequest(
  fallbackUrl: string,
  fetcher: typeof fetch,
  signal: AbortSignal | undefined,
): Promise<KeywordTransportResult> {
  const response = await fetcher(fallbackUrl, {
    headers: { Accept: "application/json" },
    signal,
  });
  const { payload } = await decoded(response);
  return { response, payload, via: "proxy" };
}

async function keywordRequest(
  directPath: "/v1/search" | "/v1/evidence",
  fallbackUrl: string,
  body: Record<string, unknown>,
  options: TransportOptions,
): Promise<KeywordTransportResult> {
  const fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
  const response = await fetcher(directPath, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    cache: "no-store",
    signal: options.signal,
  });
  if (fallbackStatus(response.status)) {
    return proxyRequest(fallbackUrl, fetcher, options.signal);
  }

  const result = await decoded(response);
  // Some local/static hosts answer unknown application routes with a successful
  // HTML shell. A successful non-JSON response is unambiguously not this API.
  if (response.ok && !result.validJson) {
    return proxyRequest(fallbackUrl, fetcher, options.signal);
  }
  return { response, payload: result.payload, via: "direct" };
}

export function requestKeywordSearch(
  query: string,
  options: TransportOptions = {},
) {
  return keywordRequest(
    "/v1/search",
    buildSearchUrl(query, "keyword"),
    { query },
    options,
  );
}

export function requestKeywordEvidence(
  query: string,
  selection: ChartSelection,
  options: TransportOptions = {},
) {
  return keywordRequest(
    "/v1/evidence",
    buildEvidenceUrl(query, "keyword", selection),
    {
      query,
      selectedQueryId: selection.queryId,
      selectedBinKey: selection.binKey,
    },
    options,
  );
}
