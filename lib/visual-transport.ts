import { buildEvidenceUrl, buildSearchUrl } from "./search-mode.ts";
import type { ChartSelection } from "./types";

type TransportOptions = {
  fetch?: typeof fetch;
  signal?: AbortSignal;
};

export type VisualTransportResult = {
  response: Response;
  payload: unknown;
  via: "proxy";
};

async function decoded(response: Response) {
  try {
    return await response.json() as unknown;
  } catch {
    return {
      error: response.ok
        ? "Visual search returned an unsupported response."
        : `Visual search returned ${response.status}.`,
    };
  }
}

async function visualRequest(
  url: string,
  options: TransportOptions,
): Promise<VisualTransportResult> {
  const fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
  const response = await fetcher(url, {
    headers: { Accept: "application/json" },
    signal: options.signal,
  });
  return { response, payload: await decoded(response), via: "proxy" };
}

export function requestVisualSearch(
  query: string,
  options: TransportOptions = {},
) {
  return visualRequest(buildSearchUrl(query, "embedding"), options);
}

export function requestVisualEvidence(
  query: string,
  selection: ChartSelection,
  options: TransportOptions = {},
) {
  return visualRequest(buildEvidenceUrl(query, "embedding", selection), options);
}
