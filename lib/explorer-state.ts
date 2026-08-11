import type { ChartSelection, SelectedEvidence } from "./types";

type Abortable = { abort(): void };

export function evidenceMatchesSelection(
  evidence: Pick<SelectedEvidence, "queryId" | "binKey"> | null | undefined,
  selection: ChartSelection,
) {
  return evidence?.queryId === selection.queryId && evidence.binKey === selection.binKey;
}

export function prepareEvidenceRequest(
  currentRequestId: number,
  currentAbort: Abortable | null,
  cached: boolean,
) {
  currentAbort?.abort();
  return {
    requestId: currentRequestId + 1,
    controller: cached ? null : new AbortController(),
    loading: !cached,
  };
}

export function invalidateExplorerRequests(
  currentSearchRequestId: number,
  currentEvidenceRequestId: number,
  searchAbort: Abortable | null,
  evidenceAbort: Abortable | null,
) {
  searchAbort?.abort();
  evidenceAbort?.abort();
  return {
    searchRequestId: currentSearchRequestId + 1,
    evidenceRequestId: currentEvidenceRequestId + 1,
  };
}

export function invalidSearchStatus(error: string) {
  return {
    error,
    loading: false,
    evidenceLoading: false,
  } as const;
}

export function searchErrorPlacement(error: string | null, hasResult: boolean) {
  if (!error) return null;
  return hasResult ? "inline" : "empty";
}

export function retryablePromise<T>(
  reference: { current: Promise<T> | null },
  load: () => Promise<T>,
) {
  if (reference.current) return reference.current;
  const pending = load();
  reference.current = pending;
  void pending.catch(() => {
    if (reference.current === pending) reference.current = null;
  });
  return pending;
}

function lastUnquotedComma(value: string) {
  let quoted = false;
  let last = -1;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (character === '"') {
      if (quoted && value[index + 1] === '"') {
        index += 1;
        continue;
      }
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      last = index;
    }
  }
  return last;
}

function unquoteFragment(value: string) {
  const trimmed = value.trim();
  if (!trimmed.startsWith('"')) return trimmed;
  const end = trimmed.endsWith('"') && trimmed.length > 1 ? -1 : undefined;
  return trimmed.slice(1, end).replace(/""/g, '"');
}

export function formatQueryTerm(value: string) {
  const normalized = value.trim().replace(/\s+/g, " ");
  return /[,\"]/.test(normalized)
    ? `"${normalized.replace(/"/g, '""')}"`
    : normalized;
}

export function activeQueryFragment(value: string) {
  return unquoteFragment(value.slice(lastUnquotedComma(value) + 1));
}

export function replaceActiveQueryFragment(value: string, replacement: string) {
  const comma = lastUnquotedComma(value);
  const formatted = formatQueryTerm(replacement);
  return comma < 0 ? formatted : `${value.slice(0, comma + 1)} ${formatted}`;
}
