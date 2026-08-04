export const MAX_QUERY_TERMS = 5;
export const MAX_QUERY_LENGTH = 500;

export type ParsedQuery = {
  label: string;
  normalized: string;
};

export class QuerySyntaxError extends Error {
  readonly code:
    | "empty-query"
    | "empty-term"
    | "too-many-terms"
    | "unterminated-quote"
    | "query-too-long";

  constructor(
    message: string,
    code:
      | "empty-query"
      | "empty-term"
      | "too-many-terms"
      | "unterminated-quote"
      | "query-too-long",
  ) {
    super(message);
    this.name = "QuerySyntaxError";
    this.code = code;
  }
}

export function normalizeQueryTerm(value: string) {
  return value.normalize("NFKC").trim().replace(/\s+/g, " ").toLocaleLowerCase("en-US");
}

/** Parse commas outside double quotes, preserving literal commas inside a concept. */
export function parseConceptQuery(input: string): ParsedQuery[] {
  if (input.length > MAX_QUERY_LENGTH) {
    throw new QuerySyntaxError(
      `Keep the complete query under ${MAX_QUERY_LENGTH} characters.`,
      "query-too-long",
    );
  }

  if (!input.trim()) {
    throw new QuerySyntaxError("Add at least one concept.", "empty-query");
  }

  const rawTerms: string[] = [];
  let current = "";
  let quoted = false;

  for (let index = 0; index < input.length; index += 1) {
    const character = input[index];
    if (character === '"') {
      if (quoted && input[index + 1] === '"') {
        current += '"';
        index += 1;
        continue;
      }
      quoted = !quoted;
      continue;
    }

    if (character === "," && !quoted) {
      rawTerms.push(current);
      current = "";
      continue;
    }

    current += character;
  }

  if (quoted) {
    throw new QuerySyntaxError("Close the quoted concept before searching.", "unterminated-quote");
  }

  rawTerms.push(current);
  if (rawTerms.some((term) => !term.trim())) {
    throw new QuerySyntaxError("Remove empty concepts between commas.", "empty-term");
  }

  const seen = new Set<string>();
  const terms: ParsedQuery[] = [];
  for (const rawTerm of rawTerms) {
    const label = rawTerm.trim().replace(/\s+/g, " ");
    const normalized = normalizeQueryTerm(label);
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    terms.push({ label, normalized });
  }

  if (terms.length > MAX_QUERY_TERMS) {
    throw new QuerySyntaxError(
      `Compare up to ${MAX_QUERY_TERMS} concepts at a time.`,
      "too-many-terms",
    );
  }

  return terms;
}
