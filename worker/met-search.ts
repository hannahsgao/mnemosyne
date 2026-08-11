import { keywordEvidenceSlices } from "../lib/evidence.ts";

type D1Row = Record<string, unknown>;

interface D1Result<T = D1Row> {
  results?: T[];
  success?: boolean;
}

interface D1PreparedStatement {
  bind(...values: unknown[]): D1PreparedStatement;
  all<T = D1Row>(): Promise<D1Result<T>>;
  first<T = D1Row>(): Promise<T | null>;
  run(): Promise<D1Result>;
}

export interface D1Database {
  prepare(query: string): D1PreparedStatement;
  batch<T = D1Row>(statements: D1PreparedStatement[]): Promise<D1Result<T>[]>;
}

interface MetEnv {
  DB?: D1Database;
  MNEMOSYNE_IMPORT_TOKEN?: string;
}

type QueryTerm = { id: string; label: string; normalized: string };
type BinRow = {
  bin_index: number;
  bin_key: string;
  bin_start: number;
  bin_end: number;
  bin_label: string;
  denominator: number;
  object_count: number;
  cluster_count: number;
};

type AggregateRow = {
  bin_index: number;
  hit_mass: number;
  object_count: number;
};

type ArtworkRow = {
  source_id: number;
  artwork_id: string;
  title: string;
  artist: string;
  date_display: string;
  date_start: number;
  date_end: number;
  date_qualifier: string;
  object_url: string;
  credit_line: string;
  public_domain: number;
  contribution_weight: number;
};

type CachedObject = {
  source_id: number;
  title: string;
  artist: string;
  object_date: string;
  object_url: string;
  image_url: string;
  credit_line: string;
  public_domain: number;
};

const PUBLIC_JSON_HEADERS = {
  "Content-Type": "application/json; charset=utf-8",
  "Cache-Control": "public, max-age=60, s-maxage=300",
};
const PRIVATE_JSON_HEADERS = {
  "Content-Type": "application/json; charset=utf-8",
  "Cache-Control": "private, no-store",
};
const MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1";
const MINIMUM_DENOMINATOR = 20;
const MAX_SERIES = 5;
const SEARCH_SCHEMA_VERSION = "mnemosyne.search.v1";
const EVIDENCE_SCHEMA_VERSION = "mnemosyne.evidence.v1";

class RequestValidationError extends Error {
  override readonly name = "RequestValidationError";
}

function publicJson(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: PUBLIC_JSON_HEADERS });
}

function privateJson(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: PRIVATE_JSON_HEADERS });
}

function validationErrorResponse(error: RequestValidationError) {
  return privateJson({ error: error.message }, 400);
}

function internalErrorResponse() {
  return privateJson({ error: "Internal server error" }, 500);
}

function normalizeTerm(value: string) {
  return value.normalize("NFKC").trim().replace(/\s+/g, " ").toLowerCase();
}

async function termId(normalized: string) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(normalized));
  return `q-${Array.from(new Uint8Array(digest).slice(0, 6), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("")}`;
}

async function parseQuery(raw: unknown): Promise<QueryTerm[]> {
  if (typeof raw !== "string") throw new RequestValidationError("query must be a string");
  if (raw.length > 500) {
    throw new RequestValidationError("query must be at most 500 characters");
  }
  const fields: string[] = [];
  let buffer = "";
  let inQuotes = false;
  for (let index = 0; index < raw.length; index += 1) {
    const character = raw[index];
    if (character === '"') {
      if (inQuotes && raw[index + 1] === '"') {
        buffer += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (character === "," && !inQuotes) {
      fields.push(buffer.trim());
      buffer = "";
    } else {
      buffer += character;
    }
  }
  if (inQuotes) throw new RequestValidationError("query contains an unmatched double quote");
  fields.push(buffer.trim());
  if (fields.some((field) => !field)) {
    throw new RequestValidationError("query contains an empty series");
  }

  const seen = new Set<string>();
  const terms: QueryTerm[] = [];
  for (const label of fields) {
    const normalized = normalizeTerm(label);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    terms.push({ id: await termId(normalized), label, normalized });
  }
  if (!terms.length) {
    throw new RequestValidationError("query must contain at least one series");
  }
  if (terms.length > MAX_SERIES) {
    throw new RequestValidationError(`query supports at most ${MAX_SERIES} unique series`);
  }
  return terms;
}

function ftsExpression(query: string) {
  return `"${query.replaceAll('"', '""')}"`;
}

function number(value: unknown) {
  return typeof value === "number" ? value : Number(value ?? 0);
}

function rows<T>(result: D1Result<T>) {
  return result.results ?? [];
}

async function loadBins(db: D1Database): Promise<BinRow[]> {
  const result = await db
    .prepare(
      `SELECT bin_index, bin_key, bin_start, bin_end, bin_label,
              denominator, object_count, cluster_count
       FROM bins ORDER BY bin_index`,
    )
    .all<BinRow>();
  return rows(result).map((row) => ({
    ...row,
    bin_index: number(row.bin_index),
    bin_start: number(row.bin_start),
    bin_end: number(row.bin_end),
    denominator: number(row.denominator),
    object_count: number(row.object_count),
    cluster_count: number(row.cluster_count),
  }));
}

async function aggregateTerm(db: D1Database, term: QueryTerm, bins: BinRow[]) {
  const expression = ftsExpression(term.normalized);
  const [countRow, aggregateResult] = await Promise.all([
    db
      .prepare(
        `SELECT COUNT(*) AS total
         FROM artwork_fts
         JOIN artworks ON artworks.row_id = artwork_fts.rowid
         WHERE artwork_fts MATCH ?`,
      )
      .bind(expression)
      .first<{ total: number }>(),
    db
      .prepare(
        `SELECT bins.bin_index AS bin_index,
                SUM(
                  CAST(
                    MAX(0, MIN(artworks.date_end, bins.bin_end) -
                           MAX(artworks.date_start, bins.bin_start) + 1 -
                           CASE
                             WHEN MAX(artworks.date_start, bins.bin_start) <= 0
                              AND MIN(artworks.date_end, bins.bin_end) >= 0 THEN 1
                             ELSE 0
                           END)
                    AS REAL
                  ) /
                  CAST(
                    artworks.date_end - artworks.date_start + 1 -
                    CASE
                      WHEN artworks.date_start <= 0 AND artworks.date_end >= 0 THEN 1
                      ELSE 0
                    END
                    AS REAL
                  )
                ) AS hit_mass,
                COUNT(*) AS object_count
         FROM artwork_fts
         JOIN artworks ON artworks.row_id = artwork_fts.rowid
         JOIN bins ON bins.bin_end >= artworks.date_start
                  AND bins.bin_start <= artworks.date_end
         WHERE artwork_fts MATCH ?
         GROUP BY bins.bin_index
         ORDER BY bins.bin_index`,
      )
      .bind(expression)
      .all<AggregateRow>(),
  ]);
  const byBin = new Map(
    rows(aggregateResult).map((row) => [
      number(row.bin_index),
      { hitMass: number(row.hit_mass), objectCount: number(row.object_count) },
    ]),
  );
  const points = bins.map((bin) => {
    const aggregate = byBin.get(bin.bin_index) ?? { hitMass: 0, objectCount: 0 };
    const share = bin.denominator ? aggregate.hitMass / bin.denominator : 0;
    return {
      binKey: bin.bin_key,
      value: share,
      share,
      lift: null,
      hitMass: aggregate.hitMass,
      objectCount: aggregate.objectCount,
      clusterCount: aggregate.objectCount,
    };
  });
  return { totalMatches: number(countRow?.total), points, expression };
}

function selectedBinIndex(
  requestedKey: unknown,
  bins: BinRow[],
  points: Array<{ value: number }>,
) {
  if (typeof requestedKey === "string") {
    const index = bins.findIndex((bin) => bin.bin_key === requestedKey);
    if (index < 0) {
      throw new RequestValidationError("selectedBinKey does not name a timeline bin");
    }
    return index;
  }
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < points.length; index += 1) {
    if (bins[index].denominator < MINIMUM_DENOMINATOR) continue;
    if (points[index].value > bestValue) {
      bestValue = points[index].value;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function stableSeed(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

async function cachedObject(db: D1Database, sourceId: number) {
  return db
    .prepare(
      `SELECT source_id, title, artist, object_date, object_url, image_url,
              credit_line, public_domain
       FROM met_object_cache WHERE source_id = ?`,
    )
    .bind(sourceId)
    .first<CachedObject>();
}

async function hydrateObject(db: D1Database, artwork: ArtworkRow): Promise<CachedObject | null> {
  const cached = await cachedObject(db, artwork.source_id);
  if (cached) return cached;
  try {
    const response = await fetch(`${MET_API_BASE}/objects/${artwork.source_id}`, {
      headers: { Accept: "application/json", "User-Agent": "Mnemosyne art history search" },
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) return null;
    const detail = (await response.json()) as Record<string, unknown>;
    const item: CachedObject = {
      source_id: artwork.source_id,
      title: String(detail.title ?? ""),
      artist: String(detail.artistDisplayName ?? ""),
      object_date: String(detail.objectDate ?? ""),
      object_url: String(detail.objectURL ?? ""),
      image_url: detail.isPublicDomain ? String(detail.primaryImageSmall ?? "") : "",
      credit_line: String(detail.creditLine ?? ""),
      public_domain: detail.isPublicDomain ? 1 : 0,
    };
    await db
      .prepare(
        `INSERT OR REPLACE INTO met_object_cache
         (source_id, title, artist, object_date, object_url, image_url,
          credit_line, public_domain, fetched_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)`,
      )
      .bind(
        item.source_id,
        item.title,
        item.artist,
        item.object_date,
        item.object_url,
        item.image_url,
        item.credit_line,
        item.public_domain,
      )
      .run();
    return item;
  } catch {
    return null;
  }
}

async function mapWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  mapper: (item: T) => Promise<R>,
) {
  const output = new Array<R>(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const index = next;
      next += 1;
      output[index] = await mapper(items[index]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  return output;
}

async function evidence(
  db: D1Database,
  term: QueryTerm,
  expression: string,
  bin: BinRow,
) {
  const seed = stableSeed(`${term.normalized}:${bin.bin_key}`);
  const result = await db
    .prepare(
      `SELECT artworks.source_id, artworks.artwork_id, artworks.title,
              artworks.artist, artworks.date_display, artworks.date_start,
              artworks.date_end, artworks.date_qualifier, artworks.object_url,
              artworks.credit_line, artworks.public_domain,
              CAST(
                MAX(0, MIN(artworks.date_end, ?) - MAX(artworks.date_start, ?) + 1 -
                  CASE
                    WHEN MAX(artworks.date_start, ?) <= 0
                     AND MIN(artworks.date_end, ?) >= 0 THEN 1
                    ELSE 0
                  END)
                AS REAL
              ) /
              CAST(
                artworks.date_end - artworks.date_start + 1 -
                CASE
                  WHEN artworks.date_start <= 0 AND artworks.date_end >= 0 THEN 1
                  ELSE 0
                END
                AS REAL
              ) AS contribution_weight
       FROM artwork_fts
       JOIN artworks ON artworks.row_id = artwork_fts.rowid
       WHERE artwork_fts MATCH ?
         AND artworks.date_end >= ? AND artworks.date_start <= ?
       ORDER BY artworks.public_domain DESC,
                ABS((artworks.source_id * 1103515245 + ?) % 2147483647)
       LIMIT 25`,
    )
    .bind(
      bin.bin_end,
      bin.bin_start,
      bin.bin_start,
      bin.bin_end,
      expression,
      bin.bin_start,
      bin.bin_end,
      seed,
    )
    .all<ArtworkRow>();
  const artworks = rows(result).map((row) => ({
    ...row,
    source_id: number(row.source_id),
    date_start: number(row.date_start),
    date_end: number(row.date_end),
    public_domain: number(row.public_domain),
    contribution_weight: number(row.contribution_weight),
  }));
  const details = await mapWithConcurrency(artworks, 6, (artwork) =>
    artwork.public_domain ? hydrateObject(db, artwork) : Promise.resolve(null),
  );
  return artworks.map((artwork, index) => {
    const detail = details[index];
    const publicDomain = Boolean(artwork.public_domain && (detail?.public_domain ?? 1));
    return {
      artworkId: artwork.artwork_id,
      physicalObjectId: artwork.artwork_id,
      visualClusterId: `object:${artwork.artwork_id}`,
      title: detail?.title || artwork.title || "Untitled",
      artist: detail?.artist || artwork.artist || "Unknown artist",
      institution: "The Metropolitan Museum of Art",
      sourceRecordUrl: detail?.object_url || artwork.object_url,
      imageUrl: publicDomain ? detail?.image_url || "" : "",
      dateDisplay: detail?.object_date || artwork.date_display || "Unknown date",
      dateStart: artwork.date_start,
      dateEnd: artwork.date_end,
      dateQualifier: artwork.date_qualifier,
      rawScore: null,
      contributionWeight: artwork.contribution_weight,
      contributor: true,
      metadataLicense: "https://creativecommons.org/publicdomain/zero/1.0/",
      imageRightsUri: "https://creativecommons.org/publicdomain/mark/1.0/",
      creditLine: detail?.credit_line || artwork.credit_line,
      publicDomain,
    };
  });
}

function selectedTermIndex(payload: Record<string, unknown>, terms: QueryTerm[]) {
  const index =
    typeof payload.selectedQueryId === "string"
      ? terms.findIndex((term) => term.id === payload.selectedQueryId)
      : 0;
  if (index < 0) {
    throw new RequestValidationError("selectedQueryId does not name a parsed query series");
  }
  return index;
}

function selectedEvidencePayload(term: QueryTerm, bin: BinRow, cards: Awaited<ReturnType<typeof evidence>>) {
  return {
    queryId: term.id,
    binKey: bin.bin_key,
    slices: keywordEvidenceSlices(cards),
  };
}

async function requestPayload(request: Request) {
  try {
    const payload = (await request.json()) as unknown;
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      throw new RequestValidationError("request body must be a JSON object");
    }
    return payload as Record<string, unknown>;
  } catch (error) {
    if (error instanceof RequestValidationError) throw error;
    throw new RequestValidationError("request body must be JSON");
  }
}

async function handleSearch(request: Request, db: D1Database) {
  let payload: Record<string, unknown>;
  try {
    payload = await requestPayload(request);
  } catch (error) {
    return error instanceof RequestValidationError
      ? validationErrorResponse(error)
      : internalErrorResponse();
  }
  try {
    const terms = await parseQuery(payload.query);
    const bins = await loadBins(db);
    if (!bins.length) {
      return privateJson({ error: "The Met corpus has not been imported yet." }, 503);
    }
    const corpusCountRow = await db.prepare("SELECT COUNT(*) AS total FROM artworks").first<{
      total: number;
    }>();
    const aggregates = await Promise.all(terms.map((term) => aggregateTerm(db, term, bins)));
    const selectedIndex = selectedTermIndex(payload, terms);
    const binIndex = selectedBinIndex(payload.selectedBinKey, bins, aggregates[selectedIndex].points);
    const cards = await evidence(
      db,
      terms[selectedIndex],
      aggregates[selectedIndex].expression,
      bins[binIndex],
    );
    const warnings: string[] = [];
    const sparseCount = bins.filter((bin) => bin.denominator < MINIMUM_DENOMINATOR).length;
    if (sparseCount) {
      warnings.push(
        `${sparseCount} bin(s) are below the minimum denominator of ${MINIMUM_DENOMINATOR}.`,
      );
    }
    terms.forEach((term, index) => {
      if (!aggregates[index].totalMatches) {
        warnings.push(`No eligible corpus matches were found for '${term.label}'.`);
      }
    });
    return publicJson({
      schemaVersion: SEARCH_SCHEMA_VERSION,
      queries: terms,
      corpus: {
        id: "met-open-access",
        version: "e901de145e60258542243571098245826a01fe47",
        label: "The Met Open Access collection",
        count: number(corpusCountRow?.total),
        countingUnit: "physical-object",
        view: "all",
        filters: {},
      },
      model: { id: "met-d1-fts5-keyword", version: "v1", promptTemplateVersion: "none" },
      metric: {
        id: "met-metadata-frequency",
        version: "v1",
        label: "Met metadata frequency",
        percentile: null,
        unit: "frequency",
        description: "Date-weighted matching objects divided by all eligible objects in each bin.",
      },
      bins: bins.map((bin) => ({
        key: bin.bin_key,
        label: bin.bin_label,
        start: bin.bin_start,
        end: bin.bin_end,
        denominator: bin.denominator,
        objectCount: bin.object_count,
        clusterCount: bin.cluster_count,
        belowMinimumDenominator: bin.denominator < MINIMUM_DENOMINATOR,
      })),
      series: terms.map((term, index) => ({
        queryId: term.id,
        k: aggregates[index].totalMatches,
        threshold: null,
        lowSignal: null,
        diagnostics: {
          standardizedSeparation: null,
          controlMean: null,
          controlStdDev: null,
          promptTopKJaccard: null,
          reasons: [],
        },
        points: aggregates[index].points,
        totalMatches: aggregates[index].totalMatches,
      })),
      selectedEvidence: selectedEvidencePayload(
        terms[selectedIndex],
        bins[binIndex],
        cards,
      ),
      warnings,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    return error instanceof RequestValidationError
      ? validationErrorResponse(error)
      : internalErrorResponse();
  }
}

async function handleEvidence(request: Request, db: D1Database) {
  let payload: Record<string, unknown>;
  try {
    payload = await requestPayload(request);
  } catch (error) {
    return error instanceof RequestValidationError
      ? validationErrorResponse(error)
      : internalErrorResponse();
  }
  try {
    const terms = await parseQuery(payload.query);
    const bins = await loadBins(db);
    if (!bins.length) {
      return privateJson({ error: "The Met corpus has not been imported yet." }, 503);
    }
    const selectedIndex = selectedTermIndex(payload, terms);
    const term = terms[selectedIndex];

    let binIndex: number;
    if (typeof payload.selectedBinKey === "string") {
      binIndex = bins.findIndex((bin) => bin.bin_key === payload.selectedBinKey);
      if (binIndex < 0) {
        throw new RequestValidationError("selectedBinKey does not name a timeline bin");
      }
    } else {
      // Retain the search endpoint's default-period behavior for callers that
      // omit a bin. Explicit period requests avoid this aggregation entirely.
      const aggregate = await aggregateTerm(db, term, bins);
      binIndex = selectedBinIndex(undefined, bins, aggregate.points);
    }
    const cards = await evidence(db, term, ftsExpression(term.normalized), bins[binIndex]);
    return publicJson({
      schemaVersion: EVIDENCE_SCHEMA_VERSION,
      selectedEvidence: selectedEvidencePayload(term, bins[binIndex], cards),
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    return error instanceof RequestValidationError
      ? validationErrorResponse(error)
      : internalErrorResponse();
  }
}

const ARTWORK_COLUMNS = [
  "row_id",
  "source_id",
  "artwork_id",
  "title",
  "tags",
  "artist",
  "culture",
  "medium",
  "object_type",
  "classification",
  "period",
  "dynasty",
  "geography",
  "department",
  "date_display",
  "date_start",
  "date_end",
  "date_qualifier",
  "object_url",
  "credit_line",
  "public_domain",
] as const;

function authorized(request: Request, token: string | undefined) {
  if (!token) return false;
  return request.headers.get("authorization") === `Bearer ${token}`;
}

async function handleImport(request: Request, db: D1Database, token: string | undefined) {
  if (!authorized(request, token)) return privateJson({ error: "Unauthorized" }, 401);
  if (request.method === "GET") {
    const [artwork, bin, cache] = await Promise.all([
      db.prepare("SELECT COUNT(*) AS count, MAX(row_id) AS max_row_id FROM artworks").first(),
      db.prepare("SELECT COUNT(*) AS count FROM bins").first(),
      db.prepare("SELECT COUNT(*) AS count FROM met_object_cache").first(),
    ]);
    return privateJson({ artwork, bin, cache });
  }
  if (request.method !== "POST") return privateJson({ error: "Method not allowed" }, 405);
  const payload = (await requestPayload(request)) as { kind?: string; rows?: D1Row[] };
  if (payload.kind === "artworks") {
    const incoming = Array.isArray(payload.rows) ? payload.rows : [];
    if (!incoming.length || incoming.length > 500) {
      return privateJson({ error: "artworks import requires 1-500 rows" }, 400);
    }
    const sql = `INSERT OR IGNORE INTO artworks (${ARTWORK_COLUMNS.join(", ")})
                 VALUES (${ARTWORK_COLUMNS.map(() => "?").join(", ")})`;
    for (let start = 0; start < incoming.length; start += 75) {
      const statements = incoming.slice(start, start + 75).map((row) =>
        db.prepare(sql).bind(...ARTWORK_COLUMNS.map((column) => row[column] ?? "")),
      );
      await db.batch(statements);
    }
    return privateJson({ imported: incoming.length });
  }
  if (payload.kind === "bins") {
    const incoming = Array.isArray(payload.rows) ? payload.rows : [];
    const sql = `INSERT OR REPLACE INTO bins
                 (bin_index, bin_key, bin_start, bin_end, bin_label,
                  denominator, object_count, cluster_count)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`;
    for (let start = 0; start < incoming.length; start += 75) {
      await db.batch(
        incoming.slice(start, start + 75).map((row) =>
          db.prepare(sql).bind(
            row.bin_index,
            row.bin_key,
            row.bin_start,
            row.bin_end,
            row.bin_label,
            row.denominator,
            row.object_count,
            row.cluster_count,
          ),
        ),
      );
    }
    return privateJson({ imported: incoming.length });
  }
  if (payload.kind === "finalize") {
    await db.prepare("INSERT INTO artwork_fts(artwork_fts) VALUES('optimize')").run();
    await db
      .prepare("INSERT OR REPLACE INTO corpus_meta(key, value) VALUES('ready', 'true')")
      .run();
    return privateJson({ ready: true });
  }
  return privateJson({ error: "Unknown import kind" }, 400);
}

export async function handleMetServiceRequest(request: Request, env: MetEnv) {
  const pathname = new URL(request.url).pathname;
  if (
    pathname !== "/v1/search" &&
    pathname !== "/v1/evidence" &&
    pathname !== "/v1/health" &&
    pathname !== "/_admin/met/import"
  ) {
    return null;
  }
  if (!env.DB) return privateJson({ error: "D1 corpus binding is unavailable" }, 503);
  try {
    if (pathname === "/v1/health") {
      const ready = await env.DB
        .prepare("SELECT value FROM corpus_meta WHERE key = 'ready'")
        .first<{ value: string }>();
      return publicJson({
        status: ready?.value === "true" ? "ok" : "loading",
        backend: "d1-fts5",
      });
    }
    if (pathname === "/_admin/met/import") {
      return await handleImport(request, env.DB, env.MNEMOSYNE_IMPORT_TOKEN);
    }
    if (request.method !== "POST") {
      return privateJson({ error: "Method not allowed" }, 405);
    }
    return pathname === "/v1/evidence"
      ? await handleEvidence(request, env.DB)
      : await handleSearch(request, env.DB);
  } catch (error) {
    return error instanceof RequestValidationError
      ? validationErrorResponse(error)
      : internalErrorResponse();
  }
}
