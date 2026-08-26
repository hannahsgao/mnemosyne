import assert from "node:assert/strict";
import test from "node:test";
import {
  aggregateDateRanges,
  handleMetServiceRequest,
  type D1Database,
} from "../worker/met-search.ts";

class FixtureStatement {
  private values: unknown[] = [];
  private readonly database: FixtureDatabase;
  private readonly query: string;

  constructor(database: FixtureDatabase, query: string) {
    this.database = database;
    this.query = query;
  }

  bind(...values: unknown[]) {
    this.values = values;
    return this;
  }

  async all<T>() {
    this.database.queries.push(this.query);
    if (this.query.includes("FROM bins ORDER BY bin_index")) {
      return {
        results: [
          {
            bin_index: 0,
            bin_key: "1880:1889",
            bin_start: 1880,
            bin_end: 1889,
            bin_label: "1880–1889",
            denominator: 100,
            object_count: 100,
            cluster_count: 100,
          },
        ] as T[],
      };
    }
    if (this.query.includes("AS contribution_weight")) {
      return {
        results: [
          {
            source_id: 10,
            artwork_id: "MET_10",
            title: "Horse",
            artist: "Ada",
            date_display: "1880",
            date_start: 1880,
            date_end: 1880,
            date_qualifier: "exact",
            object_url: "https://www.metmuseum.org/art/collection/search/10",
            credit_line: "Fixture credit",
            public_domain: 1,
            contribution_weight: 1,
          },
        ] as T[],
      };
    }
    if (this.query.includes("COUNT(*) AS match_count")) {
      return {
        results: [
          { date_start: 1880, date_end: 1880, match_count: 1 },
        ] as T[],
      };
    }
    throw new Error(`Unexpected all() query: ${this.query}`);
  }

  async first<T>() {
    this.database.queries.push(this.query);
    if (this.query.includes("FROM corpus_meta WHERE key = 'ready'")) {
      return { value: "true" } as T;
    }
    if (this.query.includes("FROM met_object_cache WHERE source_id = ?")) {
      assert.deepEqual(this.values, [10]);
      return {
        source_id: 10,
        title: "Horse",
        artist: "Ada",
        object_date: "1880",
        object_url: "https://www.metmuseum.org/art/collection/search/10",
        image_url: "https://images.example/MET_10.jpg",
        credit_line: "Fixture credit",
        public_domain: 1,
      } as T;
    }
    if (this.query.includes("COUNT(*) AS total")) {
      return { total: 1 } as T;
    }
    throw new Error(`Unexpected first() query: ${this.query}`);
  }

  async run() {
    this.database.queries.push(this.query);
    return { success: true };
  }
}

class FixtureDatabase {
  readonly queries: string[] = [];

  prepare(query: string) {
    return new FixtureStatement(this, query);
  }

  async batch() {
    return [];
  }
}

class FailingStatement {
  bind() {
    return this;
  }

  async all() {
    throw new Error("D1 exploded: secret-infrastructure-detail");
  }

  async first() {
    throw new Error("D1 exploded: secret-infrastructure-detail");
  }

  async run() {
    throw new Error("D1 exploded: secret-infrastructure-detail");
  }
}

class FailingDatabase {
  prepare() {
    return new FailingStatement();
  }

  async batch() {
    throw new Error("D1 exploded: secret-infrastructure-detail");
  }
}

test("D1 evidence route omits timeline aggregation and preserves keyword slices", async () => {
  const fixture = new FixtureDatabase();
  const request = new Request("https://mnemosyne.example/v1/evidence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: "horse",
      selectedBinKey: "1880:1889",
    }),
  });

  const response = await handleMetServiceRequest(request, {
    DB: fixture as unknown as D1Database,
  });

  assert.ok(response);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Cache-Control"), "public, max-age=60, s-maxage=300");
  const payload = (await response.json()) as Record<string, any>;
  assert.deepEqual(Object.keys(payload).sort(), [
    "generatedAt",
    "schemaVersion",
    "selectedEvidence",
  ]);
  assert.equal(payload.schemaVersion, "mnemosyne.evidence.v1");
  assert.equal(payload.selectedEvidence.binKey, "1880:1889");
  assert.deepEqual(
    payload.selectedEvidence.slices.strongest,
    payload.selectedEvidence.slices.randomContributors,
  );
  assert.equal(payload.selectedEvidence.slices.strongest[0].artworkId, "MET_10");
  assert.equal(
    fixture.queries.some((query) => query.includes("AS contribution_weight")),
    true,
  );
  assert.equal(
    fixture.queries.some((query) => query.includes("FROM met_object_cache")),
    true,
  );
  assert.equal(fixture.queries.some((query) => query.includes("SUM(")), false);
  assert.equal(
    fixture.queries.some(
      (query) => query.includes("COUNT(*) AS total") && query.includes("FROM artwork_fts"),
    ),
    false,
  );
});

test("D1 search route returns the timeline without querying or hydrating evidence", async () => {
  const fixture = new FixtureDatabase();
  const request = new Request("https://mnemosyne.example/v1/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: "horse", selectedBinKey: "1880:1889" }),
  });

  const response = await handleMetServiceRequest(request, {
    DB: fixture as unknown as D1Database,
  });

  assert.ok(response);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Cache-Control"), "public, max-age=60, s-maxage=300");
  const payload = (await response.json()) as Record<string, any>;
  assert.equal(payload.schemaVersion, "mnemosyne.search.v1");
  assert.equal(payload.bins.length, 1);
  assert.equal(payload.series.length, 1);
  assert.equal(payload.selectedEvidence, null);
  assert.equal(
    fixture.queries.some((query) => query.includes("AS contribution_weight")),
    false,
  );
  assert.equal(
    fixture.queries.some((query) => query.includes("FROM met_object_cache")),
    false,
  );
  assert.equal(
    fixture.queries.some((query) => query.includes("COUNT(*) AS match_count")),
    true,
  );
  assert.equal(
    fixture.queries.some((query) => query.includes("JOIN bins ON")),
    false,
  );
  assert.equal(
    fixture.queries.some(
      (query) => query.includes("COUNT(*) AS total") && query.includes("FROM artwork_fts"),
    ),
    false,
  );
});

test("date-range aggregation preserves grouped counts and excludes year zero", () => {
  const bins = [
    { bin_start: -10, bin_end: -1 },
    { bin_start: 0, bin_end: 9 },
  ];
  const aggregate = aggregateDateRanges(
    [
      { date_start: -2, date_end: 2, match_count: 2 },
      { date_start: -10, date_end: -1, match_count: 3 },
    ],
    bins,
  );

  assert.equal(aggregate.totalMatches, 5);
  assert.deepEqual(Array.from(aggregate.hitMass), [4, 1]);
  assert.deepEqual(Array.from(aggregate.objectCounts), [5, 2]);
});

test("unauthorized admin responses are private and never cached", async () => {
  const response = await handleMetServiceRequest(
    new Request("https://mnemosyne.example/_admin/met/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "finalize" }),
    }),
    {
      DB: new FixtureDatabase() as unknown as D1Database,
      MNEMOSYNE_IMPORT_TOKEN: "correct-token",
    },
  );

  assert.ok(response);
  assert.equal(response.status, 401);
  assert.equal(response.headers.get("Cache-Control"), "private, no-store");
  assert.deepEqual(await response.json(), { error: "Unauthorized" });
});

test("request validation errors remain safe private 400 responses", async () => {
  const response = await handleMetServiceRequest(
    new Request("https://mnemosyne.example/v1/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: { secret: "do-not-reflect" } }),
    }),
    { DB: new FixtureDatabase() as unknown as D1Database },
  );

  assert.ok(response);
  assert.equal(response.status, 400);
  assert.equal(response.headers.get("Cache-Control"), "private, no-store");
  assert.deepEqual(await response.json(), { error: "query must be a string" });
});

test("unexpected D1 failures return a redacted private 500", async () => {
  const response = await handleMetServiceRequest(
    new Request("https://mnemosyne.example/v1/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "horse" }),
    }),
    { DB: new FailingDatabase() as unknown as D1Database },
  );

  assert.ok(response);
  assert.equal(response.status, 500);
  assert.equal(response.headers.get("Cache-Control"), "private, no-store");
  const payload = await response.json();
  assert.deepEqual(payload, { error: "Internal server error" });
  assert.equal(JSON.stringify(payload).includes("secret-infrastructure-detail"), false);
});

test("successful public health responses retain shared-cache policy", async () => {
  const response = await handleMetServiceRequest(
    new Request("https://mnemosyne.example/v1/health"),
    { DB: new FixtureDatabase() as unknown as D1Database },
  );

  assert.ok(response);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Cache-Control"), "public, max-age=60, s-maxage=300");
  assert.deepEqual(await response.json(), { status: "ok", backend: "d1-fts5" });
});
