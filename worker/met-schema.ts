interface SchemaDatabase {
  prepare(query: string): { run(): Promise<unknown> };
}

const MET_SCHEMA = [
  `CREATE TABLE IF NOT EXISTS artworks (
    row_id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL UNIQUE,
    artwork_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    artist TEXT NOT NULL DEFAULT '',
    culture TEXT NOT NULL DEFAULT '',
    medium TEXT NOT NULL DEFAULT '',
    object_type TEXT NOT NULL DEFAULT '',
    classification TEXT NOT NULL DEFAULT '',
    period TEXT NOT NULL DEFAULT '',
    dynasty TEXT NOT NULL DEFAULT '',
    geography TEXT NOT NULL DEFAULT '',
    department TEXT NOT NULL DEFAULT '',
    date_display TEXT NOT NULL DEFAULT '',
    date_start INTEGER NOT NULL,
    date_end INTEGER NOT NULL,
    date_qualifier TEXT NOT NULL DEFAULT 'range',
    object_url TEXT NOT NULL DEFAULT '',
    credit_line TEXT NOT NULL DEFAULT '',
    public_domain INTEGER NOT NULL DEFAULT 0
  )`,
  `CREATE VIRTUAL TABLE IF NOT EXISTS artwork_fts USING fts5(
    title, tags, artist, culture, medium, object_type, classification,
    period, dynasty, geography, department,
    content='artworks', content_rowid='row_id',
    tokenize='porter unicode61 remove_diacritics 2'
  )`,
  `CREATE TRIGGER IF NOT EXISTS artworks_ai AFTER INSERT ON artworks BEGIN
    INSERT INTO artwork_fts(
      rowid, title, tags, artist, culture, medium, object_type,
      classification, period, dynasty, geography, department
    ) VALUES (
      new.row_id, new.title, new.tags, new.artist, new.culture, new.medium,
      new.object_type, new.classification, new.period, new.dynasty,
      new.geography, new.department
    );
  END`,
  `CREATE TRIGGER IF NOT EXISTS artworks_ad AFTER DELETE ON artworks BEGIN
    INSERT INTO artwork_fts(
      artwork_fts, rowid, title, tags, artist, culture, medium, object_type,
      classification, period, dynasty, geography, department
    ) VALUES (
      'delete', old.row_id, old.title, old.tags, old.artist, old.culture,
      old.medium, old.object_type, old.classification, old.period, old.dynasty,
      old.geography, old.department
    );
  END`,
  `CREATE TRIGGER IF NOT EXISTS artworks_au AFTER UPDATE ON artworks BEGIN
    INSERT INTO artwork_fts(
      artwork_fts, rowid, title, tags, artist, culture, medium, object_type,
      classification, period, dynasty, geography, department
    ) VALUES (
      'delete', old.row_id, old.title, old.tags, old.artist, old.culture,
      old.medium, old.object_type, old.classification, old.period, old.dynasty,
      old.geography, old.department
    );
    INSERT INTO artwork_fts(
      rowid, title, tags, artist, culture, medium, object_type,
      classification, period, dynasty, geography, department
    ) VALUES (
      new.row_id, new.title, new.tags, new.artist, new.culture, new.medium,
      new.object_type, new.classification, new.period, new.dynasty,
      new.geography, new.department
    );
  END`,
  `CREATE TABLE IF NOT EXISTS bins (
    bin_index INTEGER PRIMARY KEY,
    bin_key TEXT NOT NULL UNIQUE,
    bin_start INTEGER NOT NULL,
    bin_end INTEGER NOT NULL,
    bin_label TEXT NOT NULL,
    denominator REAL NOT NULL,
    object_count INTEGER NOT NULL,
    cluster_count INTEGER NOT NULL
  )`,
  `CREATE TABLE IF NOT EXISTS met_object_cache (
    source_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    artist TEXT NOT NULL DEFAULT '',
    object_date TEXT NOT NULL DEFAULT '',
    object_url TEXT NOT NULL DEFAULT '',
    image_url TEXT NOT NULL DEFAULT '',
    credit_line TEXT NOT NULL DEFAULT '',
    public_domain INTEGER NOT NULL DEFAULT 0,
    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
  )`,
  `CREATE TABLE IF NOT EXISTS corpus_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
  )`,
  "CREATE INDEX IF NOT EXISTS idx_bins_end_start ON bins(bin_end, bin_start)",
] as const;

/** Initialize an empty standalone D1 before the resumable corpus import. */
export async function ensureMetSchema(db: SchemaDatabase) {
  for (const statement of MET_SCHEMA) {
    await db.prepare(statement).run();
  }
}
