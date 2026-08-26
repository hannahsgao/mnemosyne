import { buildBackendImageUrl } from "./search-mode.ts";
import type {
  ChartSelection,
  EvidenceArtwork,
  EvidenceSlices,
  SearchResponse,
  SearchSeries,
  SelectedEvidence,
  TimeBin,
} from "./types";
import { normalizeQueryTerm } from "./query.ts";

export type ConceptDefinition = {
  id: string;
  label: string;
  normalized: string;
  aliases: string[];
  category?: string;
  description?: string;
};

export type ConceptResolution = {
  requested: string;
  requestedNormalized: string;
  concept: ConceptDefinition;
  matchedText: string;
  matchedBy: "label" | "alias";
};

export type ConceptSuggestion = {
  concept: ConceptDefinition;
  matchedText: string;
  matchedBy: "label" | "alias";
};

type CatalogPointer = {
  release: string;
  complete?: boolean;
  fullCatalog?: boolean;
  catalogVersion?: string;
};

type ReleaseManifest = {
  releaseFingerprint: string;
  catalogVersion: string;
  complete: boolean;
  corpus: {
    id: string;
    version: string;
    label: string;
    count: number;
    countingUnit: "physical-object" | "visual-cluster" | "catalog-record";
  };
  model: {
    id: string;
    revision: string;
    promptTemplateVersion: string;
  };
  metric: {
    id: string;
    version: string;
    qualifiedFraction: number | null;
  };
  files: {
    bins: string;
    concepts: string;
    seriesTemplate: string;
    evidenceTemplate: string;
  };
  failures?: Array<{ conceptId?: string; error?: string }>;
};

type CompactSeries = {
  conceptId: string;
  k: number;
  threshold: number | null;
  candidateK?: number;
  candidateThreshold?: number | null;
  lowSignal: boolean | null;
  diagnostics: SearchSeries["diagnostics"];
  pointIndices: number[];
  values: number[];
  shares: Array<number | null>;
  hitMasses: number[];
  objectCounts: number[];
  clusterCounts: number[];
  suppressedBinIndices?: number[];
  defaultEvidenceBinIndex?: number | null;
};

type EvidencePeriod = {
  binIndex: number;
  contributorCount: number;
  artworkIds: string[];
  contributionWeights: number[];
};

type EvidenceBundle = {
  conceptId: string;
  percentile?: number | null;
  threshold?: number | null;
  artworks: Record<string, Omit<EvidenceArtwork, "contributionWeight">>;
  periods: EvidencePeriod[];
};

type LoadedSeries = {
  series: SearchSeries;
  defaultBinKey: string | null;
};

export type ConceptCatalog = {
  baseUrl: string;
  releaseBaseUrl: string;
  pointer: CatalogPointer;
  manifest: ReleaseManifest;
  concepts: ConceptDefinition[];
  bins: TimeBin[];
  fetcher: typeof fetch;
  names: Map<string, { concept: ConceptDefinition; matchedText: string; matchedBy: "label" | "alias" }>;
  conceptsById: Map<string, ConceptDefinition>;
  seriesCache: Map<string, Promise<LoadedSeries>>;
  evidenceCache: Map<string, Promise<EvidenceBundle>>;
  defaultBinKeys: Map<string, string | null>;
};

export class UnknownConceptError extends Error {
  readonly input: string;
  readonly suggestions: ConceptSuggestion[];

  constructor(input: string, suggestions: ConceptSuggestion[]) {
    const labels = suggestions.map((item) => item.concept.label);
    super(labels.length
      ? `“${input}” is not in the visual concept catalog. Try ${labels.join(", ")}.`
      : `“${input}” is not in the visual concept catalog.`);
    this.name = "UnknownConceptError";
    this.input = input;
    this.suggestions = suggestions;
  }
}

const catalogPromises = new Map<string, Promise<ConceptCatalog>>();

function assetUrl(base: string, relative: string) {
  if (/^https?:\/\//i.test(relative)) return relative;
  return `${base.replace(/\/$/, "")}/${relative.replace(/^\//, "")}`;
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object.`);
  }
  return value as Record<string, unknown>;
}

function stringValue(value: unknown, label: string) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} is missing.`);
  return value;
}

function numberArray(value: unknown, label: string, nullable = false) {
  if (!Array.isArray(value) || value.some((item) =>
    item !== null && (typeof item !== "number" || !Number.isFinite(item)) || (!nullable && item === null)
  )) {
    throw new Error(`${label} must be an array of finite numbers${nullable ? " or nulls" : ""}.`);
  }
  return value as number[];
}

function stringArray(value: unknown, label: string) {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new Error(`${label} must be an array of strings.`);
  }
  return value as string[];
}

async function fetchJson(fetcher: typeof fetch, url: string, label: string) {
  const response = await fetcher(url, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`${label} could not be loaded (${response.status}).`);
  try {
    return await response.json() as unknown;
  } catch {
    throw new Error(`${label} is not valid JSON.`);
  }
}

function parseConcepts(payload: unknown) {
  const source = record(payload, "Concept catalog");
  if (!Array.isArray(source.concepts)) throw new Error("Concept catalog has no concepts array.");
  const seen = new Set<string>();
  return source.concepts.map((value, index): ConceptDefinition => {
    const item = record(value, `Concept ${index + 1}`);
    const id = stringValue(item.id, `Concept ${index + 1} id`);
    const label = stringValue(item.label, `Concept ${id} label`);
    const normalized = stringValue(item.normalized, `Concept ${id} normalized`);
    const aliases = stringArray(item.aliases ?? [], `Concept ${id} aliases`);
    if (seen.has(id)) throw new Error(`Concept id ${id} is duplicated.`);
    seen.add(id);
    return {
      id,
      label,
      normalized,
      aliases,
      ...(typeof item.category === "string" && item.category ? { category: item.category } : {}),
      ...(typeof item.description === "string" && item.description ? { description: item.description } : {}),
    };
  });
}

export function expandCompactBins(payload: unknown): TimeBin[] {
  const source = record(payload, "Timeline bins");
  const keys = stringArray(source.keys, "Timeline bin keys");
  const labels = stringArray(source.labels, "Timeline bin labels");
  const starts = numberArray(source.starts, "Timeline bin starts");
  const ends = numberArray(source.ends, "Timeline bin ends");
  const denominators = numberArray(source.denominators, "Timeline denominators", true) as Array<number | null>;
  const objectCounts = numberArray(source.objectCounts, "Timeline object counts", true) as Array<number | null>;
  const clusterCounts = numberArray(source.clusterCounts, "Timeline cluster counts", true) as Array<number | null>;
  const unreliable = new Set(numberArray(source.unreliableIndices ?? [], "Unreliable bin indices"));
  const lengths = [labels, starts, ends, denominators, objectCounts, clusterCounts].map((items) => items.length);
  if (lengths.some((length) => length !== keys.length)) {
    throw new Error("Compact timeline arrays have different lengths.");
  }
  return keys.map((key, index) => ({
    key,
    label: labels[index],
    start: starts[index],
    end: ends[index],
    denominator: denominators[index],
    objectCount: objectCounts[index],
    clusterCount: clusterCounts[index],
    belowMinimumDenominator: unreliable.has(index),
  }));
}

function parsePointer(payload: unknown): CatalogPointer {
  const source = record(payload, "Concept catalog pointer");
  return {
    release: stringValue(source.release, "Concept catalog release"),
    ...(typeof source.complete === "boolean" ? { complete: source.complete } : {}),
    ...(typeof source.fullCatalog === "boolean" ? { fullCatalog: source.fullCatalog } : {}),
    ...(typeof source.catalogVersion === "string" ? { catalogVersion: source.catalogVersion } : {}),
  };
}

function parseManifest(payload: unknown): ReleaseManifest {
  const source = record(payload, "Concept release manifest");
  const corpus = record(source.corpus, "Concept release corpus");
  const model = record(source.model, "Concept release model");
  const metric = record(source.metric, "Concept release metric");
  const files = record(source.files, "Concept release files");
  const countingUnit = corpus.countingUnit;
  if (
    countingUnit !== "physical-object" &&
    countingUnit !== "visual-cluster" &&
    countingUnit !== "catalog-record"
  ) {
    throw new Error("Concept release has an unsupported counting unit.");
  }
  return {
    releaseFingerprint: stringValue(source.releaseFingerprint, "Release fingerprint"),
    catalogVersion: stringValue(source.catalogVersion, "Catalog version"),
    complete: source.complete === true,
    corpus: {
      id: stringValue(corpus.id, "Corpus id"),
      version: stringValue(corpus.version, "Corpus version"),
      label: stringValue(corpus.label, "Corpus label"),
      count: typeof corpus.count === "number" ? corpus.count : 0,
      countingUnit,
    },
    model: {
      id: stringValue(model.id, "Model id"),
      revision: stringValue(model.revision, "Model revision"),
      promptTemplateVersion: stringValue(model.promptTemplateVersion, "Prompt template version"),
    },
    metric: {
      id: stringValue(metric.id, "Metric id"),
      version: stringValue(metric.version, "Metric version"),
      qualifiedFraction: typeof metric.qualifiedFraction === "number" ? metric.qualifiedFraction : null,
    },
    files: {
      bins: stringValue(files.bins, "Bins file"),
      concepts: stringValue(files.concepts, "Concepts file"),
      seriesTemplate: stringValue(files.seriesTemplate, "Series template"),
      evidenceTemplate: stringValue(files.evidenceTemplate, "Evidence template"),
    },
    failures: Array.isArray(source.failures) ? source.failures as ReleaseManifest["failures"] : [],
  };
}

export async function loadConceptCatalog(options: {
  baseUrl?: string;
  fetch?: typeof fetch;
} = {}): Promise<ConceptCatalog> {
  const baseUrl = (options.baseUrl ?? "/catalog-data/v1").replace(/\/$/, "");
  const fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
  const cacheKey = baseUrl;
  if (catalogPromises.has(cacheKey)) return catalogPromises.get(cacheKey)!;
  const pending = (async () => {
    const pointer = parsePointer(await fetchJson(fetcher, `${baseUrl}/manifest.json`, "Concept catalog"));
    const releaseBaseUrl = assetUrl(baseUrl, pointer.release);
    const manifest = parseManifest(await fetchJson(fetcher, `${releaseBaseUrl}/manifest.json`, "Concept release"));
    const [conceptPayload, binPayload] = await Promise.all([
      fetchJson(fetcher, assetUrl(releaseBaseUrl, manifest.files.concepts), "Concept list"),
      fetchJson(fetcher, assetUrl(releaseBaseUrl, manifest.files.bins), "Timeline bins"),
    ]);
    const concepts = parseConcepts(conceptPayload);
    const bins = expandCompactBins(binPayload);
    const names: ConceptCatalog["names"] = new Map();
    const conceptsById = new Map<string, ConceptDefinition>();
    for (const concept of concepts) {
      conceptsById.set(concept.id, concept);
      const entries: Array<[string, "label" | "alias"]> = [
        [concept.label, "label"],
        ...concept.aliases.map((alias): [string, "alias"] => [alias, "alias"]),
      ];
      for (const [matchedText, matchedBy] of entries) {
        const normalized = normalizeQueryTerm(matchedText);
        const existing = names.get(normalized);
        if (existing && existing.concept.id !== concept.id) {
          throw new Error(`Concept name “${matchedText}” resolves to more than one concept.`);
        }
        names.set(normalized, { concept, matchedText, matchedBy });
      }
    }
    return {
      baseUrl,
      releaseBaseUrl,
      pointer,
      manifest,
      concepts,
      bins,
      fetcher,
      names,
      conceptsById,
      seriesCache: new Map(),
      evidenceCache: new Map(),
      defaultBinKeys: new Map(),
    };
  })();
  catalogPromises.set(cacheKey, pending);
  pending.catch(() => catalogPromises.delete(cacheKey));
  return pending;
}

export function resetConceptCatalogCache() {
  catalogPromises.clear();
}

export function resolveConcept(catalog: ConceptCatalog, input: string): ConceptResolution | null {
  const requested = input.trim().replace(/\s+/g, " ");
  const requestedNormalized = normalizeQueryTerm(requested);
  const match = catalog.names.get(requestedNormalized);
  return match ? { requested, requestedNormalized, ...match } : null;
}

function levenshtein(left: string, right: string) {
  const previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  for (let leftIndex = 1; leftIndex <= left.length; leftIndex += 1) {
    let diagonal = previous[0];
    previous[0] = leftIndex;
    for (let rightIndex = 1; rightIndex <= right.length; rightIndex += 1) {
      const above = previous[rightIndex];
      previous[rightIndex] = Math.min(
        previous[rightIndex] + 1,
        previous[rightIndex - 1] + 1,
        diagonal + (left[leftIndex - 1] === right[rightIndex - 1] ? 0 : 1),
      );
      diagonal = above;
    }
  }
  return previous[right.length];
}

export function suggestConcepts(catalog: ConceptCatalog, input: string, limit = 7): ConceptSuggestion[] {
  const needle = normalizeQueryTerm(input);
  const ranked = catalog.concepts.map((concept, catalogIndex) => {
    const candidates = [
      { text: concept.label, matchedBy: "label" as const },
      ...concept.aliases.map((text) => ({ text, matchedBy: "alias" as const })),
    ];
    const best = candidates.map((candidate) => {
      const normalized = normalizeQueryTerm(candidate.text);
      const exact = Boolean(needle) && normalized === needle;
      const prefix = needle && normalized.startsWith(needle);
      const contains = needle && normalized.includes(needle);
      const distance = needle ? levenshtein(needle, normalized) / Math.max(needle.length, normalized.length, 1) : 0;
      return { ...candidate, score: exact ? -4 : prefix ? -3 : contains ? -2 : distance };
    }).sort((left, right) => left.score - right.score || left.text.localeCompare(right.text))[0];
    return { concept, matchedText: best.text, matchedBy: best.matchedBy, score: best.score, catalogIndex };
  }).sort((left, right) => left.score - right.score || left.catalogIndex - right.catalogIndex);
  return ranked.slice(0, Math.max(0, limit)).map(({ score: _score, catalogIndex: _index, ...item }) => item);
}

function conceptFromInput(catalog: ConceptCatalog, input: string | ConceptDefinition | ConceptResolution) {
  if (typeof input !== "string") return "concept" in input ? input.concept : input;
  const byId = catalog.conceptsById.get(input.replace(/^concept:/, ""));
  if (byId) return byId;
  const resolved = resolveConcept(catalog, input);
  if (!resolved) throw new UnknownConceptError(input, suggestConcepts(catalog, input, 3));
  return resolved.concept;
}

function expandCompactSeries(catalog: ConceptCatalog, concept: ConceptDefinition, payload: unknown): LoadedSeries {
  const source = record(payload, `Series for ${concept.label}`) as unknown as CompactSeries;
  const pointIndices = numberArray(source.pointIndices, `${concept.label} point indices`);
  const values = numberArray(source.values, `${concept.label} values`);
  const shares = numberArray(source.shares, `${concept.label} shares`, true) as Array<number | null>;
  const hitMasses = numberArray(source.hitMasses, `${concept.label} hit masses`);
  const objectCounts = numberArray(source.objectCounts, `${concept.label} object counts`);
  const clusterCounts = numberArray(source.clusterCounts, `${concept.label} cluster counts`);
  if ([values, shares, hitMasses, objectCounts, clusterCounts].some((items) => items.length !== pointIndices.length)) {
    throw new Error(`Compact series arrays for ${concept.label} have different lengths.`);
  }
  const seen = new Set<number>();
  const points = pointIndices.map((binIndex, index) => {
    if (!Number.isInteger(binIndex) || !catalog.bins[binIndex] || seen.has(binIndex)) {
      throw new Error(`Series for ${concept.label} contains an invalid bin index.`);
    }
    seen.add(binIndex);
    return {
      binKey: catalog.bins[binIndex].key,
      value: values[index],
      share: shares[index],
      lift: values[index],
      hitMass: hitMasses[index],
      objectCount: objectCounts[index],
      clusterCount: clusterCounts[index],
    };
  });
  const suppressedIndices = numberArray(source.suppressedBinIndices ?? [], `${concept.label} suppressed indices`);
  const defaultIndex = typeof source.defaultEvidenceBinIndex === "number" ? source.defaultEvidenceBinIndex : null;
  const series: SearchSeries = {
    queryId: `concept:${concept.id}`,
    k: typeof source.k === "number" ? source.k : 0,
    threshold: typeof source.threshold === "number" ? source.threshold : null,
    ...(typeof source.candidateK === "number" ? { candidateK: source.candidateK } : {}),
    ...(typeof source.candidateThreshold === "number" || source.candidateThreshold === null
      ? { candidateThreshold: source.candidateThreshold }
      : {}),
    lowSignal: typeof source.lowSignal === "boolean" ? source.lowSignal : null,
    diagnostics: source.diagnostics && typeof source.diagnostics === "object"
      ? source.diagnostics
      : { standardizedSeparation: null, controlMean: null, controlStdDev: null, promptTopKJaccard: null, reasons: [] },
    points,
    suppressedBinKeys: suppressedIndices.flatMap((index) => catalog.bins[index] ? [catalog.bins[index].key] : []),
  };
  return { series, defaultBinKey: defaultIndex !== null ? catalog.bins[defaultIndex]?.key ?? null : null };
}

async function loadSeries(catalog: ConceptCatalog, concept: ConceptDefinition) {
  let pending = catalog.seriesCache.get(concept.id);
  if (!pending) {
    const relative = catalog.manifest.files.seriesTemplate.replace("{conceptId}", encodeURIComponent(concept.id));
    pending = fetchJson(catalog.fetcher, assetUrl(catalog.releaseBaseUrl, relative), `${concept.label} timeline`)
      .then((payload) => expandCompactSeries(catalog, concept, payload));
    catalog.seriesCache.set(concept.id, pending);
    pending.then((loaded) => catalog.defaultBinKeys.set(concept.id, loaded.defaultBinKey))
      .catch(() => catalog.seriesCache.delete(concept.id));
  }
  return pending;
}

export async function loadConceptSearch(
  catalog: ConceptCatalog,
  inputs: Array<string | ConceptDefinition | ConceptResolution>,
): Promise<SearchResponse> {
  const concepts = inputs.map((input) => conceptFromInput(catalog, input));
  const unique = concepts.filter((concept, index) => concepts.findIndex((item) => item.id === concept.id) === index);
  if (!unique.length) throw new Error("Add at least one visual concept.");
  if (unique.length > 5) throw new Error("Compare up to five visual concepts at a time.");
  const loaded = await Promise.all(unique.map((concept) => loadSeries(catalog, concept)));
  return {
    schemaVersion: "mnemosyne.search.v1",
    queries: unique.map((concept) => ({ id: `concept:${concept.id}`, label: concept.label, normalized: concept.normalized })),
    corpus: {
      ...catalog.manifest.corpus,
      view: "all",
      filters: {},
    },
    model: {
      id: catalog.manifest.model.id,
      version: catalog.manifest.model.revision,
      promptTemplateVersion: catalog.manifest.model.promptTemplateVersion,
    },
    metric: {
      id: catalog.manifest.metric.id,
      version: catalog.manifest.metric.version,
      label: "Filtered visual-match concentration",
      percentile: catalog.manifest.metric.qualifiedFraction,
      unit: "lift",
      description: "Zero means no score-qualified matches were found for that period; gaps mark periods with too little coverage or too few independent visual matches.",
    },
    bins: catalog.bins,
    series: loaded.map((item) => item.series),
    selectedEvidence: null,
    warnings: catalog.manifest.complete ? [] : ["This visual concept release is incomplete."],
    generatedAt: new Date().toISOString(),
  };
}

function parseEvidenceBundle(payload: unknown, expectedConceptId: string): EvidenceBundle {
  const source = record(payload, `Evidence for ${expectedConceptId}`);
  const conceptId = stringValue(source.conceptId, "Evidence concept id");
  if (conceptId !== expectedConceptId) throw new Error("Evidence bundle names a different concept.");
  const artworksSource = record(source.artworks, "Evidence artwork map");
  const artworks = artworksSource as EvidenceBundle["artworks"];
  if (!Array.isArray(source.periods)) throw new Error("Evidence bundle has no periods array.");
  const periods = source.periods.map((value, index): EvidencePeriod => {
    const period = record(value, `Evidence period ${index + 1}`);
    const artworkIds = stringArray(period.artworkIds, `Evidence period ${index + 1} artwork ids`);
    const contributionWeights = numberArray(period.contributionWeights, `Evidence period ${index + 1} weights`);
    if (artworkIds.length !== contributionWeights.length) {
      throw new Error("Evidence artwork ids and weights have different lengths.");
    }
    return {
      binIndex: typeof period.binIndex === "number" ? period.binIndex : -1,
      contributorCount: typeof period.contributorCount === "number" ? period.contributorCount : 0,
      artworkIds,
      contributionWeights,
    };
  });
  return {
    conceptId,
    percentile: typeof source.percentile === "number" ? source.percentile : null,
    threshold: typeof source.threshold === "number" ? source.threshold : null,
    artworks,
    periods,
  };
}

function emptySlices(): EvidenceSlices {
  return { strongest: [], representative: [], borderline: [], randomContributors: [], bestNonContributors: [], randomDenominator: [] };
}

export async function loadConceptEvidence(
  catalog: ConceptCatalog,
  conceptId: string,
  binKey: string,
): Promise<SelectedEvidence | null> {
  const stableId = conceptId.replace(/^concept:/, "");
  const concept = catalog.conceptsById.get(stableId);
  if (!concept) throw new UnknownConceptError(conceptId, suggestConcepts(catalog, conceptId, 3));
  let pending = catalog.evidenceCache.get(stableId);
  if (!pending) {
    const relative = catalog.manifest.files.evidenceTemplate.replace("{conceptId}", encodeURIComponent(stableId));
    pending = fetchJson(catalog.fetcher, assetUrl(catalog.releaseBaseUrl, relative), `${concept.label} evidence`)
      .then((payload) => parseEvidenceBundle(payload, stableId));
    catalog.evidenceCache.set(stableId, pending);
    pending.catch(() => catalog.evidenceCache.delete(stableId));
  }
  const bundle = await pending;
  const binIndex = catalog.bins.findIndex((bin) => bin.key === binKey);
  const period = bundle.periods.find((item) => item.binIndex === binIndex);
  if (!period) return null;
  const slices = emptySlices();
  slices.strongest = period.artworkIds.flatMap((artworkId, index) => {
    const artwork = bundle.artworks[artworkId];
    if (!artwork) return [];
    let imageUrl = artwork.imageUrl;
    if (imageUrl?.startsWith("/v1/images/")) {
      try {
        imageUrl = buildBackendImageUrl(decodeURIComponent(imageUrl.slice("/v1/images/".length)), "embedding");
      } catch {
        imageUrl = null;
      }
    }
    return [{ ...artwork, artworkId, imageUrl, contributionWeight: period.contributionWeights[index] } as EvidenceArtwork];
  });
  return {
    queryId: `concept:${stableId}`,
    binKey,
    percentile: bundle.percentile,
    threshold: bundle.threshold,
    contributorCount: period.contributorCount,
    slices,
  };
}

export function defaultConceptSelection(
  catalog: ConceptCatalog,
  response: SearchResponse,
  preferredConceptId?: string,
): ChartSelection | null {
  const preferred = preferredConceptId?.replace(/^concept:/, "");
  const ordered = preferred
    ? [...response.series].sort((left) => left.queryId === `concept:${preferred}` ? -1 : 1)
    : response.series;
  for (const series of ordered) {
    const stableId = series.queryId.replace(/^concept:/, "");
    const defaultBinKey = catalog.defaultBinKeys.get(stableId);
    if (defaultBinKey && series.points.some((point) => point.binKey === defaultBinKey)) {
      return { queryId: series.queryId, binKey: defaultBinKey };
    }
    const reliable = series.points.filter((point) =>
      response.bins.find((bin) => bin.key === point.binKey)?.belowMinimumDenominator !== true
    );
    if (reliable.length) {
      const peak = reliable.reduce((best, point) => point.value > best.value ? point : best);
      return { queryId: series.queryId, binKey: peak.binKey };
    }
  }
  return null;
}
