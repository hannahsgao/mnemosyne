export type QueryDescriptor = {
  id: string;
  label: string;
  normalized: string;
};

export type CorpusMetadata = {
  id: string;
  version: string;
  label: string;
  count: number | null;
  countingUnit: "physical-object" | "visual-cluster";
  view: string;
  filters: Record<string, string[]>;
};

export type ModelMetadata = {
  id: string;
  version: string;
  promptTemplateVersion: string;
};

export type MetricMetadata = {
  id: string;
  version: string;
  label: string;
  percentile: number | null;
  unit: "lift" | "relative-density" | "frequency";
  description?: string;
};

export type TimeBin = {
  key: string;
  label: string;
  start: number;
  end: number;
  denominator: number | null;
  objectCount: number | null;
  clusterCount: number | null;
  belowMinimumDenominator: boolean | null;
};

export type SeriesPoint = {
  binKey: string;
  value: number;
  share: number | null;
  lift: number | null;
  hitMass: number;
  objectCount: number;
  clusterCount: number;
};

export type SeriesDiagnostics = {
  standardizedSeparation: number | null;
  controlMean: number | null;
  controlStdDev: number | null;
  promptTopKJaccard: number | null;
  reasons: string[];
};

export type SearchSeries = {
  queryId: string;
  k: number;
  threshold: number | null;
  lowSignal: boolean | null;
  diagnostics: SeriesDiagnostics;
  points: SeriesPoint[];
  cacheKey?: string;
  /** Available on the live metadata adapter, where the catalogue exposes this count. */
  totalMatches?: number;
};

export type EvidenceSliceName =
  | "strongest"
  | "representative"
  | "borderline"
  | "randomContributors"
  | "bestNonContributors"
  | "randomDenominator";

export type EvidenceArtwork = {
  artworkId: string;
  physicalObjectId: string;
  visualClusterId: string;
  institution: string;
  title: string;
  artist: string;
  dateDisplay: string;
  dateStart: number | null;
  dateEnd: number | null;
  dateQualifier: string;
  imageUrl: string | null;
  sourceRecordUrl: string;
  metadataLicense: string;
  imageRightsUri: string;
  creditLine: string;
  publicDomain: boolean;
  rawScore: number | null;
  contributionWeight: number;
  contributor: boolean;
};

export type EvidenceSlices = Record<EvidenceSliceName, EvidenceArtwork[]>;

export type SelectedEvidence = {
  queryId: string;
  binKey: string;
  slices: EvidenceSlices;
};

export type SearchResponse = {
  schemaVersion: "mnemosyne.search.v1";
  queries: QueryDescriptor[];
  corpus: CorpusMetadata;
  model: ModelMetadata;
  metric: MetricMetadata;
  bins: TimeBin[];
  series: SearchSeries[];
  selectedEvidence: SelectedEvidence | null;
  warnings: string[];
  generatedAt: string;
};

export type ChartSelection = {
  queryId: string;
  binKey: string;
};
