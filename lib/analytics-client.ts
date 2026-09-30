import {
  ANALYTICS_SCHEMA_VERSION,
  type AnalyticsEnvelope,
  type AnalyticsEvent,
  type AnalyticsEventName,
  type AnalyticsEventProperties,
  type SearchScopedAnalyticsEventName,
} from "./analytics-events.ts";

const ANALYTICS_ENDPOINT = "/api/analytics";
const ANALYTICS_SESSION_KEY = "mnemosyne.analytics.session.v1";
const DEFAULT_BATCH_SIZE = 20;
const DEFAULT_MAX_QUEUE_SIZE = 64;
const DEFAULT_FLUSH_DELAY_MS = 2_000;
const FIRST_PARTY_MIRROR_ENABLED =
  process.env.NEXT_PUBLIC_MNEMOSYNE_ANALYTICS_D1_MIRROR === "true";

type AnalyticsPrimitive = string | number | boolean;
type AnalyticsTracker = (
  eventName: string,
  properties?: Record<string, AnalyticsPrimitive>,
) => void | Promise<unknown>;

type NavigatorPrivacy = Pick<Navigator, "doNotTrack" | "webdriver"> & {
  globalPrivacyControl?: boolean;
};

type AnalyticsClientOptions = {
  sessionId: string;
  viewId: string;
  createId: () => string;
  send: (envelope: AnalyticsEnvelope, final: boolean) => void;
  schedule: (callback: () => void) => unknown;
  cancel: (token: unknown) => void;
  trackExternal?: (event: AnalyticsEvent) => void;
  queueEnabled?: boolean;
  batchSize?: number;
  maxQueueSize?: number;
};

declare global {
  interface Navigator {
    globalPrivacyControl?: boolean;
  }

  interface Window {
    doNotTrack?: string;
    zaraz?: {
      track: AnalyticsTracker;
    };
  }
}

export function analyticsPrivacyAllowed(
  navigatorValue: NavigatorPrivacy,
  windowDoNotTrack?: string,
) {
  return (
    navigatorValue.globalPrivacyControl !== true &&
    navigatorValue.doNotTrack !== "1" &&
    windowDoNotTrack !== "1" &&
    navigatorValue.webdriver !== true
  );
}

export function createAnalyticsId(cryptoValue: Pick<Crypto, "getRandomValues" | "randomUUID">) {
  if (typeof cryptoValue.randomUUID === "function") return cryptoValue.randomUUID();
  const bytes = cryptoValue.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}

export class AnalyticsClient {
  private readonly batchSize: number;
  private readonly cancel: (token: unknown) => void;
  private readonly createId: () => string;
  private readonly maxQueueSize: number;
  private readonly queueEnabled: boolean;
  private readonly schedule: (callback: () => void) => unknown;
  private readonly send: (envelope: AnalyticsEnvelope, final: boolean) => void;
  private readonly sessionId: string;
  private readonly trackExternal?: (event: AnalyticsEvent) => void;
  private readonly viewId: string;
  private readonly onceKeys = new Set<string>();
  private queue: AnalyticsEvent[] = [];
  private scheduledFlush: unknown = null;
  private sequence = 0;

  constructor(options: AnalyticsClientOptions) {
    this.sessionId = options.sessionId;
    this.viewId = options.viewId;
    this.createId = options.createId;
    this.send = options.send;
    this.schedule = options.schedule;
    this.cancel = options.cancel;
    this.trackExternal = options.trackExternal;
    this.queueEnabled = options.queueEnabled ?? true;
    this.batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;
    this.maxQueueSize = options.maxQueueSize ?? DEFAULT_MAX_QUEUE_SIZE;
  }

  track(
    name: "page_view",
    properties: AnalyticsEventProperties["page_view"],
  ): AnalyticsEvent<"page_view">;
  track<K extends SearchScopedAnalyticsEventName>(
    name: K,
    properties: AnalyticsEventProperties[K],
    searchId: string,
  ): AnalyticsEvent<K>;
  track<K extends AnalyticsEventName>(
    name: K,
    properties: AnalyticsEventProperties[K],
    searchId?: string,
  ): AnalyticsEvent<K> {
    const event: AnalyticsEvent<K> = {
      id: this.createId(),
      sequence: ++this.sequence,
      name,
      ...(searchId ? { searchId } : {}),
      properties,
    } as AnalyticsEvent<K>;
    try {
      this.trackExternal?.(event as AnalyticsEvent);
    } catch {
      // Analytics must never affect the interaction that emitted it.
    }

    if (!this.queueEnabled) return event;
    if (this.queue.length >= this.maxQueueSize) this.queue.shift();
    this.queue.push(event as AnalyticsEvent);
    if (this.queue.length >= this.batchSize) {
      this.flush(false);
    } else {
      this.ensureScheduledFlush();
    }
    return event;
  }

  trackOnce(
    key: string,
    name: "page_view",
    properties: AnalyticsEventProperties["page_view"],
  ): AnalyticsEvent<"page_view"> | null;
  trackOnce<K extends SearchScopedAnalyticsEventName>(
    key: string,
    name: K,
    properties: AnalyticsEventProperties[K],
    searchId: string,
  ): AnalyticsEvent<K> | null;
  trackOnce<K extends AnalyticsEventName>(
    key: string,
    name: K,
    properties: AnalyticsEventProperties[K],
    searchId?: string,
  ): AnalyticsEvent<K> | null {
    if (this.onceKeys.has(key)) return null;
    this.onceKeys.add(key);
    return this.trackEvent(name, properties, searchId);
  }

  private trackEvent<K extends AnalyticsEventName>(
    name: K,
    properties: AnalyticsEventProperties[K],
    searchId?: string,
  ) {
    return searchId
      ? this.track(
          name as SearchScopedAnalyticsEventName,
          properties as AnalyticsEventProperties[SearchScopedAnalyticsEventName],
          searchId,
        ) as AnalyticsEvent<K>
      : this.track(
          "page_view",
          properties as AnalyticsEventProperties["page_view"],
        ) as AnalyticsEvent<K>;
  }

  flush(final = false) {
    if (this.scheduledFlush !== null) {
      this.cancel(this.scheduledFlush);
      this.scheduledFlush = null;
    }

    do {
      const events = this.queue.splice(0, this.batchSize);
      if (!events.length) break;
      try {
        this.send({
          schemaVersion: ANALYTICS_SCHEMA_VERSION,
          sessionId: this.sessionId,
          viewId: this.viewId,
          events,
        }, final);
      } catch {
        // Best-effort telemetry is intentionally fail-open and never retried here.
      }
    } while (final && this.queue.length > 0);

    if (this.queue.length > 0) this.ensureScheduledFlush();
  }

  private ensureScheduledFlush() {
    if (this.scheduledFlush !== null) return;
    this.scheduledFlush = this.schedule(() => {
      this.scheduledFlush = null;
      this.flush(false);
    });
  }
}

let browserClient: AnalyticsClient | null | undefined;
let browserListenersAttached = false;

function sessionId(createId: () => string) {
  try {
    const existing = window.sessionStorage.getItem(ANALYTICS_SESSION_KEY);
    if (existing) return existing;
    const created = createId();
    window.sessionStorage.setItem(ANALYTICS_SESSION_KEY, created);
    return created;
  } catch {
    return createId();
  }
}

function zarazProperties(event: AnalyticsEvent) {
  const values: Record<string, AnalyticsPrimitive> = {
    analytics_schema: ANALYTICS_SCHEMA_VERSION,
    sequence: event.sequence,
    ...(event.searchId ? { search_id: event.searchId } : {}),
  };
  for (const [key, value] of Object.entries(event.properties)) {
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      values[key] = value;
    }
  }
  return values;
}

export function cloudflareAnalyticsEventName(event: AnalyticsEvent) {
  const properties = event.properties as Record<string, unknown>;
  if (event.name === "search_attempt") {
    if (properties.outcome === "invalid") return "mnemosyne_search_invalid";
    return properties.source === "initial_default" || properties.source === "url_restore"
      ? "mnemosyne_search_automatic"
      : "mnemosyne_search_user";
  }
  if (event.name === "search_result") {
    return `mnemosyne_search_result_${String(properties.outcome)}`;
  }
  if (event.name === "evidence_result") {
    return `mnemosyne_evidence_${String(properties.outcome)}`;
  }
  if (event.name === "series_toggle") {
    return `mnemosyne_series_${String(properties.action)}`;
  }
  if (event.name === "help_open") {
    return `mnemosyne_help_${String(properties.target)}`;
  }
  return `mnemosyne_${event.name}`;
}

function createBrowserClient() {
  if (
    typeof window === "undefined" ||
    typeof document === "undefined" ||
    !analyticsPrivacyAllowed(window.navigator, window.doNotTrack)
  ) {
    return null;
  }

  const createId = () => createAnalyticsId(window.crypto);
  const client = new AnalyticsClient({
    sessionId: sessionId(createId),
    viewId: createId(),
    createId,
    schedule: (callback) => window.setTimeout(callback, DEFAULT_FLUSH_DELAY_MS),
    cancel: (token) => window.clearTimeout(token as number),
    send: (envelope, final) => {
      const body = JSON.stringify(envelope);
      if (
        final &&
        typeof window.navigator.sendBeacon === "function" &&
        window.navigator.sendBeacon(
          ANALYTICS_ENDPOINT,
          new Blob([body], { type: "application/json;charset=UTF-8" }),
        )
      ) {
        return;
      }
      void window.fetch(ANALYTICS_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        cache: "no-store",
        credentials: "omit",
        keepalive: true,
      }).catch(() => undefined);
    },
    trackExternal: (event) => {
      const zaraz = window.zaraz;
      if (!zaraz || typeof zaraz.track !== "function") return;
      const tracked = zaraz.track(cloudflareAnalyticsEventName(event), zarazProperties(event));
      if (tracked && typeof tracked.catch === "function") {
        void tracked.catch(() => undefined);
      }
    },
    queueEnabled: FIRST_PARTY_MIRROR_ENABLED,
  });

  if (FIRST_PARTY_MIRROR_ENABLED && !browserListenersAttached) {
    browserListenersAttached = true;
    window.addEventListener("pagehide", () => browserClient?.flush(true), { capture: true });
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") browserClient?.flush(true);
    });
  }
  return client;
}

function getBrowserClient() {
  if (browserClient === undefined) browserClient = createBrowserClient();
  return browserClient;
}

export function trackSearchAnalytics<K extends SearchScopedAnalyticsEventName>(
  name: K,
  properties: AnalyticsEventProperties[K],
  searchId?: string,
) {
  if (!searchId) return null;
  return getBrowserClient()?.track(name, properties, searchId) ?? null;
}

export function trackPageAnalyticsOnce(
  key: string,
  properties: AnalyticsEventProperties["page_view"],
) {
  return getBrowserClient()?.trackOnce(key, "page_view", properties) ?? null;
}

export function trackSearchAnalyticsOnce<K extends SearchScopedAnalyticsEventName>(
  key: string,
  name: K,
  properties: AnalyticsEventProperties[K],
  searchId?: string,
) {
  if (!searchId) return null;
  return getBrowserClient()?.trackOnce(key, name, properties, searchId) ?? null;
}
