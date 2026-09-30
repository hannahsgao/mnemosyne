# Usage analytics

Mnemosyne uses a Cloudflare-first event taxonomy designed to answer product questions without collecting search text, artwork titles, full URLs, or full referrers. The default production path is Zaraz-only. An exact-event D1 mirror exists as an opt-in diagnostic path and is disabled by default so analytics writes cannot contend with metadata search reads.

## What is measured

| Event | Meaning | Useful dimensions |
| --- | --- | --- |
| `mnemosyne_page_view` | One explorer page load | initial source, search mode, coarse viewport, coarse referrer category |
| `mnemosyne_search_user`, `mnemosyne_search_automatic`, `mnemosyne_search_invalid` | A submitted, automatic, or rejected search | source, mode, accepted/invalid, term count, query-length bucket |
| `mnemosyne_search_result_<outcome>` | A completed, failed, or aborted search | outcome is `timeline`, `empty`, `nearest_only`, `error`, or `aborted`; properties add duration, counts, mode, transport, cache status, and corpus/model versions |
| `mnemosyne_point_select` | A chart point selected | mode, pointer/keyboard, period key, series position |
| `mnemosyne_evidence_<outcome>` | Evidence served from the current response or fetched | outcome is `success`, `empty`, `error`, or `aborted`; properties add source, duration, cached, and result count |
| `mnemosyne_artwork_open` | A museum record link opened | evidence/nearest placement, rank, institution, image/contributor flags |
| `mnemosyne_series_activate` | A chart series focused | legend/endpoint, pointer/keyboard, series position |
| `mnemosyne_series_show`, `mnemosyne_series_hide` | A chart series shown or hidden | pointer/keyboard, series position, visible-series count |
| `mnemosyne_help_calculation`, `mnemosyne_help_chart_reading` | Help became visible | deduplicated once per search and help target |
| `mnemosyne_gallery_depth` | A later five-card evidence batch became visible | committed visible-card count and selected period |

`search_attempt.source` distinguishes user searches (`form`, `example`, and `mode_switch`) from automatic searches (`initial_default` and `url_restore`). Do not report all accepted attempts as “user searches.”

Hover requests, pointer movement, chart panning/zooming, image loads, raw query strings, deterministic query IDs, artwork IDs, titles, exact viewport sizes, IP addresses, and full referrers are intentionally not included in custom event payloads.

## Cloudflare dashboards

Cloudflare Web Analytics/RUM is already enabled on the `hannahgao.studio` zone. Use it for traffic, referrers, browsers, geography, and Web Vitals. Cloudflare Web Analytics does not support custom events, so searches and clicks use Zaraz Monitoring instead.

After Zaraz is enabled and the instrumented site version is deployed:

1. In Cloudflare, open **Tag Management → Zaraz** and select `hannahgao.studio`.
2. Open **Monitoring → Events**.
3. Filter for the `mnemosyne_` prefix. User search totals are `mnemosyne_search_user`; automatic/default searches are `mnemosyne_search_automatic`; outbound clicks are `mnemosyne_artwork_open`.
4. Search and evidence outcomes are encoded in the event name so the standard Monitoring view can count them without Advanced Monitoring. The flat properties remain available to Zaraz triggers and tools. Use the optional D1 report when exact dimensional breakdowns are required without sending data to another analytics provider.

The Sites custom hostname is a DNS-only CNAME, so Cloudflare cannot auto-inject
Zaraz into its HTML. Production loads Zaraz manually from the proxied
`hannahgao.studio` apex immediately before `</head>`. Keep zone auto-injection
off so this configuration does not also instrument the portfolio site. The
cross-origin script uses `referrerpolicy="origin"`, which prevents the
shareable query from entering the script request.

Cloudflare requires at least one enabled Zaraz tool before the loader is
available. Use an inert Custom HTML bootstrap with no external requests or
active actions. Advanced Monitoring is not required for the event-name totals
above and should remain disabled because it expands the collected dimensions.

## Required Zaraz configuration

These are launch requirements, not optional tuning:

- Turn on **Remove URL query parameters** before enabling Zaraz. Mnemosyne's shareable URL stores the search in `?q=`, so enabling Zaraz first could disclose raw search text through its automatic page context.
- Turn on **Trim IP addresses**, **Clean User Agent strings**, and **Remove external referrers**.
- Keep **Auto-inject script** off and load the script manually from the proxied
  apex as implemented in `app/layout.tsx`.
- Keep iframe injection off.
- Turn off Single Page Application pageview tracking for this site. Search state uses `history.replaceState`; those URL changes are searches, not new page views, and are already measured explicitly.
- Use **Block automated and likely automated** (or at least **Block automated only**) for the bot-score threshold.
- Do not configure blocking triggers or wait for actions. Custom calls are intentionally fire-and-forget.
- Decide whether Cloudflare's automatic Pageview event requires consent for the site's audience. The app's custom analytics honors Global Privacy Control, Do Not Track, and automated-browser signals; Cloudflare's independently injected pageview must be covered by the Zaraz consent configuration if required.

Useful Cloudflare references:

- [Zaraz track API](https://developers.cloudflare.com/zaraz/web-api/track/)
- [Zaraz Monitoring](https://developers.cloudflare.com/zaraz/monitoring/)
- [Zaraz privacy and injection settings](https://developers.cloudflare.com/zaraz/reference/settings/)
- [Zaraz setup requirements](https://developers.cloudflare.com/zaraz/faq/#why-is-zaraz-not-working)
- [Zaraz on domains not proxied by Cloudflare](https://developers.cloudflare.com/zaraz/advanced/domains-not-proxied/)
- [Cloudflare Web Analytics FAQ](https://developers.cloudflare.com/web-analytics/faq/)

## Performance behavior

The normal path calls `zaraz.track()` without awaiting it. There is no third-party analytics SDK in the bundle, no tracking in high-frequency interaction paths, and no analytics call can block a search, selection, or navigation. Cloudflare describes Zaraz as edge-executed with a near-zero performance hit; this repository describes the implementation as minimal and non-blocking rather than literally zero-cost.

The first-party D1 queue is not even scheduled unless `NEXT_PUBLIC_MNEMOSYNE_ANALYTICS_D1_MIRROR=true` at build time. Its Worker sink separately requires `MNEMOSYNE_ANALYTICS_D1_MIRROR=true`, validates a maximum 24 KiB/20-event same-origin batch, returns `202` before persistence, and writes through `ctx.waitUntil()`.

## Optional exact D1 report

Only enable the D1 mirror if exact, queryable event records are worth the additional writes to the D1 database that also serves metadata search:

```dotenv
NEXT_PUBLIC_MNEMOSYNE_ANALYTICS_D1_MIRROR=true
MNEMOSYNE_ANALYTICS_D1_MIRROR=true
MNEMOSYNE_ANALYTICS_TOKEN=<long-random-server-secret>
```

After applying `drizzle/0002_usage_analytics.sql` and deploying with those settings, request a privacy-bounded report with:

```sh
curl -H "Authorization: Bearer $MNEMOSYNE_ANALYTICS_TOKEN" \
  "https://mnemosyne.hannahgao.studio/_admin/analytics?days=30"
```

The report separates user and automatic searches and returns daily totals, event counts, search outcomes/durations, artwork opens, and evidence outcomes. It is `private, no-store`; missing configuration returns `404`, and invalid authorization returns `401`.

The D1 report always queries at most the last 90 days. Old rows are purged on the next valid mirrored ingestion or authenticated report, with a persisted once-per-day guard. Because Sites does not currently expose a scheduled D1 cleanup binding here, this is activity-triggered retention rather than a wall-clock deletion guarantee.

If the mirror is enabled, add a Cloudflare rate-limiting/WAF rule for `POST /api/analytics`. Same-origin browser headers and a strict schema reduce accidental or cross-site writes, but public browser telemetry cannot authenticate a determined non-browser client.

## Verification

Run:

```sh
npm run check
npm test
npm run build:sites
```

The analytics tests cover batching, final-page flushes, opt-out signals, Zaraz-only mode, bounded queues, all ten client/Worker event contracts, streaming body limits, origin validation, D1 migration idempotency, background writes, and private report authorization.
