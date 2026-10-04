import handler from "vinext/server/app-router-entry";
import { serveCatalogAsset } from "../lib/static-release";
import { handleAnalyticsRequest } from "./analytics";
import { handleMetServiceRequest, type D1Database } from "./met-search";

interface Env {
  ASSETS: { fetch(request: Request): Promise<Response> };
  DB: D1Database;
  MNEMOSYNE_ANALYTICS_D1_MIRROR?: string;
  MNEMOSYNE_ANALYTICS_TOKEN?: string;
  MNEMOSYNE_IMPORT_TOKEN?: string;
}

interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
  passThroughOnException(): void;
}

const worker = {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/api/analytics" || url.pathname === "/_admin/analytics") {
      const analyticsResponse = await handleAnalyticsRequest(request, env, ctx, url.pathname);
      if (analyticsResponse) return analyticsResponse;
    }
    const releaseAsset = await serveCatalogAsset(
      request,
      env.ASSETS.fetch.bind(env.ASSETS),
    );
    if (releaseAsset) return releaseAsset;
    const metResponse = await handleMetServiceRequest(request, env);
    if (metResponse) return metResponse;
    return handler.fetch(request, env, ctx);
  },
};

export default worker;
