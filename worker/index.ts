import {
  DEFAULT_DEVICE_SIZES,
  DEFAULT_IMAGE_SIZES,
  handleImageOptimization,
} from "vinext/server/image-optimization";
import handler from "vinext/server/app-router-entry";
import { serveCatalogAsset } from "../lib/static-release";
import {
  shouldStoreVisualEdgeResponse,
  visualEdgeCacheKey,
  withVisualEdgeCacheStatus,
} from "../lib/visual-edge-cache";
import { handleMetServiceRequest, type D1Database } from "./met-search";

declare const caches: CacheStorage & { default: Cache };

interface Env {
  ASSETS: { fetch(request: Request): Promise<Response> };
  DB: D1Database;
  MNEMOSYNE_IMPORT_TOKEN?: string;
  IMAGES: {
    input(stream: ReadableStream): {
      transform(options: Record<string, unknown>): {
        output(options: { format: string; quality: number }): Promise<{ response(): Response }>;
      };
    };
  };
}

interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
  passThroughOnException(): void;
}

const worker = {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const visualCacheKey = visualEdgeCacheKey(request);
    if (visualCacheKey) {
      try {
        const cached = await caches.default.match(visualCacheKey);
        if (cached) return withVisualEdgeCacheStatus(cached, "HIT");
      } catch {
        // Cache availability must never make search unavailable.
      }
    }
    const releaseAsset = await serveCatalogAsset(
      request,
      env.ASSETS.fetch.bind(env.ASSETS),
    );
    if (releaseAsset) return releaseAsset;
    const metResponse = await handleMetServiceRequest(request, env);
    if (metResponse) return metResponse;
    if (url.pathname === "/_vinext/image") {
      return handleImageOptimization(
        request,
        {
          fetchAsset: (path) => env.ASSETS.fetch(new Request(new URL(path, request.url))),
          transformImage: async (body, { width, format, quality }) => {
            const result = await env.IMAGES.input(body)
              .transform(width > 0 ? { width } : {})
              .output({ format, quality });
            return result.response();
          },
        },
        [...DEFAULT_DEVICE_SIZES, ...DEFAULT_IMAGE_SIZES],
      );
    }
    const response = await handler.fetch(request, env, ctx);
    if (visualCacheKey && shouldStoreVisualEdgeResponse(response)) {
      try {
        // Await the local edge write so MISS means the response is available to
        // the next request in this data center. A cache failure still fails open.
        await caches.default.put(visualCacheKey, response.clone());
        return withVisualEdgeCacheStatus(response, "MISS");
      } catch (error) {
        console.warn(
          "visual_edge_cache_write_failed",
          error instanceof Error ? error.message : "unknown cache error",
        );
        return withVisualEdgeCacheStatus(response, "BYPASS");
      }
    }
    return response;
  },
};

export default worker;
