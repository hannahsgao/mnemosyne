import {
  DEFAULT_DEVICE_SIZES,
  DEFAULT_IMAGE_SIZES,
  handleImageOptimization,
} from "vinext/server/image-optimization";
import handler from "vinext/server/app-router-entry";
import { serveCatalogAsset } from "../lib/static-release";
import {
  checkVisualRateLimit,
  type RateLimitBinding,
} from "../lib/visual-rate-limit";
import { handleMetServiceRequest, type D1Database } from "./met-search";

interface Env {
  ASSETS: { fetch(request: Request): Promise<Response> };
  DB: D1Database;
  VISUAL_RATE_LIMITER?: RateLimitBinding;
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
    const rateLimitResponse = await checkVisualRateLimit(
      request,
      env.VISUAL_RATE_LIMITER,
    );
    if (rateLimitResponse) return rateLimitResponse;
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
    return handler.fetch(request, env, ctx);
  },
};

export default worker;
