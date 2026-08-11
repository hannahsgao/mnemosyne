import assert from "node:assert/strict";
import test from "node:test";
import {
  IMMUTABLE_RELEASE_CACHE_CONTROL,
  serveImmutableReleaseAsset,
} from "./static-release.ts";

test("fingerprinted release assets receive immutable browser caching", async () => {
  const request = new Request(
    "https://mnemosyne.example/data/v1/releases/fingerprint/series/horse.json",
  );
  let calls = 0;
  const response = await serveImmutableReleaseAsset(request, async () => {
    calls += 1;
    return Response.json(
      { values: [1, 2, 3] },
      { headers: { ETag: '"release-etag"', "Cache-Control": "public, max-age=0" } },
    );
  });

  assert.equal(calls, 1);
  assert.ok(response);
  assert.equal(response.headers.get("Cache-Control"), IMMUTABLE_RELEASE_CACHE_CONTROL);
  assert.equal(response.headers.get("X-Content-Type-Options"), "nosniff");
  assert.equal(response.headers.get("ETag"), '"release-etag"');
  assert.deepEqual(await response.json(), { values: [1, 2, 3] });
});

test("mutable pointers and non-read methods bypass the release asset binding", async () => {
  let calls = 0;
  const fetchAsset = async () => {
    calls += 1;
    return new Response("unexpected");
  };

  assert.equal(
    await serveImmutableReleaseAsset(
      new Request("https://mnemosyne.example/data/v1/manifest.json"),
      fetchAsset,
    ),
    null,
  );
  assert.equal(
    await serveImmutableReleaseAsset(
      new Request("https://mnemosyne.example/data/v1/releases/fingerprint/series/horse.json", {
        method: "POST",
      }),
      fetchAsset,
    ),
    null,
  );
  assert.equal(calls, 0);
});

test("missing release assets retain the asset service error response", async () => {
  const response = await serveImmutableReleaseAsset(
    new Request("https://mnemosyne.example/data/v1/releases/missing/series/horse.json"),
    async () => new Response("missing", { status: 404, headers: { "Cache-Control": "no-store" } }),
  );

  assert.ok(response);
  assert.equal(response.status, 404);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
});

test("not-modified release responses renew the immutable cache policy", async () => {
  const response = await serveImmutableReleaseAsset(
    new Request("https://mnemosyne.example/data/v1/releases/fingerprint/series/horse.json", {
      headers: { "If-None-Match": '"release-etag"' },
    }),
    async () => new Response(null, {
      status: 304,
      headers: { ETag: '"release-etag"', "Cache-Control": "public, max-age=0" },
    }),
  );

  assert.ok(response);
  assert.equal(response.status, 304);
  assert.equal(response.body, null);
  assert.equal(response.headers.get("Cache-Control"), IMMUTABLE_RELEASE_CACHE_CONTROL);
  assert.equal(response.headers.get("ETag"), '"release-etag"');
});
