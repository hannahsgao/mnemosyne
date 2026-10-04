// Production analytics is intercepted by the Cloudflare Worker before this route.
// Keep local development fail-open so telemetry never disrupts the interface.
export function POST() {
  return new Response(null, {
    status: 204,
    headers: { "Cache-Control": "private, no-store" },
  });
}

export function GET() {
  return new Response(null, { status: 404 });
}
