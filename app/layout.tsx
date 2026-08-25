import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

const TITLE = "Mnemosyne — Trace visual ideas through art history";
const DESCRIPTION =
  "Search open-access museum collections by metadata or any visual idea, then trace results through time and inspect the artworks behind every signal.";
const FALLBACK_ORIGIN = "https://mnemosyne.hannahgao.studio";

function requestOrigin(requestHeaders: Pick<Headers, "get">) {
  const forwardedHost = requestHeaders.get("x-forwarded-host")?.split(",")[0]?.trim();
  const host = forwardedHost || requestHeaders.get("host")?.trim();
  if (!host || !/^(?:[a-z0-9-]+\.)*[a-z0-9-]+(?::\d+)?$/i.test(host)) {
    return new URL(FALLBACK_ORIGIN);
  }
  const forwardedProtocol = requestHeaders.get("x-forwarded-proto")?.split(",")[0]?.trim();
  const local = host.startsWith("localhost") || host.startsWith("127.0.0.1");
  const protocol = forwardedProtocol === "http" || forwardedProtocol === "https"
    ? forwardedProtocol
    : local ? "http" : "https";
  return new URL(`${protocol}://${host}`);
}

export async function generateMetadata(): Promise<Metadata> {
  const origin = requestOrigin(await headers());
  const socialImage = new URL("/og.png", origin).toString();
  return {
    title: TITLE,
    description: DESCRIPTION,
    openGraph: {
      type: "website",
      title: TITLE,
      description: DESCRIPTION,
      siteName: "Mnemosyne",
      url: origin,
      images: [{
        url: socialImage,
        width: 1731,
        height: 909,
        alt: "Mnemosyne visual-search timeline with archival art imagery",
      }],
    },
    twitter: {
      card: "summary_large_image",
      title: TITLE,
      description: DESCRIPTION,
      images: [socialImage],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
