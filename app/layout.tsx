import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Mnemosyne — search visual culture across time",
  description:
    "Explore how visual ideas appear across art history and inspect the artworks behind every signal.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
