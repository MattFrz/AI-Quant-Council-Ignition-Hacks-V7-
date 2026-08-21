// Placeholder root layout so `npm run dev` runs. Cecile replaces this in step D3.
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Autonomous Alpha",
  description: "Autonomous quant research platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
