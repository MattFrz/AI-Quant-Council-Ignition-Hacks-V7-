import type { Metadata } from "next";
import "./globals.css";
import "./components.css";
import { Sidebar } from "../components/nav/Sidebar";

export const metadata: Metadata = {
  title: "AI Quant Council",
  description:
    "Turn an investment thesis into evidence, debate, quantitative validation, and an auditable decision.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/*
          Loaded by link rather than next/font on purpose: next/font resolves at
          BUILD time, so a build without network access fails outright. The
          token file already declares fallback stacks, so a blocked request
          degrades to system fonts instead of breaking the page.
        */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
        />
        <style>{`
          :root {
            --font-inter: "Inter";
            --font-mono-jb: "JetBrains Mono";
          }
        `}</style>
      </head>
      <body>
        <div className="shell">
          <Sidebar />
          <div className="shell-main">{children}</div>
        </div>
      </body>
    </html>
  );
}
