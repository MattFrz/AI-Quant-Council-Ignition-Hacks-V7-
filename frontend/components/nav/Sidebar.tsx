"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/**
 * Persistent left rail.
 *
 * The three sections are views of ONE run, not separate tools, so they stay
 * visible rather than living behind tabs: the reader can see that a portfolio
 * and a shortlist exist while they are still reading the research. Each carries
 * a one-line description for the same reason, since "Opportunities" alone does
 * not say what is behind it.
 */
const SECTIONS = [
  { href: "/", label: "Research", hint: "Thesis to decision" },
  { href: "/opportunity", label: "Opportunities", hint: "Funnel and shortlist" },
  { href: "/portfolio", label: "Portfolio", hint: "Sized, or refused" },
  { href: "/execution", label: "Execution", hint: "C++ order book" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar">
      <div className="sidebar-inner">
        <Link href="/" className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-text">
            AI Quant
            <br />
            Council
          </span>
        </Link>

        <nav aria-label="Sections">
          <ul className="side-nav">
            {SECTIONS.map((s) => {
              const active = pathname === s.href;
              return (
                <li key={s.href}>
                  <Link
                    href={s.href}
                    className="side-link"
                    data-active={active || undefined}
                    aria-current={active ? "page" : undefined}
                  >
                    <span className="side-label">{s.label}</span>
                    <span className="side-hint">{s.hint}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        <p className="sidebar-foot">
          Research and evidence.
          <br />
          Not investment advice.
        </p>
      </div>
    </aside>
  );
}
