const REPO = "https://github.com/MattFrz/AI-Quant-Council-Ignition-Hacks-V7-";
const TEAM = ["Matt", "Nalin", "Zain", "Cecile"];

/**
 * Attribution and the disclaimer, on every page.
 *
 * The disclaimer sits here rather than only on the research screen because the
 * portfolio page is the one that shows position sizes, and that is the page
 * most likely to be read as instructions.
 */
export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="container site-footer-inner">
        <span>
          &copy; {new Date().getFullYear()} {TEAM.join(", ")}
        </span>
        <span className="text-muted">
          Built for IgnitionHacks V7 &middot; research and evidence, not
          investment advice
        </span>
        <a href={REPO} target="_blank" rel="noopener noreferrer">
          Source on GitHub &#8599;
        </a>
      </div>
    </footer>
  );
}
