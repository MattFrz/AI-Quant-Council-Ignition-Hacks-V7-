import { shortDate } from "../../lib/format";

/**
 * A catalyst's link back to the filing it came from.
 *
 * This is the component that separates the product from a chatbot with a
 * ticker, so it always renders the destination as a real anchor with a real
 * href. A catalyst that reaches here without a source_url is a contract
 * violation upstream, not something to paper over with a disabled span - so
 * it renders as an explicit warning instead.
 */
export function SourceLink({
  url,
  sourceType,
  sourceDate,
}: {
  url: string | null;
  sourceType: string;
  sourceDate: string;
}) {
  const label = sourceType.replace(/_/g, " ");

  if (!url) {
    return (
      <span className="text-negative" title="Catalyst arrived without a source URL">
        no source link
      </span>
    );
  }

  return (
    <a
      className="source-link"
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      title={url}
    >
      {label} · {shortDate(sourceDate)} ↗
    </a>
  );
}
