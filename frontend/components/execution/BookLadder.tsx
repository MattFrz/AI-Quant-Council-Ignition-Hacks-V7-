import type { BookLevel } from "../../lib/api";
import { int, num } from "../../lib/format";

/**
 * The limit order book, asks descending to bids, with the spread in the middle.
 *
 * Depth bars are scaled to the largest resting size on the screen so relative
 * liquidity reads at a glance. Asks are drawn above bids because that is how a
 * ladder is read on a trading screen, and getting it upside down would be the
 * first thing anyone who works in this domain notices.
 */
export function BookLadder({
  bids,
  asks,
  mid,
  spreadBps,
}: {
  bids: BookLevel[];
  asks: BookLevel[];
  mid: number;
  spreadBps: number;
}) {
  const peak = Math.max(...[...bids, ...asks].map((l) => l.shares), 1);

  const row = (level: BookLevel, side: "ask" | "bid") => (
    <li className="ladder-row" data-side={side} key={`${side}-${level.price}`}>
      <span className="ladder-depth" style={{ width: `${(level.shares / peak) * 100}%` }} />
      <span className="ladder-price mono">{num(level.price, 2)}</span>
      <span className="ladder-size mono">{int(level.shares)}</span>
    </li>
  );

  return (
    <div className="ladder">
      <ul className="ladder-side">{[...asks].reverse().map((l) => row(l, "ask"))}</ul>
      <div className="ladder-mid">
        <span className="mono">{num(mid, 3)}</span>
        <span className="text-muted">spread {num(spreadBps, 2)} bps</span>
      </div>
      <ul className="ladder-side">{bids.map((l) => row(l, "bid"))}</ul>
    </div>
  );
}
