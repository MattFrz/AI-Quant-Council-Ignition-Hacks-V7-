"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CurvePoint } from "../lib/types";
import { pct } from "../lib/format";

/**
 * Underwater curve: how far below its own high-water mark the strategy sat on
 * any given day.
 *
 * The equity curve alone flatters a strategy, because a line that ends higher
 * than it started hides how it got there. This is the chart that shows what
 * holding it would actually have felt like, so the worst point is marked
 * rather than left for the reader to find.
 */
export function DrawdownChart({ drawdown }: { drawdown: CurvePoint[] }) {
  if (!drawdown?.length) {
    return <p className="empty">No drawdown curve for this run.</p>;
  }

  const data = drawdown.map((p) => ({ date: String(p.date), value: p.value }));
  const worst = data.reduce((a, b) => (b.value < a.value ? b : a), data[0]);

  return (
    <>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--negative)" stopOpacity={0} />
                <stop offset="100%" stopColor="var(--negative)" stopOpacity={0.3} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: "var(--text-muted)", fontSize: 11 }}
              stroke="var(--border)"
              minTickGap={48}
              tickFormatter={(v: string) => String(v).slice(0, 7)}
            />
            <YAxis
              tick={{ fill: "var(--text-muted)", fontSize: 11 }}
              stroke="var(--border)"
              width={48}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border-strong)",
                borderRadius: "var(--radius-md)",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
              }}
              labelStyle={{ color: "var(--text-muted)" }}
              formatter={(v) => [pct(Number(v ?? 0)), "Drawdown"]}
            />
            <ReferenceLine
              y={worst.value}
              stroke="var(--negative)"
              strokeDasharray="3 3"
              strokeOpacity={0.6}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="var(--negative)"
              strokeWidth={1.5}
              fill="url(#ddFill)"
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        <span>
          Worst: <strong className="mono text-negative">{pct(worst.value)}</strong> on{" "}
          <span className="mono">{worst.date}</span>
        </span>
      </div>
    </>
  );
}
