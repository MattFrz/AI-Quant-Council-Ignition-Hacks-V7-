"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CurvePoint } from "../lib/types";

/**
 * Strategy equity against the benchmark.
 *
 * The benchmark is drawn as a plain line rather than a second filled area:
 * two fills read as competing quantities, when the thing the reader needs to
 * see is the GAP between them. Without the benchmark on the chart at all, an
 * 19% annualised return looks like skill rather than mostly market.
 */
export function EquityCurve({
  equity,
  benchmark,
}: {
  equity: CurvePoint[];
  benchmark: CurvePoint[];
}) {
  if (!equity?.length) {
    return <p className="empty">No equity curve for this run.</p>;
  }

  const byDate = new Map<string, { date: string; strategy: number; benchmark?: number }>();
  for (const p of equity) {
    byDate.set(String(p.date), { date: String(p.date), strategy: p.value });
  }
  for (const p of benchmark ?? []) {
    const row = byDate.get(String(p.date));
    if (row) row.benchmark = p.value;
  }
  const data = [...byDate.values()];

  return (
    <>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--positive)" stopOpacity={0.28} />
                <stop offset="100%" stopColor="var(--positive)" stopOpacity={0} />
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
              width={44}
              tickFormatter={(v: number) => `${v.toFixed(1)}x`}
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
              formatter={(v, name) => [`${Number(v ?? 0).toFixed(3)}x`, String(name)]}
            />
            <Area
              type="monotone"
              dataKey="strategy"
              name="Strategy"
              stroke="var(--positive)"
              strokeWidth={2}
              fill="url(#equityFill)"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="benchmark"
              name="SPY"
              stroke="var(--text-muted)"
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        <span>
          <i className="swatch" style={{ background: "var(--positive)" }} /> Strategy
        </span>
        <span>
          <i className="swatch" style={{ background: "var(--text-muted)" }} /> SPY benchmark
        </span>
      </div>
    </>
  );
}
