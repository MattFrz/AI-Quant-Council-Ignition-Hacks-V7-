/**
 * frontend/lib/api.ts - D1
 *
 * Every screen calls through this file, never `fetch()` directly. In fixture
 * mode (NEXT_PUBLIC_USE_FIXTURE=true, the Phase 2 default) every function
 * resolves from the local data/fixtures/sample_trade_idea.json fixture with a
 * small artificial delay, so loading states get exercised too. In live mode
 * it hits the real FastAPI backend at NEXT_PUBLIC_API_BASE.
 *
 * Phase 3 step 3.8: flip NEXT_PUBLIC_USE_FIXTURE to false in .env.local.
 * Nothing else in the frontend should need to change - that's the whole
 * point of routing every call through here.
 */
import type {
  ThesisRequest,
  ScanResponse,
  ScreeningCriterion,
  FunnelStage,
  CandidateSummary,
  ResearchResponse,
  TradeIdea,
  JobHandle,
  JobStatus,
} from "./types";

// data/fixtures/sample_trade_idea.json lives at the repo root (shared with the
// backend - see backend/main.py's /api/fixture route). This is the single
// source of truth for every screen until Phase 3.
import rawFixture from "../../data/fixtures/sample_trade_idea.json";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
// Anything other than the literal string "false" is treated as fixture mode,
// so an unset env var fails safe into fixture mode rather than silently
// trying to hit a backend that isn't there yet.
export const USE_FIXTURE = process.env.NEXT_PUBLIC_USE_FIXTURE !== "false";

// Strip the human-readable _README key - the backend's pydantic model ignores
// it too (see data/schemas/trade_idea.py), this just keeps our TS type honest.
const { _README, ...fixtureTradeIdeaRaw } = rawFixture as Record<string, unknown> & {
  _README?: string;
};
export const FIXTURE_TRADE_IDEA = fixtureTradeIdeaRaw as unknown as TradeIdea;

// Matt's A6 universe-funnel counts aren't wired up yet. These are the exact
// cascade numbers named in the plan (D7: 1,247 -> 183 -> 74 -> 21 -> 7 -> 3)
// so UniverseFunnel and CandidateTable have real numbers to build and animate
// against. Delete this block once /api/scan returns real data in Phase 3.
export const FIXTURE_SCAN: ScanResponse = {
  thesis:
    "Find companies benefiting from the AI data-center buildout that the market may be underpricing.",
  as_of: FIXTURE_TRADE_IDEA.as_of,
  criteria: [
    {
      label: "Market cap",
      metric: "market_cap",
      operator: "gte",
      value: "1000000000",
      rationale: "Liquid enough to size a real position",
    },
    {
      label: "Avg daily volume",
      metric: "adv_20d",
      operator: "gte",
      value: "5000000",
      rationale: "Keeps exit risk manageable",
    },
    {
      label: "AI data-center capex exposure",
      metric: "capex_guidance_delta",
      operator: "gt",
      value: "0",
      rationale: "Thesis-aligned",
    },
  ] as ScreeningCriterion[],
  funnel: [
    { label: "Universe", count: 1247, description: "Full liquid US equity universe" },
    { label: "Liquidity + market cap", count: 183, description: null },
    { label: "Fundamental screen", count: 74, description: null },
    { label: "Event / catalyst screen", count: 21, description: null },
    { label: "Retrieval + research", count: 7, description: null },
    { label: "Survived validation", count: 3, description: null },
  ] as FunnelStage[],
  candidates: [
    {
      ticker: FIXTURE_TRADE_IDEA.ticker,
      company_name: FIXTURE_TRADE_IDEA.company_name,
      sector: FIXTURE_TRADE_IDEA.risk?.sector ?? null,
      alpha_score: FIXTURE_TRADE_IDEA.alpha_score,
      headline_catalyst: FIXTURE_TRADE_IDEA.catalysts[0]?.headline ?? null,
    },
  ] as CandidateSummary[],
};

// Fake network latency in fixture mode so spinners/skeletons (D4) actually
// get exercised during dev instead of only ever flashing instantly.
const FIXTURE_LATENCY_MS = 400;

function delay<T>(value: T, ms: number = FIXTURE_LATENCY_MS): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    // FastAPI puts the human-readable reason in `detail`. Surfacing the raw
    // JSON envelope instead turns a deliberate, explanatory refusal - like the
    // public-demo guard's 409 - into something that reads as a crash.
    let detail = body;
    try {
      const parsed = JSON.parse(body);
      if (typeof parsed?.detail === "string") detail = parsed.detail;
    } catch {
      /* not JSON: fall through to the raw body */
    }
    throw new Error(detail || res.statusText || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

// -------------------------------------------------------------- thesis run

/**
 * Kicks off a research run. Fixture mode returns a fake JobHandle almost
 * immediately - stream.ts recognizes the sentinel stream_url "fixture://replay"
 * and replays the D6 pipeline timeline on a timer instead of opening a real
 * EventSource.
 */
export async function submitThesis(request: ThesisRequest): Promise<JobHandle> {
  if (USE_FIXTURE) {
    return delay<JobHandle>(
      {
        job_id: "fixture-job-001",
        status: "queued" as JobStatus,
        created_at: new Date().toISOString(),
        stream_url: "fixture://replay",
        from_cache: false,
      },
      150,
    );
  }
  // NOTE: confirm this path with whoever builds backend/api/routes/research.py
  // in Phase 3 step 3.2 - adjust here if it lands somewhere else. This is the
  // only place that should need to change.
  return apiFetch<JobHandle>("/api/research", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/** Full result for a finished job: top idea, runners-up, scan, LLM cost. */
export async function fetchResearchResult(jobId: string): Promise<ResearchResponse> {
  if (USE_FIXTURE) {
    return delay<ResearchResponse>({
      job_id: jobId,
      thesis: FIXTURE_SCAN.thesis,
      as_of: FIXTURE_TRADE_IDEA.as_of,
      top_idea: FIXTURE_TRADE_IDEA,
      runners_up: [],
      scan: FIXTURE_SCAN,
      llm_cost_usd: 0.13,
    });
  }
  return apiFetch<ResearchResponse>(`/api/research/${jobId}`);
}

/** Standalone scan call - handy if the scanner screen (D7/D8) is reachable
 * on its own before a full research run exists. */
export async function fetchScan(thesis: string): Promise<ScanResponse> {
  if (USE_FIXTURE) {
    return delay<ScanResponse>({ ...FIXTURE_SCAN, thesis });
  }
  return apiFetch<ScanResponse>("/api/scan", {
    method: "POST",
    body: JSON.stringify({ thesis }),
  });
}

/** GET /health passthrough - handy for a small "backend connected" indicator. */
export async function fetchHealth(): Promise<{ status: string; offline_mode: boolean }> {
  if (USE_FIXTURE) {
    return delay({ status: "ok", offline_mode: false }, 100);
  }
  return apiFetch("/health");
}

// ---------------------------------------------------------------- portfolio

export interface PortfolioPosition {
  ticker: string;
  company_name: string;
  side: string;
  position_size_pct: number;
  alpha_score: number;
  confidence: number;
  risk_band: string;
}

export interface ExcludedCandidate {
  ticker: string;
  reasons: string[];
}

export interface PortfolioResponse {
  positions: PortfolioPosition[];
  excluded: ExcludedCandidate[];
  total_invested_pct: number;
  cash_pct: number;
}

/**
 * Size a book from validated ideas.
 *
 * The interesting half of the response is `excluded`: a name that cleared
 * research and the backtest can still be refused here on risk grounds, and
 * that refusal is the product working rather than an empty state.
 */
export async function buildPortfolio(
  candidates: TradeIdea[],
  maxPositionPct?: number,
): Promise<PortfolioResponse> {
  if (USE_FIXTURE) {
    return delay<PortfolioResponse>({
      positions: [],
      excluded: [{ ticker: FIXTURE_TRADE_IDEA.ticker, reasons: ["fixture mode"] }],
      total_invested_pct: 0,
      cash_pct: 100,
    });
  }
  return apiFetch<PortfolioResponse>("/api/portfolio", {
    method: "POST",
    body: JSON.stringify({
      candidates,
      ...(maxPositionPct !== undefined ? { max_position_pct: maxPositionPct } : {}),
    }),
  });
}

// ---------------------------------------------------------------- execution

export interface BookLevel {
  price: number;
  shares: number;
}

export interface ExecutionOutcome {
  mode: string;
  requested: number;
  filled: number;
  avg_price: number;
  arrival_mid: number;
  slippage_bps: number;
  levels_consumed: number;
  complete: boolean;
  note: string;
}

export interface SimulationResponse {
  available: boolean;
  bids: BookLevel[];
  asks: BookLevel[];
  mid: number;
  spread_bps: number;
  capacity_5bps: number;
  outcomes: ExecutionOutcome[];
  engine: string;
}

export interface SimulationRequest {
  shares?: number;
  side?: "BUY" | "SELL";
  slices?: number;
  levels?: number;
  level_shares?: number;
  queue_volume?: number;
}

/**
 * Run the C++ execution simulator.
 *
 * Every number in the response is computed by the native extension walking a
 * real price-time-priority book, not by anything in this file. A deployment
 * without the extension built answers 503, which surfaces as an error rather
 * than as plausible-looking output.
 */
export async function simulateExecution(
  req: SimulationRequest,
): Promise<SimulationResponse> {
  return apiFetch<SimulationResponse>("/api/execution", {
    method: "POST",
    body: JSON.stringify(req),
  });
}
