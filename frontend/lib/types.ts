/**
 * TypeScript mirror of the Python data contract. Step 1.11.
 *
 * Field names are byte-identical to data/schemas/*.py and backend/api/schemas.py.
 * If you change a name here, change it there in the same commit.
 *
 * Dates arrive as ISO strings ("2024-11-20"), not Date objects.
 */

// ---------------------------------------------------------------- security

export interface Security {
  ticker: string;
  name: string;
  sector: string | null;
  industry: string | null;
  exchange: string | null;
  market_cap: number | null;
  adv_20d: number | null;
  is_active: boolean;
  first_date: string | null;
  last_date: string | null;
}

// ---------------------------------------------------------------- catalyst

export type SourceType =
  | "sec_filing"
  | "earnings_release"
  | "transcript"
  | "presentation"
  | "news"
  | "market_data";

export type Direction = "bullish" | "bearish" | "neutral";

export interface Catalyst {
  catalyst_id: string;
  ticker: string;
  headline: string;
  /** Verbatim source text. Render inside quotation marks. */
  quote: string;
  source_type: SourceType;
  /** Always an http(s) link. Make it clickable - this is the differentiator. */
  source_url: string;
  source_date: string;
  event_date: string | null;
  direction: Direction;
  confidence: number;
  extracted_by: string | null;
}

// ------------------------------------------------------------------ signal

export type FactorCategory =
  | "fundamental"
  | "earnings_revision"
  | "event"
  | "nlp"
  | "momentum"
  | "risk"
  | "valuation";

export interface SignalContribution {
  factor: string;
  category: FactorCategory;
  zscore: number;
  weight: number;
  /** zscore * weight. These sum to composite_alpha. */
  contribution: number;
}

export interface AlphaBreakdown {
  ticker: string;
  as_of: string;
  contributions: SignalContribution[];
  composite_alpha: number;
  alpha_score: number;
}

// ---------------------------------------------------------------- backtest

export interface CurvePoint {
  date: string;
  value: number;
}

export interface BacktestWindow {
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
}

export interface BacktestResult {
  strategy_name: string;
  universe_size: number;
  window: BacktestWindow;
  equity_curve: CurvePoint[];
  drawdown_curve: CurvePoint[];
  benchmark_curve: CurvePoint[];
  total_return: number;
  annualized_return: number;
  benchmark_annualized_return: number | null;
  excess_return: number | null;
  sharpe: number;
  sortino: number | null;
  volatility: number | null;
  /** Negative fraction, e.g. -0.084 */
  max_drawdown: number;
  turnover: number | null;
  win_rate: number | null;
  n_trades: number | null;
  commission_bps: number;
  slippage_model: string;
  max_adv_participation: number;
  is_walk_forward: boolean;
  notes: string | null;
}

// -------------------------------------------------------------------- risk

export type RiskBand = "low" | "medium" | "high";

export interface RiskMetrics {
  beta: number | null;
  volatility: number | null;
  max_drawdown: number | null;
  var_95: number | null;
  cvar_95: number | null;
  sector: string | null;
  sector_exposure: Record<string, number>;
  concentration: number | null;
  avg_correlation: number | null;
  days_to_liquidate: number | null;
  risk_band: RiskBand;
}

// -------------------------------------------------------------- trade idea

export type Side = "LONG" | "SHORT" | "NO_TRADE";
export type Verdict = "survived" | "rejected" | "inconclusive";

export interface TradeIdea {
  idea_id: string;
  ticker: string;
  company_name: string;
  side: Side;
  as_of: string;
  alpha_score: number;
  confidence: number;
  expected_alpha: number | null;
  position_size_pct: number | null;
  catalysts: Catalyst[];
  alpha: AlphaBreakdown | null;
  backtest: BacktestResult | null;
  risk: RiskMetrics | null;
  validator_verdict: Verdict;
  bull_case: string;
  bear_case: string;
  pm_rationale: string;
}

// --------------------------------------------------------------- api layer

export interface ThesisRequest {
  thesis: string;
  as_of?: string | null;
  max_candidates?: number;
  universe_size?: number | null;
}

export interface ScreeningCriterion {
  label: string;
  metric: string;
  operator: string;
  value: string;
  rationale: string | null;
}

export interface FunnelStage {
  label: string;
  count: number;
  description: string | null;
}

export interface CandidateSummary {
  ticker: string;
  company_name: string;
  sector: string | null;
  alpha_score: number;
  headline_catalyst: string | null;
}

export interface ScanResponse {
  thesis: string;
  as_of: string;
  criteria: ScreeningCriterion[];
  funnel: FunnelStage[];
  candidates: CandidateSummary[];
}

export interface ResearchResponse {
  job_id: string;
  thesis: string;
  as_of: string;
  top_idea: TradeIdea | null;
  runners_up: TradeIdea[];
  scan: ScanResponse | null;
  llm_cost_usd: number | null;
}

export type JobStatus = "queued" | "running" | "done" | "failed";

export interface JobHandle {
  job_id: string;
  status: JobStatus;
  created_at: string;
  stream_url: string;
  from_cache: boolean;
}

// ---------------------------------------------------------------- timeline

export type StepStatus = "pending" | "running" | "done" | "failed";

export interface ResearchEvent {
  step_id: string;
  label: string;
  status: StepStatus;
  detail: string | null;
  progress: number | null;
  timestamp: string;
}

/** Mirrors PIPELINE_STEPS in backend/services/events.py - keep in sync. */
export const PIPELINE_STEPS = [
  "parse_thesis",
  "define_criteria",
  "scan_universe",
  "identify_candidates",
  "retrieve_filings",
  "analyze_transcripts",
  "extract_catalysts",
  "generate_bull",
  "generate_bear",
  "run_event_study",
  "backtest_signal",
  "calculate_risk",
  "final_recommendation",
] as const;

export const STEP_LABELS: Record<string, string> = {
  parse_thesis: "Parsed investment thesis",
  define_criteria: "Defined screening criteria",
  scan_universe: "Scanned universe",
  identify_candidates: "Identified candidates",
  retrieve_filings: "Retrieved 10-K / 10-Q filings",
  analyze_transcripts: "Analyzed earnings transcripts",
  extract_catalysts: "Extracted catalysts",
  generate_bull: "Generated bull thesis",
  generate_bear: "Generated bear thesis",
  run_event_study: "Ran historical event study",
  backtest_signal: "Backtested signal",
  calculate_risk: "Calculated portfolio risk",
  final_recommendation: "Generated final recommendation",
};
