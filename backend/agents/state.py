from dataclasses import dataclass, field
from datetime import date
from data.schemas.backtest_result import BacktestResult
from data.schemas.security import Security
from data.schemas.catalyst import Catalyst
from backend.services.events import ResearchEvent
from data.schemas.filing import FilingChunk


@dataclass
class ResearchState:
    thesis: str
    as_of: date
    criteria: dict | None = None
    universe: list[Security] | None = None
    catalysts: list[Catalyst] = field(default_factory=list)
    retrieved_chunks: list[FilingChunk] = field(default_factory=list)
    factor_scores: dict | None = None
    backtest_result: BacktestResult | None = None
    risk_metrics: dict | None = None
    bull_case: str | None = None
    bear_case: str | None = None
    events: list[ResearchEvent] = field(default_factory=list)
