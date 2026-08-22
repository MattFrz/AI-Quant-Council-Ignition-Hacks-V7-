from __future__ import annotations

from backend.agents.base import Agent
from backend.agents.state import ResearchState
from backend.agents.prompts import PORTFOLIO_MANAGER_PROMPT
from data.schemas.risk import RiskBand, RiskMetrics
from data.schemas.trade_idea import Side, TradeIdea, Verdict


class PortfolioManager(Agent):
    """
    Final synthesis. Same boundary as QuantValidator: the LLM only writes
    pm_rationale (narrative). Every numeric field on TradeIdea comes from
    state, populated by earlier agents/lanes — never from this LLM call.
    """

    def run(self, state: ResearchState) -> TradeIdea:
        state.emit("synthesis", "Synthesizing final trade idea", "in_progress")

        if state.backtest_result is None or state.risk_metrics is None:
            raise ValueError(
                "PortfolioManager requires backtest_result and risk_metrics "
                "on state — run QuantValidator first."
            )

        messages = [
            {"role": "system", "content": PORTFOLIO_MANAGER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Thesis: {state.thesis}\n\n"
                    f"Bull case:\n{state.bull_case}\n\n"
                    f"Bear case:\n{state.bear_case}\n\n"
                    f"Quant results: Sharpe={getattr(state.backtest_result, 'sharpe', 'n/a')}, "
                    f"max_drawdown={getattr(state.backtest_result, 'max_drawdown', 'n/a')}\n\n"
                    "Write a final rationale (3-5 sentences) synthesizing the "
                    "evidence, the debate, and the quant results into a single "
                    "recommendation. Do not invent any numbers — reference only "
                    "the figures given above."
                ),
            },
        ]
        response = self.llm.complete(messages)

        ticker = self._infer_primary_ticker(state)
        direction = self._infer_direction(state)

        # Field names below must match data/schemas/trade_idea.py exactly - it
        # is the frozen Phase 1 contract and the frontend reads it verbatim.
        # `side` not `direction`; alpha_score is 0-10 and required.
        raw_alpha = self._top_alpha_score(state, ticker)
        risk = state.risk_metrics
        if isinstance(risk, dict):
            risk = self._risk_from_dict(risk)

        # Sector comes from the cached company profiles, not from the LLM.
        if risk is not None and not risk.sector:
            risk.sector = self._sector_for(ticker)

        trade_idea = TradeIdea(
            idea_id=f"{ticker}-{state.as_of.isoformat()}",
            ticker=ticker,
            company_name=self._company_name(ticker),
            side=Side.LONG if direction == "long" else Side.SHORT,
            as_of=state.as_of,
            alpha_score=self._clamp_alpha(raw_alpha),
            confidence=self._confidence_from_agreement(state),
            expected_alpha=self._expected_alpha(state, ticker),
            catalysts=state.catalysts,
            backtest=state.backtest_result,
            risk=risk,
            validator_verdict=self._verdict(state),
            bull_case=state.bull_case or "",
            bear_case=state.bear_case or "",
            pm_rationale=response.text,
        )

        state.emit("synthesis", "Trade idea complete", "done")
        return trade_idea

    # --- helpers pulling structured values off state; no LLM involvement below ---

    @staticmethod
    def _clamp_alpha(value) -> float:
        """alpha_score is a required 0-10 display score.

        A missing score becomes 0.0 rather than a flattering guess - an idea
        the alpha model never scored should not look strong.
        """
        if value is None:
            return 0.0
        try:
            return float(min(max(float(value), 0.0), 10.0))
        except (TypeError, ValueError):
            return 0.0

    def _expected_alpha(self, state: ResearchState, ticker: str):
        """Annualized excess return over the benchmark, from the backtest.

        Explicitly a BACKTESTED figure, not a forecast. We have no
        forward-return model, and inventing one would be exactly the kind of
        made-up number the quant validator exists to prevent. If the lane
        upstream supplied a real expected_alpha it wins; otherwise this is the
        historical excess return the strategy actually produced, which is
        defensible as long as it is described as such.
        """
        supplied = self._score_entry(state, ticker).get("expected_alpha")
        if supplied is not None:
            return supplied

        bt = state.backtest_result
        if bt is None:
            return None

        excess = getattr(bt, "excess_return", None)
        if excess is not None:
            return float(excess)

        # No benchmark on the run - fall back to the absolute annualized
        # return rather than reporting nothing.
        annualized = getattr(bt, "annualized_return", None)
        return float(annualized) if annualized is not None else None

    @staticmethod
    def _sector_for(ticker: str):
        """GICS-style sector from the cached profiles."""
        try:
            from data.pipelines.prices import load_profiles

            profiles = load_profiles()
            match = profiles.loc[profiles["ticker"] == ticker, "sector"]
            if len(match) and match.iloc[0]:
                return str(match.iloc[0])
        except Exception:  # noqa: BLE001 - a missing cache must not block the idea
            pass
        return None

    @staticmethod
    def _company_name(ticker: str) -> str:
        """Real name from the cached profiles; ticker if we have no better."""
        try:
            from data.pipelines.prices import load_profiles

            profiles = load_profiles()
            match = profiles.loc[profiles["ticker"] == ticker, "name"]
            if len(match):
                return str(match.iloc[0])
        except Exception:  # noqa: BLE001 - a missing cache must not block the idea
            pass
        return ticker

    @staticmethod
    def _risk_from_dict(payload: dict) -> RiskMetrics:
        """QuantValidator stores risk as a plain dict; the schema wants the
        model. Only known fields are carried across - nothing is invented."""
        band = payload.get("risk_band")
        return RiskMetrics(
            beta=payload.get("beta"),
            volatility=payload.get("volatility"),
            max_drawdown=payload.get("max_drawdown"),
            var_95=payload.get("var_95"),
            cvar_95=payload.get("cvar_95"),
            concentration=payload.get("concentration"),
            avg_correlation=payload.get("avg_correlation"),
            days_to_liquidate=payload.get("days_to_liquidate"),
            risk_band=RiskBand(band) if band in {b.value for b in RiskBand} else RiskBand.MEDIUM,
        )

    @staticmethod
    def _verdict(state: ResearchState) -> Verdict:
        """Survived only if a real backtest produced a positive Sharpe.

        Anything else is inconclusive. The validator's job is to be able to say
        no, so this must never default to SURVIVED.
        """
        sharpe = getattr(state.backtest_result, "sharpe", None)
        if sharpe is None:
            return Verdict.INCONCLUSIVE
        return Verdict.SURVIVED if sharpe > 0 else Verdict.REJECTED

    def _infer_primary_ticker(self, state: ResearchState) -> str:
        """The idea must be about a RANKED candidate that we found evidence for.

        Taking catalysts[0].ticker alone picks whatever the retriever surfaced
        first, which can be a company the alpha model never shortlisted - so
        the system would claim to have ranked 483 names and then recommend one
        that was not among the top 7. It also breaks the score lookup, since
        factor_scores is keyed by candidate.

        Preference order:
          1. highest-scoring candidate that has catalysts
          2. any candidate with catalysts
          3. whatever the evidence is mostly about (research went elsewhere,
             so report that honestly rather than forcing a candidate we have
             no evidence for)
        """
        if not state.catalysts:
            raise ValueError("Cannot determine primary ticker - no catalysts on state")

        with_evidence = {c.ticker for c in state.catalysts}
        scores = state.factor_scores or {}

        ranked = [t for t in scores if t in with_evidence]
        if ranked:
            def score_of(t):
                entry = scores.get(t)
                if isinstance(entry, dict):
                    return entry.get("alpha_score") or 0.0
                try:
                    return float(entry)
                except (TypeError, ValueError):
                    return 0.0

            return max(ranked, key=score_of)

        counts = {}
        for c in state.catalysts:
            counts[c.ticker] = counts.get(c.ticker, 0) + 1
        return max(counts, key=counts.get)

    def _infer_direction(self, state: ResearchState) -> str:
        # naive default; replace with real logic once factor_scores/backtest
        # give a clear directional signal (e.g. sign of alpha_score)
        return "long"

    @staticmethod
    def _score_entry(state: ResearchState, ticker: str):
        """factor_scores arrives in one of two shapes and both are legitimate.

        quant/api.run_backtest needs a flat {ticker: float} to build a signal
        frame, while this agent wants the richer {ticker: {...}} form. Rather
        than force one lane to change, normalise here.
        """
        entry = (state.factor_scores or {}).get(ticker)
        if entry is None:
            return {}
        if isinstance(entry, dict):
            return entry
        # Flat numeric score: it is the composite alpha and nothing else.
        return {"alpha_score": float(entry)}

    def _top_alpha_score(self, state: ResearchState, ticker: str):
        return self._score_entry(state, ticker).get("alpha_score")

    def _top_expected_alpha(self, state: ResearchState, ticker: str):
        return self._score_entry(state, ticker).get("expected_alpha")

    def _top_signal_contributions(self, state: ResearchState, ticker: str):
        return self._score_entry(state, ticker).get("signal_contributions", [])

    def _confidence_from_agreement(self, state: ResearchState) -> float:
        # placeholder heuristic — tune once you have real bull/bear/quant output
        # to calibrate against
        return 0.6