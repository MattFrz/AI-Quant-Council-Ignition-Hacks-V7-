"""End-to-end pipeline, cache and job runner. Step 3.9.

Run this before every commit from here on.

Tests that need the seeded price cache are skipped when it is absent, so the
suite still passes on a fresh clone. Everything else runs anywhere.
"""
from __future__ import annotations

import time
from datetime import date

import pytest

from backend.api.schemas import JobStatus
from backend.core import cache
from backend.services import job_runner
from backend.services.events import PIPELINE_STEPS, ResearchEvent, StepStatus
from backend.services.pipeline import Pipeline, PipelineResult, run_pipeline
from data.schemas.backtest_result import BacktestResult, BacktestWindow
from data.schemas.trade_idea import Side, TradeIdea, Verdict

DEMO_THESIS = (
    "Find companies benefiting from accelerating AI data-center spending "
    "that the market may be underpricing."
)


def _has_market_data() -> bool:
    try:
        from data.pipelines.prices import load_prices, load_profiles

        load_prices()
        load_profiles()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_data = pytest.mark.skipif(
    not _has_market_data(),
    reason="no seeded market data - run scripts/seed_data.py",
)


@pytest.fixture
def fake_result() -> PipelineResult:
    """A minimal but valid result, for cache tests that must not run the
    pipeline."""
    window = BacktestWindow(
        train_start=date(2015, 1, 1), train_end=date(2022, 12, 31),
        test_start=date(2023, 1, 1), test_end=date(2024, 12, 31),
    )
    idea = TradeIdea(
        idea_id="test-001", ticker="TEST", company_name="Test Co",
        side=Side.LONG, as_of=date(2025, 1, 15),
        alpha_score=7.5, confidence=0.7,
        backtest=BacktestResult(
            strategy_name="t", universe_size=10, window=window,
            total_return=0.2, annualized_return=0.1, sharpe=1.1, max_drawdown=-0.08,
        ),
        validator_verdict=Verdict.SURVIVED,
    )
    return PipelineResult(
        thesis="cache round trip test", as_of=date(2025, 1, 15), top_idea=idea,
        funnel_stages=[{"label": "Scanned", "count": 500, "description": "all"}],
        events=[ResearchEvent(step_id="parse_thesis", label="Parsed", status=StepStatus.DONE)],
        elapsed_s=12.3, degraded=["something: reason"],
    )


# ------------------------------------------------------------------ contract

def test_pipeline_result_funnel_prefers_stored_stages(fake_result):
    assert fake_result.funnel()[0]["count"] == 500


def test_emit_uses_only_valid_step_ids():
    """Every step_id the pipeline emits must be a known pipeline step, or the
    UI has no row to light up."""
    seen = []
    p = Pipeline(emit=seen.append)
    p.emit("parse_thesis", "running")
    p.emit("scan_universe", "done", "500 scanned")

    assert [e.step_id for e in seen] == ["parse_thesis", "scan_universe"]
    assert all(e.step_id in PIPELINE_STEPS for e in seen)
    assert seen[0].label == "Parsed investment thesis"


def test_degrade_records_reason_without_raising():
    p = Pipeline()
    p._degrade("some_stage", "dependency missing")
    assert p.degraded == ["some_stage: dependency missing"]


# --------------------------------------------------------------------- cache

def test_cache_key_is_stable_and_normalises_whitespace():
    a = cache.cache_key("Find  mispriced   names")
    b = cache.cache_key("find mispriced names")
    assert a == b


def test_cache_key_changes_with_thesis():
    assert cache.cache_key("thesis one") != cache.cache_key("thesis two")


def test_cache_key_changes_with_as_of():
    assert cache.cache_key("t", date(2024, 1, 1)) != cache.cache_key("t", date(2024, 6, 1))


def test_cache_round_trip_preserves_the_trade_idea(fake_result, tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_cache_dir", lambda: tmp_path)

    cache.put(fake_result)
    loaded = cache.get(fake_result.thesis, fake_result.as_of)

    assert loaded is not None
    assert loaded.top_idea.ticker == "TEST"
    assert loaded.top_idea.backtest.sharpe == 1.1
    assert loaded.top_idea.validator_verdict == Verdict.SURVIVED
    assert loaded.funnel()[0]["count"] == 500
    assert loaded.degraded == ["something: reason"]
    assert loaded.events[0].status == StepStatus.DONE


def test_cache_miss_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_cache_dir", lambda: tmp_path)
    assert cache.get("never computed") is None


def test_cache_refuses_to_store_a_failed_run(tmp_path, monkeypatch):
    """Caching a failure would make every later run replay it instantly."""
    monkeypatch.setattr(cache, "_cache_dir", lambda: tmp_path)
    empty = PipelineResult(thesis="failed run", as_of=date(2025, 1, 1), top_idea=None)
    with pytest.raises(ValueError, match="no TradeIdea"):
        cache.put(empty)


def test_corrupt_cache_entry_is_a_miss_not_a_crash(fake_result, tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_cache_dir", lambda: tmp_path)
    path = cache.put(fake_result)
    path.write_text("{ not json", encoding="utf-8")
    assert cache.get(fake_result.thesis, fake_result.as_of) is None


# ---------------------------------------------------------------- job runner

def test_job_starts_and_returns_immediately(monkeypatch):
    """The whole point of 3.6: starting a job must not block."""
    def slow(**kwargs):
        time.sleep(0.5)
        return PipelineResult(thesis=kwargs["thesis"], as_of=date(2025, 1, 1))

    monkeypatch.setattr(job_runner, "run_pipeline", slow)

    t0 = time.monotonic()
    job = job_runner.start_job("slow thesis", use_cache=False)
    assert time.monotonic() - t0 < 0.2
    assert job.status in (JobStatus.QUEUED, JobStatus.RUNNING)


def test_job_reaches_done_and_keeps_its_result(monkeypatch):
    monkeypatch.setattr(
        job_runner, "run_pipeline",
        lambda **kw: PipelineResult(thesis=kw["thesis"], as_of=date(2025, 1, 1)),
    )
    job = job_runner.start_job("quick thesis", use_cache=False)

    for _ in range(50):
        if job.is_finished:
            break
        time.sleep(0.05)

    assert job.status == JobStatus.DONE
    assert job.result is not None
    assert job_runner.get_job(job.job_id) is job


def test_job_failure_is_captured_not_raised(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(job_runner, "run_pipeline", boom)
    job = job_runner.start_job("bad thesis", use_cache=False)

    for _ in range(50):
        if job.is_finished:
            break
        time.sleep(0.05)

    assert job.status == JobStatus.FAILED
    assert "pipeline exploded" in job.error
    assert job.events[-1].status == StepStatus.FAILED


def test_stream_events_yields_then_terminates(monkeypatch):
    """A stream that never ends leaves the UI spinning forever."""
    def emitting(**kwargs):
        emit = kwargs["emit"]
        for step in ("parse_thesis", "scan_universe", "final_recommendation"):
            emit(ResearchEvent(step_id=step, label=step, status=StepStatus.DONE))
        return PipelineResult(thesis=kwargs["thesis"], as_of=date(2025, 1, 1))

    monkeypatch.setattr(job_runner, "run_pipeline", emitting)
    job = job_runner.start_job("streaming thesis", use_cache=False)

    received = list(job_runner.stream_events(job.job_id))
    assert [e.step_id for e in received] == [
        "parse_thesis", "scan_universe", "final_recommendation"
    ]


def test_late_subscriber_gets_the_backlog(monkeypatch):
    monkeypatch.setattr(
        job_runner, "run_pipeline",
        lambda **kw: (kw["emit"](ResearchEvent(step_id="parse_thesis", label="p",
                                               status=StepStatus.DONE))
                      or PipelineResult(thesis=kw["thesis"], as_of=date(2025, 1, 1))),
    )
    job = job_runner.start_job("backlog thesis", use_cache=False)

    for _ in range(50):
        if job.is_finished:
            break
        time.sleep(0.05)

    assert len(list(job_runner.stream_events(job.job_id))) >= 1


def test_cache_hit_completes_the_job_without_running(fake_result, tmp_path, monkeypatch):
    """Demo insurance: a warmed thesis returns instantly and never calls the
    pipeline."""
    monkeypatch.setattr(cache, "_cache_dir", lambda: tmp_path)
    cache.put(fake_result)

    def must_not_run(**kwargs):
        raise AssertionError("pipeline ran despite a cache hit")

    monkeypatch.setattr(job_runner, "run_pipeline", must_not_run)

    job = job_runner.start_job(fake_result.thesis, as_of=fake_result.as_of)
    assert job.status == JobStatus.DONE
    assert job.from_cache is True
    assert job.result.top_idea.ticker == "TEST"


def test_unknown_job_id_raises():
    with pytest.raises(KeyError):
        list(job_runner.stream_events("does-not-exist"))


def test_pending_timeline_covers_every_step():
    timeline = job_runner.pending_timeline()
    assert len(timeline) == len(PIPELINE_STEPS)
    assert all(e.status == StepStatus.PENDING for e in timeline)


# --------------------------------------------------------- end to end (real)

@needs_data
def test_full_pipeline_produces_a_result():
    result = run_pipeline(DEMO_THESIS, max_candidates=3, universe_size=60)

    assert result.top_idea is not None, f"no idea produced; degraded={result.degraded}"
    assert result.funnel(), "funnel is empty"
    assert result.elapsed_s > 0


@needs_data
def test_full_pipeline_emits_a_coherent_timeline():
    seen = []
    run_pipeline(DEMO_THESIS, max_candidates=3, universe_size=60, emit=seen.append)

    assert seen, "no events emitted"
    assert all(e.step_id in PIPELINE_STEPS for e in seen)
    assert seen[-1].step_id == "final_recommendation"


@needs_data
def test_degraded_run_never_claims_an_audit_trail():
    """The rule that matters: if retrieval failed there are no catalysts, and
    the idea must not present itself as researched."""
    result = run_pipeline(DEMO_THESIS, max_candidates=3, universe_size=60)
    idea = result.top_idea
    assert idea is not None

    if not idea.catalysts:
        assert idea.has_audit_trail() is False
        assert idea.validator_verdict == Verdict.INCONCLUSIVE
        assert "QUANT ONLY" in idea.pm_rationale


@needs_data
def test_backtest_numbers_are_real_when_present():
    result = run_pipeline(DEMO_THESIS, max_candidates=3, universe_size=60)
    bt = result.top_idea.backtest
    if bt is not None:
        assert bt.equity_curve, "backtest has no equity curve"
        assert bt.max_drawdown <= 0
        assert -10 < bt.sharpe < 10, f"implausible sharpe {bt.sharpe}"
