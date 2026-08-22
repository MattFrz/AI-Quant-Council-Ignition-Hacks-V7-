from __future__ import annotations
from datetime import date as Date
from fastapi import APIRouter, HTTPException

from backend.api.schemas import ThesisRequest, ScanResponse, FunnelStage
from backend.api.routes.thesis import decompose, get_llm_client
from backend.agents.llm_client import LLMClient
from backend.config import settings
from quant.universe.builder import build_universe  # Matt, A6

router = APIRouter(prefix="/scan", tags=["scan"])


@router.post("", response_model=ScanResponse)
def scan(request: ThesisRequest):
    """
    Second stage of the §1 pipeline order (3.2): criteria -> universe screen.

    HONEST GAP: ScanResponse.candidates needs CandidateSummary objects with
    alpha_score - that's Nalin's factor composite (B9), not something Lane C
    or Lane A alone can produce. This route currently returns an EMPTY
    candidates list rather than fabricating scores. Once B9 exists, wire its
    output in here - don't ship a placeholder alpha_score in the meantime.
    """
    llm = LLMClient(api_key=settings.llm_api_key)
    criteria_list = decompose(request, llm)  # reuses the /thesis/decompose logic directly

    panel, profiles = _load_panel_and_profiles()
    if panel is None:
        raise HTTPException(
            status_code=503,
            detail="Price/fundamentals cache not available - run seed_data.py (A4) first.",
        )

    result = build_universe(
        panel=panel,
        profiles=profiles,
        min_market_cap=settings.min_market_cap,
        min_adv_usd=settings.min_adv_usd,
        min_days=500,
        max_size=request.universe_size or settings.universe_size,
    )

    funnel_rows = result.funnel()  # [{"label":..., "count":..., "description":...}, ...]

    return ScanResponse(
        thesis=request.thesis,
        as_of=request.as_of or Date.today(),
        criteria=criteria_list,
        funnel=[FunnelStage(**row) for row in funnel_rows],
        candidates=[],  # see HONEST GAP above - needs Lane B's alpha scoring
    )


def _load_panel_and_profiles():
    """
    STILL A GUESS - I don't know where Matt's pipeline.py (3.1) actually
    loads cached panel/profiles from. Confirm the real loader with him
    before trusting this. Fails to a clean 503 rather than crashing if the
    import is wrong.
    """
    try:
        from data.pipelines.prices import load_cached_panel, load_cached_profiles
        return load_cached_panel(), load_cached_profiles()
    except ImportError:
        return None, None