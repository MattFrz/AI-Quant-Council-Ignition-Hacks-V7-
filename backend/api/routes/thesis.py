from __future__ import annotations
from fastapi import APIRouter, Depends

from backend.api.schemas import ThesisRequest, ScreeningCriterion
from backend.agents.llm_client import LLMClient
from backend.research.thesis.decomposer import decompose_thesis
from backend.config import settings

router = APIRouter(prefix="/thesis", tags=["thesis"])


def get_llm_client() -> LLMClient:
    return LLMClient(api_key=settings.llm_api_key)


@router.post("/decompose", response_model=list[ScreeningCriterion])
def decompose(request: ThesisRequest, llm: LLMClient = Depends(get_llm_client)):
    """
    First stage of the §1 pipeline order (3.2). Returns ScreeningCriterion
    objects, not the raw ThesisCriteria dataclass from decomposer.py —
    the two shapes genuinely differ (label/metric/operator/value/rationale
    vs sector/theme/direction_hint/key_entities), so this is a real mapping,
    not a field rename.

    HONEST GAP: ThesisCriteria (C4) doesn't produce metric/operator/value
    triples — it produces a loose sector/theme/entities read of the thesis.
    Turning that into machine-executable criteria (e.g. "adv_20d", "gt",
    "5000000") likely needs either a second, more targeted LLM call, or a
    rewrite of decompose_thesis's prompt/output shape to match
    ScreeningCriterion directly. The mapping below is a placeholder that
    produces one soft criterion — not enough for scan.py to build on yet.
    """
    from backend.research.thesis.decomposer import ThesisCriteria  # local import for typing clarity

    criteria: ThesisCriteria = decompose_thesis(request.thesis, llm)

    result = []
    if criteria.sector:
        result.append(ScreeningCriterion(
            label=f"Sector: {criteria.sector}",
            metric="sector",
            operator="in",
            value=criteria.sector,
            rationale="Inferred from thesis text",
        ))
    for entity in criteria.key_entities:
        result.append(ScreeningCriterion(
            label=f"Mentions {entity}",
            metric="ticker",
            operator="in",
            value=entity,
            rationale="Explicitly named in thesis",
        ))
    return result