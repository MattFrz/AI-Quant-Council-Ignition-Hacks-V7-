"""An evidence-backed event. Step 1.5 of the data contract.

The section 3 audit trail is only ever as good as this schema. A Catalyst without
a working source_url is an unsupported claim, so the URL is required and
validated here rather than hoped for downstream.
"""
from __future__ import annotations

from datetime import date as Date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    SEC_FILING = "sec_filing"
    EARNINGS_RELEASE = "earnings_release"
    TRANSCRIPT = "transcript"
    PRESENTATION = "presentation"
    NEWS = "news"
    MARKET_DATA = "market_data"


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Catalyst(BaseModel):
    catalyst_id: str
    ticker: str

    headline: str = Field(..., description="One line, plain English, no hedging")
    quote: str = Field(
        ...,
        min_length=1,
        description="VERBATIM text from the source. Never paraphrase - the "
                    "quote is what makes the trail auditable.",
    )

    source_type: SourceType
    source_url: str = Field(..., min_length=1)
    source_date: Date = Field(..., description="When the source was published")
    event_date: Optional[Date] = Field(None, description="When the event occurred")

    direction: Direction = Direction.NEUTRAL
    confidence: float = Field(..., ge=0.0, le=1.0)
    extracted_by: Optional[str] = Field(None, description="Agent/model that produced it")

    @field_validator("source_url")
    @classmethod
    def _must_be_a_real_link(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                "source_url must be a clickable http(s) link. A catalyst a judge "
                "cannot click back to its filing does not ship."
            )
        return v

    def is_known_at(self, as_of: Date) -> bool:
        return self.source_date <= as_of
