from __future__ import annotations
import json
from dataclasses import dataclass, field

from backend.agents.llm_client import LLMClient
from backend.agents.prompts import DECOMPOSER_SYSTEM_PROMPT


@dataclass
class ThesisCriteria:
    sector: str | None
    theme: str
    direction_hint: str | None  # "long" | "short" | None
    key_entities: list[str] = field(default_factory=list)


def decompose_thesis(thesis_text: str, llm: LLMClient) -> ThesisCriteria:
    """
    Turns a free-text thesis into structured criteria via a single LLM call.
    Forces JSON output so we never have to regex free text.
    """
    messages = [
        {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Thesis: " + thesis_text + "\n\n"
                "Respond with ONLY a JSON object with keys: "
                "sector (string or null), theme (string), "
                "direction_hint ('long', 'short', or null), "
                "key_entities (list of ticker/company strings mentioned explicitly)."
            ),
        },
    ]

    response = llm.complete(messages)
    data = _parse_json_response(response.text)

    return ThesisCriteria(
        sector=data.get("sector"),
        theme=data.get("theme", thesis_text),
        direction_hint=data.get("direction_hint"),
        key_entities=data.get("key_entities", []),
    )


def _parse_json_response(text: str) -> dict:
    """
    Strips markdown code fences if the model wraps the JSON in ```json ... ```
    and parses. Raises a clear error rather than silently returning {}.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Decomposer did not return valid JSON. Raw output: {text!r}"
        ) from e


if __name__ == "__main__":
    # quick manual smoke test — run with: python -m backend.research.thesis.decomposer
    import os
    llm = LLMClient(api_key=os.environ["LLM_API_KEY"])
    result = decompose_thesis("NVDA data center revenue growth is underpriced", llm)
    print(result)