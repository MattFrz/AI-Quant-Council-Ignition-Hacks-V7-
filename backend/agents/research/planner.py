from __future__ import annotations
import json
from dataclasses import dataclass

from backend.agents.base import Agent
from backend.agents.state import ResearchState
from backend.agents.llm_client import LLMClient
from backend.agents.prompts import RESEARCH_PLANNER_PROMPT

# NOTE: this assumes ResearchState has an `emit(step_id, label, status)` helper
# and an `events` list, as sketched in the C3 guidance. If your actual C3
# implementation names things differently, adjust the calls below to match -
# the important part is that every step change gets logged as a ResearchEvent.


@dataclass
class PlanStep:
    id: str
    label: str
    query: str  # the retrieval query this step will run in ResearchAgent


class ResearchPlanner(Agent):
    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client)

    def run(self, state: ResearchState) -> ResearchState:
        state.emit("plan", "Planning research steps", "in_progress")

        plan = self._generate_plan(state.thesis)
        state.plan = plan  # stash on state for ResearchAgent to consume next

        state.emit("plan", f"Planned {len(plan)} research steps", "done")
        return state

    def _generate_plan(self, thesis: str) -> list[PlanStep]:
        messages = [
            {"role": "system", "content": RESEARCH_PLANNER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Thesis: {thesis}\n\n"
                    "Break this into 3-5 concrete retrieval queries that would "
                    "surface evidence relevant to evaluating it. Respond with "
                    "ONLY a JSON array of objects: "
                    '{"id": "step_1", "label": "short description", "query": "search text"}'
                ),
            },
        ]
        response = self.llm.complete(messages)
        raw_steps = _parse_json_array(response.text)

        # Fallback: if the model returns nothing usable, run one broad query
        # rather than failing the whole pipeline on a planning hiccup.
        if not raw_steps:
            return [PlanStep(id="step_1", label="General thesis search", query=thesis)]

        return [
            PlanStep(id=s.get("id", f"step_{i}"), label=s.get("label", ""), query=s.get("query", thesis))
            for i, s in enumerate(raw_steps)
        ]


def _parse_json_array(text: str) -> list:
    """Best-effort JSON array out of an LLM response.

    Model output is untrusted input. Three failure modes are common and all of
    them used to break the run:

      1. Fenced code blocks (```json ... ```)
      2. Valid JSON wrapped in prose ("Here you go:" ... "Hope that helps!")
      3. Right shape, wrong item types - a list of bare strings, or nulls

    (3) was the dangerous one: callers do item.get(...), so a list of strings
    raised AttributeError and killed the whole pipeline, surfacing as "the run
    randomly produced no catalysts". Only dict items are returned now.
    """
    if not text:
        return []

    cleaned = text.strip()

    # Strip a fenced block if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    def only_dicts(value):
        return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []

    try:
        return only_dicts(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # Recover an array embedded in surrounding prose.
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            return only_dicts(json.loads(cleaned[start:end + 1]))
        except json.JSONDecodeError:
            pass

    return []

