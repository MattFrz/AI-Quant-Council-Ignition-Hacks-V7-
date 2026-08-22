from __future__ import annotations

from backend.agents.base import Agent
from backend.agents.state import ResearchState
from backend.agents.llm_client import LLMClient
from backend.rag.retrieval.retriever import Retriever
from backend.rag.retrieval.citations import to_citations


class ResearchAgent(Agent):
    """
    Executes the plan built by ResearchPlanner: runs each step's query
    through the retriever, respecting the as_of date on state, and
    accumulates retrieved chunks + citations onto ResearchState.
    """

    def __init__(self, llm_client: LLMClient, retriever: Retriever):
        super().__init__(llm_client)
        self.retriever = retriever

    def run(self, state: ResearchState) -> ResearchState:
        plan = getattr(state, "plan", None)
        if not plan:
            raise ValueError("ResearchAgent requires state.plan - run ResearchPlanner first")

        for step in plan:
            state.emit(step.id, step.label, "in_progress")

            results = self.retriever.retrieve(step.query, as_of=state.as_of, k=10)
            state.retrieved_chunks.extend(results)

            citations = to_citations(results, self.retriever.filing_lookup)
            state.citations = getattr(state, "citations", []) + citations

            state.emit(step.id, f"{step.label} - found {len(results)} sources", "done")

        return state