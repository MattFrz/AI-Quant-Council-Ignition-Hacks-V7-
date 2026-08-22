from abc import ABC, abstractmethod
from backend.agents.state import ResearchState


class Agent(ABC):
    def __init__(self, llm_client):
        # Subclasses reach for `self.llm` (planner, bull, bear, manager) while
        # this base originally exposed only `self.llm_client`. Both names point
        # at the same object so neither convention breaks - cheaper and safer
        # mid-integration than renaming across every agent.
        self.llm_client = llm_client
        self.llm = llm_client

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        raise NotImplementedError