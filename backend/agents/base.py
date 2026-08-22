from abc import ABC, abstractmethod
from backend.agents.state import ResearchState


class Agent(ABC):
    def __init__(self, llm_client):
        self.llm_client = llm_client

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        raise NotImplementedError