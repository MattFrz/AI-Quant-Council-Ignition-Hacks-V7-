from __future__ import annotations
from openai import OpenAI
from dataclasses import dataclass, field
import json

@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class LLMClient:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.total_cost_usd = 0.0

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        # TODO check current pricing for your model before relying on this 
        cost = (input_tokens / 1_000_000) * 0.15 + (output_tokens / 1_000_000) * 0.60
        self.total_cost_usd += cost
        return cost

    def complete(self, messages: list[dict]) -> LLMResponse:
        response = self.client.responses.create(model=self.model_name, input=messages)
        cost = self._cost(response.usage.input_tokens, response.usage.output_tokens)
        return LLMResponse(
            text=response.output_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
        )

    def complete_with_tools(
        self, messages: list[dict], tools: list[dict], tool_impls: dict
    ) -> LLMResponse:
        """
        tool_impls: maps tool name -> actual python callable, e.g.
            {"run_backtest": run_backtest_fn}
        """
        input_items = list(messages)
        total_cost = 0.0

        while True:
            if self.total_cost_usd > 15: #$15 per agent
                return LLMResponse(text="LLM usage exceeded")
            response = self.client.responses.create(
                model=self.model_name, input=input_items, tools=tools
            )
            total_cost += self._cost(response.usage.input_tokens, response.usage.output_tokens)

            # find function calls the model wants to make
            calls = [item for item in response.output if item.type == "function_call"]

            if not calls:
                # model is done calling tools, this is the final answer
                return LLMResponse(
                    text=response.output_text,
                    tool_calls=None,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cost_usd=total_cost,
                )

            # append the model's function-call items to the running input
            input_items += response.output

            # execute each requested tool call locally, append results
            for call in calls:
                fn = tool_impls[call.name]
                args = json.loads(call.arguments)
                result = fn(**args)
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                })
