import os
from typing import Any, List
from pydantic import Field

from mistralai import Mistral

import weave

from .mistral_helper import MistralAIStream
from .console import Console
from .tool_calling import function_tool, perform_tool_calls
from .state import AgentState

client = Mistral(os.getenv("MISTRAL_API_KEY"))


class Agent(weave.Object):
    model_name: str
    temperature: float
    tools: List[Any] = Field(default_factory=list)
    max_tokens: int = 1000

    def model_post_init(self, __context: Any) -> None:
        if self.tools:
            self.tools = [function_tool(tool) for tool in self.tools]

    @weave.op
    def step(self, state: AgentState) -> AgentState:
        """Run a step of the agent."""
        Console.step_start("agent", "green")
        roles = [message["role"] for message in state.messages]
        print(f"Agent state roles: {roles}")

        messages = state.messages

        Console.chat_response_start()
        stream = client.chat.stream(
            model=self.model_name,
            messages=messages,
            tools=[t.tool_schema for t in self.tools],
            max_tokens=self.max_tokens,
        )
        wrapped_stream = MistralAIStream(stream)
        for chunk in wrapped_stream:
            if chunk.data.choices[0].delta.content:
                Console.chat_message_content_delta(chunk.data.choices[0].delta.content)

        response = wrapped_stream.final_response()
        response_message = response.choices[0].message
        if response_message.content:
            Console.chat_response_complete()

        new_messages = []
        new_messages.append(response_message.model_dump(exclude_none=True))
        if response_message.tool_calls:
            new_messages.extend(perform_tool_calls(self.tools, response_message.tool_calls))

        return AgentState(messages=state.messages + new_messages)

    @weave.op
    def run(self, state: AgentState):
        """Run the agent until user intervention is needed."""
        while state.messages[-1]["role"] != "assistant":
            state = self.step(state)
        return state
