from typing import Any, List, Optional
from pydantic import Field
from openai import OpenAI
from openai._types import NotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
)

import weave
from weave.flow.chat_util import OpenAIStream

from .console import Console
from .tool_calling import chat_call_tool_params, perform_tool_calls
from .state import AgentState

class Agent(weave.Object):
    model_name: str
    temperature: float
    system_message: str
    tools: List[Any] = Field(default_factory=list)

    @weave.op()
    def step(self, state: AgentState) -> AgentState:
        """Run a step of the agent."""
        Console.step_start("agent", "green")
        ref = weave.obj_ref(state)
        if ref:
            print("state ref:", ref.uri())

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.system_message},
        ]
        messages += state.history

        # make type checkers happy by passing NotGiven instead of None
        tools = NotGiven()
        if self.tools:
            tools = chat_call_tool_params(self.tools)

        Console.chat_response_start()

        client = OpenAI()
        stream = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=messages,
            tools=tools,
            stream=True,
            timeout=15,
        )
        wrapped_stream = OpenAIStream(stream)
        for chunk in wrapped_stream:
            if chunk.choices[0].delta.content:
                Console.chat_message_content_delta(chunk.choices[0].delta.content)

        response = wrapped_stream.final_response()
        response_message = response.choices[0].message
        if response_message.content:
            Console.chat_response_complete(response_message.content)

        new_messages = []
        new_messages.append(response_message.model_dump(exclude_none=True))
        if response_message.tool_calls:
            new_messages.extend(
                perform_tool_calls(self.tools, response_message.tool_calls, state=state)
            )

        return AgentState(history=state.history + new_messages)

    @weave.op()
    def run(self, state: AgentState):
        """Run the agent until user intervention is needed."""
        state = self.step(state)
        return state
