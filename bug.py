import litellm
import weave
from typing import List, Dict, Any

weave.init("litellm_bug")

messages = [{ "content": "Hello, LLM! How does an AI agent work?", "role": "user"}]

@weave.op
def call_model(model_name: str, messages: List[Dict[str, Any]], **kwargs) -> str:
    "Call a model with the given messages and kwargs."
    response = litellm.completion(
        model=model_name,
        messages=messages,
        **kwargs
    )

    assistant_response = response.choices[0].message.content
    return assistant_response

response = call_model(model_name="mistral/mistral-small", messages=messages)
print(response)