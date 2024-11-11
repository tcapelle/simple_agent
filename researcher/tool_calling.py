import inspect
import json
from typing import Callable, get_type_hints, Any

from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolParam

from .console import Console


def generate_json_schema(func: Callable) -> dict:
    """Given a function, generate an OpenAI tool compatible JSON schema.
    
    Handles special cases like AgentState and Enums.
    """
    # Extract function signature
    signature = inspect.signature(func)
    parameters = signature.parameters

    # Extract annotations
    type_hints = get_type_hints(func)

    # Initialize the schema structure
    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__.split("\n")[0] if func.__doc__ else "",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }

    # Process each parameter
    for name, param in parameters.items():

        # Determine if this parameter is required (no default value)
        is_required = param.default == inspect.Parameter.empty

        # Get parameter type
        param_type = type_hints.get(name, Any)
        
        # Convert Python types to JSON schema types
        if param_type == str:
            json_type = "string"
        elif param_type == int:
            json_type = "integer"
        elif param_type == float:
            json_type = "number"
        elif param_type == bool:
            json_type = "boolean"
        else:
            json_type = "string"  # Default to string for complex types

        # Extract parameter description from docstring
        param_desc = ""
        if func.__doc__:
            for line in func.__doc__.split("\n"):
                if f"{name}:" in line:
                    param_desc = line.split(":", 1)[1].strip()
                    break

        # Build parameter schema
        param_schema = {
            "type": json_type,
            "description": param_desc
        }

        # Handle Enum types
        if hasattr(param_type, "__members__"):
            param_schema["enum"] = [e.value for e in param_type]

        # Add default value if present
        if param.default != inspect.Parameter.empty and param.default is not None:
            param_schema["default"] = param.default

        schema["function"]["parameters"]["properties"][name] = param_schema

        if is_required:
            schema["function"]["parameters"]["required"].append(name)

    return schema


def chat_call_tool_params(tools: list[Callable]) -> list[ChatCompletionToolParam]:
    chat_tools = [generate_json_schema(tool) for tool in tools]
    return [ChatCompletionToolParam(**tool) for tool in chat_tools]


def get_tool(tools: list[Callable], name: str) -> Callable:
    for t in tools:
        if t.__name__ == name:
            return t
    raise KeyError(f"No tool with name {name} found")


def perform_tool_calls(
    tools: list[Callable], tool_calls: list[ChatCompletionMessageToolCall]
) -> list[dict]:
    messages = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        tool = get_tool(tools, function_name)
        function_response = None
        tool_call_s = f"{function_name}({tool_call.function.arguments})"
        Console.tool_call_start(tool_call_s)
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            function_response = str(e)
        if not function_response:
            try:
                function_response = tool(**function_args)
            except Exception as e:
                function_response = str(e)

        additional_message = None
        if isinstance(function_response, tuple):
            additional_message = function_response[1]
            function_response = str(function_response[0])
        else:
            function_response = str(function_response)

        Console.tool_call_complete(function_response)
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": function_response,
            }
        )
        if additional_message:
            messages.append(additional_message)
    return messages
