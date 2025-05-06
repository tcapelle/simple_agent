# %% [markdown]
# # Introduction to Building LLM Agents with Tools and Tracing

# This script walks through the process of building a simple LLM-powered agent that can use tools (functions) to answer questions. We'll cover:
# 1. Making basic LLM calls.
# 2. Introducing Weave for tracing and observability.
# 3. Defining tools for the LLM (manually and automatically).
# 4. Implementing a basic agentic loop.
# 5. Structuring the agent using Python classes.
# 6. Running the agent on a multi-step task.

# **Prerequisites:**
# Make sure you have the necessary libraries installed:
# ```bash
# pip install litellm weave openai pydantic requests
# ```
# You also need an API key for your chosen LLM provider (e.g., OpenAI, Mistral, Anthropic) accessible as an environment variable (e.g., `OPENAI_API_KEY`, `MISTRAL_API_KEY`). LiteLLM automatically picks up keys for many providers.

# %%
# Global Configuration & Setup
import os
import litellm
import weave # Must import weave before litellm for auto-patching
import json
import inspect
from pydantic import BaseModel, Field
from typing import List, Callable, Dict, Any
from enum import Enum
import requests

# Set the model name you want to use globally.
# LiteLLM supports models from OpenAI, Anthropic, Cohere, Mistral, Gemini, etc.
# Make sure your environment has the corresponding API key set (e.g., MISTRAL_API_KEY).
MODEL_NAME = "mistral/mistral-large-latest"
# MODEL_NAME = "gpt-4o" # Example alternative

# Initialize a Weave project. Traces will be sent here.
# You can view them in the Weave UI (usually runs locally).
try:
    weave.init('intro-to-agents-script')
    print(f"Weave initialized for project 'intro-to-agents-script'. Model set to: {MODEL_NAME}")
except Exception as e:
    print(f"Failed to initialize Weave: {e}")
    print("Weave tracing will not be available.")

# %% [markdown]
# ## 1. Basic LLM Call with LiteLLM

# Let's start with a simple call to the LLM using `litellm`. LiteLLM provides a unified interface (`litellm.completion`) to interact with various LLM providers.

# %%
print("\\n--- Cell 1: Basic LLM Call ---")
# Define a simple message list (conversation history)
basic_messages = [{ "content": "Hello, LLM! How does an AI agent work?", "role": "user"}]

try:
    # Make the call
    response = litellm.completion(
        model=MODEL_NAME,
        messages=basic_messages
    )

    # Print the response content
    assistant_response = response.choices[0].message.content
    print(f"LLM Response:\\n{assistant_response}")

except Exception as e:
    print(f"An error occurred during the basic LLM call: {e}")
    print("Please ensure your API key is set correctly in your environment and the model name is valid.")

# %% [markdown]
# **Weave Tracing:** Because we imported `weave` before `litellm` and called `weave.init()`, the `litellm.completion` call above was automatically traced. You can open your Weave dashboard and navigate to the `intro-to-agents-script` project to see the trace, including the input messages, output response, latency, model used, etc. This is invaluable for debugging and monitoring.

# %% [markdown]
# ## 2. Introducing Tool Calling (Manual Definition)

# Agents become much more powerful when they can use **tools** – external functions or APIs – to get information or perform actions beyond the LLM's internal knowledge. To allow an LLM to use a tool, we need to provide it with a description (schema) of the tool, including its name, purpose, and expected arguments.

# First, let's define a simple Python function we want the LLM to be able to call. We add `@weave.op` to trace when this function actually gets executed.

# %%
print("\\n--- Cell 2a: Defining a Tool Function ---")
@weave.op 
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers.
    Args:
        a: The first number.
        b: The second number.
    """
    print(f"TOOL EXECUTED (traced by Weave): add_numbers(a={a}, b={b})")
    return a + b

print("`add_numbers` function defined and decorated with @weave.op.")

# %% [markdown]
# Next, we manually create the JSON schema describing this tool in a format that models like GPT or Mistral understand.

# %%
print("\\n--- Cell 2b: Manual Tool Schema ---")
# Manually define the tool schema
tool_add_numbers_schema = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Adds two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "The first number."
                },
                "b": {
                    "type": "integer",
                    "description": "The second number."
                }
            },
            "required": ["a", "b"]
        }
    }
}
print("Manual schema `tool_add_numbers_schema` created.")
# print(json.dumps(tool_add_numbers_schema, indent=2)) # Uncomment to view schema

# %% [markdown]
# Now, we make an LLM call, passing the `tools` parameter with our schema. We ask a question that should trigger the tool.

# %%
print("\\n--- Cell 2c: LLM Call with Manual Tool ---")
tool_call_messages = [
    {"role": "user", "content": "My lucky numbers are 77 and 11. What is their sum?"}
]

try:
    response_with_manual_tool = litellm.completion(
        model=MODEL_NAME,
        messages=tool_call_messages,
        tools=[tool_add_numbers_schema] # Provide the schema here
    )
    manual_tool_response_message = response_with_manual_tool.choices[0].message
    print("LLM call made with manual tool schema.")
    # We'll process the response in the next cell

except Exception as e:
    print(f"An error occurred during the LLM call with manual tool: {e}")
    manual_tool_response_message = None # Ensure it's defined for next cell

# %% [markdown]
# The LLM's response might contain a request to call our tool (`response.choices[0].message.tool_calls`) or it might respond directly (`response.choices[0].message.content`). If it requests a tool call, we need to:
# 1. Parse the arguments it provides.
# 2. Execute our actual Python function (`add_numbers`) with those arguments.
# 3. (In a real agent loop) Send the result back to the LLM in a new message with `role="tool"`.

# %%
print("\\n--- Cell 2d: Processing Manual Tool Response ---")
if manual_tool_response_message:
    if manual_tool_response_message.tool_calls:
        print("LLM requested a tool call:")
        for tool_call in manual_tool_response_message.tool_calls:
            function_name = tool_call.function.name
            function_args_str = tool_call.function.arguments
            print(f"  - Tool: {function_name}, Args: {function_args_str}")

            if function_name == "add_numbers":
                try:
                    # Parse arguments
                    function_args = json.loads(function_args_str)
                    # Execute the actual Python function
                    result = add_numbers(**function_args)
                    print(f"  - Execution Result: {result}")
                    # Note: In a loop, we'd send this result back to the LLM.
                except json.JSONDecodeError as e:
                    print(f"    ERROR decoding arguments: {e}")
                except Exception as e:
                    print(f"    ERROR executing tool {function_name}: {e}")
            else:
                print(f"  - Unknown tool requested: {function_name}")
    elif manual_tool_response_message.content:
        print(f"LLM responded directly: {manual_tool_response_message.content}")
    else:
        print("LLM response contained neither content nor tool calls.")
else:
    print("Skipping response processing due to error in previous cell.")

# %% [markdown]
# ## 3. Simplifying Tool Definition with a Processor Function

# Manually writing JSON schemas is tedious and error-prone. We can automate this by inspecting our Python function's signature, type hints, and docstring.

# First, let's define a helper function (`generate_tool_schema`) that takes a Python function and generates the schema.

# %%
print("\\n--- Cell 3a: Schema Generation Helper ---")
def generate_tool_schema(func: Callable) -> dict:
    """Given a Python function, generate a tool-compatible JSON schema.
    Handles basic types and Enums. Assumes docstrings are formatted for arg descriptions.
    """
    signature = inspect.signature(func)
    parameters = signature.parameters
    type_hints = get_type_hints(func)

    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func).split("\\n")[0] if inspect.getdoc(func) else "",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }

    docstring = inspect.getdoc(func)
    param_descriptions = {}
    if docstring:
        args_section = False
        current_param = None
        for line in docstring.split('\\n'):
            line_stripped = line.strip()
            if line_stripped.lower().startswith(("args:", "arguments:", "parameters:")):
                args_section = True
                continue
            if args_section:
                if ":" in line_stripped:
                    param_name, desc = line_stripped.split(":", 1)
                    param_descriptions[param_name.strip()] = desc.strip()
                elif line_stripped and not line_stripped.startswith(" "): # Heuristic: end of args section
                     args_section = False

    for name, param in parameters.items():
        is_required = param.default == inspect.Parameter.empty
        param_type = type_hints.get(name, Any)
        json_type = "string"
        param_schema = {}

        # Basic type mapping
        if param_type == str: json_type = "string"
        elif param_type == int: json_type = "integer"
        elif param_type == float: json_type = "number"
        elif param_type == bool: json_type = "boolean"
        elif hasattr(param_type, '__origin__') and param_type.__origin__ is list: # Handle List[type]
             item_type = param_type.__args__[0] if param_type.__args__ else Any
             if item_type == str: param_schema = {"type": "array", "items": {"type": "string"}}
             elif item_type == int: param_schema = {"type": "array", "items": {"type": "integer"}}
             # Add more list item types if needed
             else: param_schema = {"type": "array", "items": {"type": "string"}} # Default list item type
        elif hasattr(param_type, "__members__") and issubclass(param_type, Enum): # Handle Enum
             json_type = "string"
             param_schema["enum"] = [e.value for e in param_type]

        if not param_schema: # If not set by List or Enum
            param_schema["type"] = json_type

        param_schema["description"] = param_descriptions.get(name, "")

        if param.default != inspect.Parameter.empty and param.default is not None:
             param_schema["default"] = param.default # Note: OpenAI schema doesn't officially use default, but useful metadata

        schema["function"]["parameters"]["properties"][name] = param_schema
        if is_required:
            schema["function"]["parameters"]["required"].append(name)
    return schema

print("`generate_tool_schema` function defined.")

# %% [markdown]
# Now, we define a `function_tool` "processor". This isn't a decorator in the `@` syntax sense here, but a function that we call *after* defining our tool function. It uses `generate_tool_schema` to attach the schema to the function object itself.

# %%
print("\\n--- Cell 3b: The `function_tool` Processor ---")
def function_tool(func: Callable) -> Callable:
    """Attaches a tool schema to the function and marks it as a tool.
    Call this *after* defining your function: my_func = function_tool(my_func)
    """
    try:
        func.tool_schema = generate_tool_schema(func)
        func.is_tool = True # Mark it as a tool
    except Exception as e:
        print(f"Error processing tool {func.__name__}: {e}")
        # Optionally raise or mark as failed
        func.tool_schema = None
        func.is_tool = False
    return func

print("`function_tool` processor defined.")

# %% [markdown]
# ## 4. Using the Tool Processor

# Let's define another simple tool, `get_current_weather`, add `@weave.op` for tracing, and then process it with `function_tool`.

# %%
print("\\n--- Cell 4a: Defining and Processing a Tool ---")
@weave.op 
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location. Returns dummy data.

    Args:
        location: The city and state, e.g., San Francisco, CA
        unit: Temperature unit, 'celsius' or 'fahrenheit'. Default is celsius.
    """
    print(f"TOOL EXECUTED (traced by Weave): get_current_weather(location='{location}', unit='{unit}')")
    # Dummy implementation for simplicity
    if "tokyo" in location.lower():
        temp = "22" if unit == "celsius" else "72"
        return json.dumps({"location": location, "temperature": temp, "unit": unit, "forecast": "sunny"})
    else:
        temp = "15" if unit == "celsius" else "59"
        return json.dumps({"location": location, "temperature": temp, "unit": unit, "forecast": "partly cloudy"})

# Apply the processor *after* definition
get_current_weather = function_tool(get_current_weather)

print("`get_current_weather` defined and processed by `function_tool`.")
if get_current_weather.is_tool:
     print("Schema successfully attached.")
     # print(json.dumps(get_current_weather.tool_schema, indent=2)) # Uncomment to view schema

# %% [markdown]
# Now we can make an LLM call using the automatically generated schema attached to the function object.

# %%
print("\\n--- Cell 4b: LLM Call with Processed Tool ---")
weather_messages = [
    {"role": "user", "content": "What's the weather like in Tokyo today in Celsius?"}
]
processed_tool_response_message = None # Init for error handling

try:
    # Check if the tool was processed correctly
    if get_current_weather.is_tool and get_current_weather.tool_schema:
        weather_response = litellm.completion(
            model=MODEL_NAME,
            messages=weather_messages,
            tools=[get_current_weather.tool_schema] # Use the attached schema
        )
        processed_tool_response_message = weather_response.choices[0].message
        print("LLM call made with processed tool schema.")
    else:
        print("Skipping LLM call as get_current_weather tool schema is missing.")

except Exception as e:
    print(f"An error occurred during the LLM call with processed tool: {e}")

# %% [markdown]
# We process the response just like before, checking for `tool_calls`.

# %%
print("\\n--- Cell 4c: Processing Processed Tool Response ---")
if processed_tool_response_message:
    if processed_tool_response_message.tool_calls:
        print("LLM requested a tool call:")
        for tool_call in processed_tool_response_message.tool_calls:
            function_name = tool_call.function.name
            function_args_str = tool_call.function.arguments
            print(f"  - Tool: {function_name}, Args: {function_args_str}")

            tool_to_run = None
            if hasattr(get_current_weather, 'is_tool') and function_name == get_current_weather.__name__:
                 tool_to_run = get_current_weather

            if tool_to_run:
                try:
                    function_args = json.loads(function_args_str)
                    result = tool_to_run(**function_args)
                    print(f"  - Execution Result: {result}")
                except json.JSONDecodeError as e:
                    print(f"    ERROR decoding arguments: {e}")
                except Exception as e:
                    print(f"    ERROR executing tool {function_name}: {e}")
            else:
                print(f"  - Unknown or unprocessed tool requested: {function_name}")
    elif processed_tool_response_message.content:
        print(f"LLM responded directly: {processed_tool_response_message.content}")
    else:
        print("LLM response contained neither content nor tool calls.")
else:
     print("Skipping response processing due to error or missing schema in previous cell.")

# %% [markdown]
# ## 5. Agentic Loop with Multiple Tools (including Real API)

# A true agent often needs multiple interactions with the LLM and tools to complete a task. This involves a loop:
# 1. Send current history (including previous tool results) to LLM.
# 2. LLM responds (content or tool call).
# 3. If tool call -> Execute tool, add tool request and tool result to history.
# 4. If content -> Add content to history, potentially stop if task is complete.
# 5. Repeat.

# Let's define a more complex tool using the PokeAPI and re-define `add_numbers` for this context.

# %%
print("\\n--- Cell 5a: Defining Tools for Agent Loop ---")

@weave.op 
def get_pokemon_info(pokemon_name: str) -> str:
    """Fetches minimal information (name, ID, weight) for a specific Pokemon. Weight is in hectograms.

    Args:
        pokemon_name: The name or Pokedex ID of the Pokemon.
    """
    print(f"TOOL EXECUTED (traced by Weave): get_pokemon_info(pokemon_name='{pokemon_name}')")
    base_url = "https://pokeapi.co/api/v2/pokemon/"
    try:
        response = requests.get(f"{base_url}{pokemon_name.lower().strip()}")
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        info = {
            "name": data.get('name', 'Unknown').capitalize(),
            "id": data.get('id', -1),
            "weight": data.get('weight', -1) # Weight in hectograms
        }
        return json.dumps(info)
    except requests.exceptions.HTTPError as e:
         if e.response.status_code == 404:
              return json.dumps({"error": f"Pokemon '{pokemon_name}' not found."})
         else:
              return json.dumps({"error": f"API request failed: {e}"})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Request failed: {str(e)}"})
    except Exception as e: # Catch potential JSON parsing errors etc.
         return json.dumps({"error": f"Error processing pokemon {pokemon_name}: {str(e)}"})

get_pokemon_info = function_tool(get_pokemon_info) # Process it

# Define add_numbers again for this context (or ensure it's imported/available)
@weave.op
def add_numbers_for_loop(a: int, b: int) -> int:
    """Adds two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    print(f"TOOL EXECUTED (traced by Weave): add_numbers_for_loop(a={a}, b={b})")
    return a + b
add_numbers_for_loop = function_tool(add_numbers_for_loop) # Process it

print("Agent loop tools (`get_pokemon_info`, `add_numbers_for_loop`) defined and processed.")

# %% [markdown]
# Now, let's run a simple loop for a few turns, demonstrating the flow.

# %%
print("\\n--- Cell 5b: Running the Agentic Loop Demo ---")

# List of available tools *functions* for this loop
agent_loop_tools = [get_pokemon_info, add_numbers_for_loop]
# List of tool *schemas* to pass to the LLM
agent_loop_tool_schemas = [t.tool_schema for t in agent_loop_tools if t.is_tool]

# Initial conversation history
loop_chat_history = [
    {"role": "system", "content": "You are a helpful assistant. Use tools for Pokemon info or calculations."},
    {"role": "user", "content": "What is the weight of Pikachu? And add 100 to that."}
]
loop_max_turns = 4 # Limit interactions

for turn in range(loop_max_turns):
    print(f"--- Agent Loop Turn {turn + 1}/{loop_max_turns} ---")
    print(f"Sending {len(loop_chat_history)} messages to LLM...")

    try:
        # 1. Send history to LLM
        loop_llm_response = litellm.completion(
            model=MODEL_NAME,
            messages=loop_chat_history,
            tools=agent_loop_tool_schemas
        )
        loop_response_message = loop_llm_response.choices[0].message
        # Add LLM response (potential tool call or content) to history immediately
        loop_chat_history.append(loop_response_message.model_dump(exclude_none=True))
        print("LLM response received.") # Less verbose print

        # 2. Process response
        if loop_response_message.tool_calls:
            print("LLM requested tool calls:")
            # 3. Execute tools and add results to history
            for tool_call in loop_response_message.tool_calls:
                function_name = tool_call.function.name
                function_args_str = tool_call.function.arguments
                print(f"  - Requesting: {function_name}({function_args_str})")

                tool_to_run = next((t for t in agent_loop_tools if t.__name__ == function_name), None)
                tool_result_content = ""
                if tool_to_run:
                    try:
                        function_args = json.loads(function_args_str)
                        tool_result = tool_to_run(**function_args)
                        tool_result_content = str(tool_result)
                        print(f"  - Executed {function_name}. Result snippet: {tool_result_content[:100]}...")
                    except json.JSONDecodeError as e:
                        print(f"    ERROR decoding arguments for {function_name}: {e}")
                        tool_result_content = json.dumps({"error": f"Argument decode error: {e}"})
                    except Exception as e:
                        print(f"    ERROR executing tool {function_name}: {e}")
                        tool_result_content = json.dumps({"error": f"Execution error: {e}"})
                else:
                    print(f"    ERROR: Tool '{function_name}' not found.")
                    tool_result_content = json.dumps({"error": "Tool definition not found by agent."})

                # Add tool result message to history segment for this step
                loop_chat_history.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_result_content
                })

        elif loop_response_message.content:
            # 4. LLM gave content response
            print(f"LLM content response: {loop_response_message.content}")
            # Simple stop condition: If LLM provides content and didn't just use a tool, maybe it's done.
            if loop_chat_history[-2].get("role") != "tool": # Check if previous wasn't a tool result
                 print("Assuming task complete after LLM content response.")
                 break
        else:
            print("LLM response had neither content nor tool calls. Stopping loop.")
            break

    except Exception as e:
        print(f"Error during agent loop turn {turn + 1}: {e}")
        break # Stop loop on error

    if turn == loop_max_turns - 1:
        print("Max turns reached for demo loop.")

# print("\nFinal loop history:", json.dumps(loop_chat_history, indent=2)) # Uncomment for detailed history

# %% [markdown]
# ## 6. Structuring the Agent with Classes

# The loop above works, but for more complex agents, encapsulating the logic and state within classes is much better. We'll define:
# - `AgentState`: A Pydantic model to hold the conversation history and potentially other state.
# - `SimpleAgent`: A class containing the agent's configuration (model, system message, tools) and logic (`step`, `run`).

# %%
print("\\n--- Cell 6: Defining AgentState Class ---")
class AgentState(BaseModel):
    """Manages the state of the agent."""
    history: List[Dict[str, Any]] = Field(default_factory=list)
    final_assistant_content: str | None = None # Populated at the end of a run

print("`AgentState` class defined.")

# %% [markdown]
# ## 7. Defining the Agent Class

# The `SimpleAgent` class holds the core logic.
# - `__init__`: Sets up the model, system message, and processes the tools list.
# - `_find_tool`: Helper to locate the Python function for a given tool name.
# - `step`: Performs one cycle of LLM call -> tool execution -> history update. Decorated with `@weave.op`.
# - `run`: Executes the `step` method in a loop. Decorated with `@weave.op`.

# %%
print("\\n--- Cell 7: Defining SimpleAgent Class ---")
class SimpleAgent:
    """A simple agent class with tracing, state, and tool processing."""
    def __init__(self, model_name: str, system_message: str, tools: List[Callable]):
        self.model_name = model_name
        self.system_message = system_message
        self.tools = tools # Expects functions already processed by function_tool
        self.tool_schemas = []
        self.tool_map = {} # For quick lookup
        for tool in self.tools:
            if hasattr(tool, 'is_tool') and tool.is_tool and hasattr(tool, 'tool_schema') and tool.tool_schema:
                self.tool_schemas.append(tool.tool_schema)
                self.tool_map[tool.__name__] = tool
            else:
                print(f"Warning: Tool '{getattr(tool,'__name__','Unnamed')}' provided to SimpleAgent is missing schema or not processed. Skipping.")

    # Note: _find_tool could be replaced by just using self.tool_map.get(tool_name)
    def _find_tool(self, tool_name: str) -> Callable | None:
        return self.tool_map.get(tool_name)

    @weave.op(name="SimpleAgent.step") # Trace each step
    def step(self, current_state: AgentState) -> AgentState:
        messages_to_send = [{"role": "system", "content": self.system_message}] + current_state.history
        new_history_segment = [] # Messages added in this step

        try:
            # === LLM Call ===
            llm_response = litellm.completion(
                model=self.model_name,
                messages=messages_to_send,
                tools=self.tool_schemas if self.tool_schemas else None
            )
            response_message = llm_response.choices[0].message
            new_history_segment.append(response_message.model_dump(exclude_none=True))

            # === Tool Call Processing ===
            if response_message.tool_calls:
                print("Agent Step: LLM requested tool calls.")
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args_str = tool_call.function.arguments
                    print(f"  - Attempting: {function_name}({function_args_str})")

                    tool_function = self._find_tool(function_name)
                    tool_result_content = ""
                    if tool_function:
                        try:
                            function_args = json.loads(function_args_str)
                            # Execute the @weave.op decorated tool function
                            result = tool_function(**function_args)
                            tool_result_content = str(result)
                        except json.JSONDecodeError as e:
                            print(f"    ERROR decoding args for {function_name}: {e}")
                            tool_result_content = json.dumps({"error": f"Argument decode error: {e}"})
                        except Exception as e:
                            print(f"    ERROR executing tool {function_name}: {e}")
                            tool_result_content = json.dumps({"error": f"Execution error: {e}"})
                    else:
                        print(f"    ERROR: Tool '{function_name}' not found by agent.")
                        tool_result_content = json.dumps({"error": "Tool definition not found by agent."})

                    # Add tool result message to history segment for this step
                    new_history_segment.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_result_content
                    })
            elif response_message.content:
                print(f"Agent Step: LLM provided direct content (snippet): {response_message.content[:100]}...")

        except Exception as e:
            print(f"ERROR in Agent Step: {e}")
            # Add an error message to history to indicate failure
            new_history_segment.append({"role": "assistant", "content": f"Agent error in step: {str(e)}"})

        # Return new state ONLY with history updated. Final content is set in run().
        return AgentState(history=current_state.history + new_history_segment)

    @weave.op(name="SimpleAgent.run") # Trace the entire run
    def run(self, initial_user_prompt: str, max_turns: int = 10) -> AgentState:
        print(f"\\n--- Agent Run Starting (max_turns={max_turns}) ---")
        print(f"User Prompt: '{initial_user_prompt}'")
        current_state = AgentState(history=[{"role": "user", "content": initial_user_prompt}])

        for i in range(max_turns):
            print(f"--- Agent Turn {i + 1}/{max_turns} ---")
            last_history_len = len(current_state.history)
            current_state = self.step(current_state) # Run one step

            # Check the outcome of the step to decide whether to continue
            step_outcome = current_state.history[last_history_len:] # Messages added by step
            assistant_msg_in_outcome = next((m for m in step_outcome if m['role'] == 'assistant'), None)

            if assistant_msg_in_outcome:
                 if assistant_msg_in_outcome.get('content') and not assistant_msg_in_outcome.get('tool_calls'):
                     # If the assistant replied with content AND did not request tools,
                     # assume it might be done. Check if tools were executed just before this.
                     if last_history_len == 0 or current_state.history[last_history_len-1].get('role') != 'tool':
                          print("Agent Run: Concluding after direct content response.")
                          break
                     else:
                          print("Agent Run: Assistant provided content after tool use, continuing...")

                 # If assistant requested tools, loop must continue
                 elif assistant_msg_in_outcome.get('tool_calls'):
                     pass # Continue loop to process tools in next step

                 # Handle case where assistant response is empty (error?)
                 elif not assistant_msg_in_outcome.get('content') and not assistant_msg_in_outcome.get('tool_calls'):
                     print("Agent Run: Assistant response is empty, concluding.")
                     break
            else:
                 # Should not happen unless step had a major error before creating assistant message
                 print("Agent Run: No assistant message found in step outcome, concluding.")
                 break

            if i == max_turns - 1:
                print("Agent Run: Max turns reached.")

        # --- Post-Loop: Populate final_assistant_content ---
        final_text_response = None
        for msg in reversed(current_state.history):
            if msg.get('role') == 'assistant' and msg.get('content') and not msg.get('tool_calls'):
                final_text_response = msg.get('content')
                break

        print("--- Agent Run Finished ---")
        # Return final state including the history and the extracted final content
        return AgentState(history=current_state.history, final_assistant_content=final_text_response)

print("`SimpleAgent` class defined.")

# %% [markdown]
# ## 8. Running the Agent on a Complex Query

# Now, let's instantiate the `SimpleAgent` and give it the multi-step task of calculating the combined weight of several Pokémon. This requires:
# - A system prompt guiding the agent on the multi-step process.
# - Providing the necessary tools (`get_pokemon_info`, `add_numbers_for_loop`).
# - Setting enough `max_turns` for the agent to call the tools multiple times and synthesize the result.

# %%
print("\\n--- Cell 8: Running Agent on Complex Query ---")

# Ensure tools are available (defined/processed in Cell 5a)
if 'get_pokemon_info' not in globals() or not getattr(get_pokemon_info,'is_tool',False) or \
   'add_numbers_for_loop' not in globals() or not getattr(add_numbers_for_loop,'is_tool',False):
    raise NameError("Required tools not defined or processed. Ensure Cell 5a executed.")

agent_tools_complex = [get_pokemon_info, add_numbers_for_loop]

# System message guiding the multi-step reasoning
complex_system_msg = (
    "You are a Pokemon data assistant. Your goal is to find the combined weight of specified Pokemon. "
    "You MUST use 'get_pokemon_info' for EACH Pokemon to get its weight (in hectograms). "
    "After getting all weights, sum them using 'add_numbers_for_loop' if needed, or sum manually. "
    "State only the final combined weight in hectograms. If a Pokemon isn't found, report that and exclude it."
)

# Instantiate the agent
try:
    complex_agent = SimpleAgent(
        model_name=MODEL_NAME,
        system_message=complex_system_msg,
        tools=agent_tools_complex
    )

    # Define the complex query
    query = "What is the combined weight of Pikachu, Charmander, Bulbasaur, and Squirtle?"

    # Run the agent (needs more turns for multiple calls + summation)
    final_state = complex_agent.run(initial_user_prompt=query, max_turns=12)

    # Print the final extracted answer
    print(f"\\n>>> Final Answer by Agent: {final_state.final_assistant_content or 'No final textual answer found.'}")

except Exception as e:
    print(f"An error occurred during complex agent run: {e}")

# %% [markdown]
# ## 9. Conclusion

# We have successfully built a simple agent framework from scratch, incorporating:
# - Interaction with LLMs via LiteLLM.
# - Automatic tracing with Weave.
# - Tool definition and schema generation.
# - An agentic loop for multi-step reasoning.
# - Class-based structure for organization.

# **Next Steps:**
# - Explore the traces in the Weave dashboard to understand the agent's execution flow, LLM inputs/outputs, and tool calls.
# - Try different models by changing `MODEL_NAME`.
# - Add more complex tools (e.g., web search, file I/O).
# - Enhance the `AgentState` with more memory or context.
# - Implement more sophisticated error handling and retry mechanisms.
# - Convert this script into a Jupyter Notebook (`.ipynb`) for interactive exploration.

# %%
print("\\nScript finished.")
