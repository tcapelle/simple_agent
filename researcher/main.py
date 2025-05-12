import sys
import os
from pathlib import Path
from dataclasses import dataclass
import simple_parsing as sp
from typing import Optional, Tuple

from rich import print
import weave

from researcher.agent import Agent
from researcher.state import AgentState
from researcher.console import Console
from researcher.config import SYSTEM_MESSAGE, DEFAULT_MODEL
from researcher.tools import setup_retriever, find_manuscript, read_from_file, count_words, get_user_input, DEFAULT_TOOLS


# @weave.op
# def get_user_input(prompt: str = "User input: "):
#     return input(prompt)


# @weave.op
# def user_input_step(state: AgentState) -> AgentState:
#     Console.step_start("user_input", "purple")
#     user_input = get_user_input()
#     return AgentState(
#         messages=state.messages
#         + [
#             {
#                 "role": "user",
#                 "content": user_input,
#             }
#         ],
#     )

@weave.op
def session(agent: Agent, agent_state: AgentState):
    try:
        while True:
            agent_state = agent.run(agent_state)
            # agent_state = user_input_step(agent_state)
            # Check for exit commands in last message
            last_message = agent_state.messages[-1]["content"].lower()
            if last_message in ["exit", "quit", "bye"]:
                print("\nEnding session at user request.")
                break    
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    
    return agent_state


def get_initial_prompt() -> str:
    if len(sys.argv) < 2:
        return get_user_input("Initial prompt: ")
    initial_prompt = " ".join(sys.argv[1:])
    print("Initial prompt:", initial_prompt)
    return initial_prompt


def handle_existing_manuscript() -> Tuple[Optional[str], Optional[str]]:
    manuscript_path, location = find_manuscript()
    if not os.path.exists(manuscript_path):
        return None, None
        
    content = read_from_file(manuscript_path)
    word_count = count_words(content)
    
    print("\n" + "="*80)
    print("Found existing manuscript!")
    print(f"📄 Location: {location}")
    print(f"📊 Stats:")
    print(f"- Characters: {len(content):,}")
    print(f"- Words: {word_count:,}")
    
    user_input = get_user_input("Would you like to continue working on this manuscript? (yes/no)")
    
    if user_input.lower().startswith("y"):
        return content, word_count
    
    print("\nStarting fresh with a new manuscript.\n")
    return None, None

def create_initial_state(existing_content: Optional[str] = None, word_count: Optional[int] = None) -> AgentState:
    initial_prompt = get_initial_prompt()
    
    if existing_content:
        return AgentState(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "assistant",
                    "content": f"Current manuscript ({word_count:,} words):\n\n{existing_content}"
                },
                {
                    "role": "user",
                    "content": initial_prompt,
                },
            ],
        )
    
    return AgentState(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": initial_prompt,
            },
        ],
    )

def main():
    @dataclass
    class MainArgs:
        """Arguments for main researcher process"""
        state: str = sp.field(
            default=None,
            help="weave ref of the state to begin from"
        )
        data_path: Path = sp.field(
            default="my_data",
            help="Folder with documents to index"
        )
        database: str = sp.field(
            default="my_data/contextual_vector_db.pkl",
            help="Path to the database file"
        )
        model_name: str = sp.field(
            default=DEFAULT_MODEL,
            help="Model name"
        )
        system_message: str = sp.field(
            default=SYSTEM_MESSAGE,
            help="System message"
        )
        max_tokens: int = sp.field(
            default=1000,
            help="Maximum tokens for context generation"
        )

    args = sp.parse(MainArgs)


    weave.init("researcher")
    Console.welcome()
    setup_retriever(args.data_path / "contextual_vector_db.pkl")

    content, word_count = handle_existing_manuscript()
    state = create_initial_state(content, word_count)
    agent = Agent(
        model_name=args.model_name,
        temperature=0.7,
        tools=DEFAULT_TOOLS,
        max_tokens=args.max_tokens
    )
    session(agent, state)


if __name__ == "__main__":
    main()
