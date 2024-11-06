import sys
import os
import argparse

from rich import print
from rich.console import Console

import weave

# Adjust import to work when run directly
if __name__ == "__main__" and __package__ is None:
    __package__ = "researcher"

from .agent import AgentState
from .console import Console
from .config import agent
from .tools import setup_retriever, find_manuscript, read_from_file, count_words


@weave.op
def get_user_input():
    return input("User input: ")


@weave.op
def user_input_step(state: AgentState) -> AgentState:
    Console.step_start("user_input", "purple")
    ref = weave.obj_ref(state)
    if ref:
        print("state ref:", ref.uri())
    user_input = get_user_input()
    return AgentState(
        history=state.history
        + [
            {
                "role": "user",
                "content": user_input,
            }
        ],
    )


@weave.op
def session(agent_state: AgentState):
    while True:
        agent_state = agent.run(agent_state)
        agent_state = user_input_step(agent_state)


def main():
    parser = argparse.ArgumentParser(description="Researcher")
    parser.add_argument(
        "--state", type=str, help="weave ref of the state to begin from"
    )
    parser.add_argument(
        "--folder", type=str, default="data", help="Folder with documents to index"
    )

    curdir = os.path.basename(os.path.abspath(os.curdir))
    # weave.init_local_client()
    weave.init("researcher")
    Console.welcome()

    args, remaining = parser.parse_known_args()
    setup_retriever(args.folder)

    try:
        if args.state:
            state = weave.ref(args.state).get()
        else:
            raise weave.trace_server.sqlite_trace_server.NotFoundError
    except weave.trace_server.sqlite_trace_server.NotFoundError:
        # Check for existing manuscript first
        manuscript_path, location = find_manuscript()
        manuscript_exists = os.path.exists(manuscript_path)
        
        if manuscript_exists:
            content = read_from_file(manuscript_path)
            word_count = count_words(content)
            
            print("\n" + "="*80)
            print("Found existing manuscript!")
            print(f"📄 Location: {location}")
            print(f"📊 Stats:")
            print(f"- Characters: {len(content):,}")
            print(f"- Words: {word_count:,}")
            
            user_input = input("\nWould you like to continue working on this manuscript? (yes/no): ")
            
            if user_input.lower().startswith("y"):
                if len(sys.argv) < 2:
                    initial_prompt = input("What would you like to do with the manuscript? ")
                else:
                    initial_prompt = " ".join(sys.argv[1:])
                state = AgentState(
                    history=[
                        {
                            "role": "assistant",
                            "content": f"Current manuscript ({word_count:,} words):\n\n{content}"
                        },
                        {
                            "role": "user",
                            "content": initial_prompt,
                        },
                    ],
                )
                print("\nGreat! Let's continue working on the manuscript.\n")
            else:
                print("\nStarting fresh with a new manuscript.\n")
                if len(sys.argv) < 2:
                    initial_prompt = input("Initial prompt: ")
                else:
                    initial_prompt = " ".join(sys.argv[1:])
                state = AgentState(
                    history=[
                        {
                            "role": "user",
                            "content": initial_prompt,
                        },
                    ],
                )
        else:
            if len(sys.argv) < 2:
                initial_prompt = input("Initial prompt: ")
            else:
                initial_prompt = " ".join(sys.argv[1:])
                print("Initial prompt:", initial_prompt)

            state = AgentState(
                history=[
                    {
                        "role": "user",
                        "content": initial_prompt,
                    },
                ],
            )

    session(state)


if __name__ == "__main__":
    main()
