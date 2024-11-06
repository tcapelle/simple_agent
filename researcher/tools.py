import base64
import json
import os
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime

import openai
import weave

from .state import AgentState
from .rag import DenseRetriever

LENGTH_LIMIT = 10000
WORKDIR = "workdir"  # Default workspace directory

# Global retriever instance
retriever = None

def ensure_workdir():
    """Ensure the workspace directory exists"""
    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)
        print(f"Created workspace directory: {WORKDIR}")

def find_manuscript():
    """
    Find the manuscript file in either current directory or workdir.
    Returns tuple of (path, location_description).
    """
    # Check common manuscript filenames
    possible_names = ["manuscript.txt", "current_manuscript.txt"]
    
    # First check workdir
    ensure_workdir()
    for name in possible_names:
        path = os.path.join(WORKDIR, name)
        if os.path.exists(path):
            return path, "workdir"
    
    # Then check current directory
    for name in possible_names:
        if os.path.exists(name):
            return name, "current directory"
    
    # If no manuscript found, return default path in workdir
    return os.path.join(WORKDIR, "manuscript.txt"), "workdir"

def get_manuscript_backup_path():
    """Get a timestamped backup path for the manuscript"""
    manuscript_path, _ = find_manuscript()
    base_dir = os.path.dirname(manuscript_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"manuscript_{timestamp}.txt")

def setup_retriever(folder):
    """Initialize the global retriever instance"""
    global retriever
    ensure_workdir()
    retriever = DenseRetriever()

class Personality(str, Enum):
    """Available personality types for critique"""
    MOM = "mom"
    PHD_ADVISOR = "phd_advisor"  # This will be our default
    NICE_BROTHER = "nice_brother"
    REVIEWER_2 = "reviewer_number_2"

def count_words(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())

@weave.op()
def critique_content(state: AgentState, personality: Personality = Personality.PHD_ADVISOR) -> AgentState:
    """Get critique for the current manuscript using specified personality.
    
    Args:
        state: Current agent state
        personality: Personality to use for critique (default: phd_advisor)
    """
    manuscript_path, _ = find_manuscript()
    
    try:
        text = read_from_file(manuscript_path)
    except FileNotFoundError:
        return AgentState(
            history=state.history
            + [{"role": "assistant", "content": "No manuscript found. Please create one first."}]
        )
    
    # Check manuscript length
    word_count = count_words(text)
    if word_count == 0:
        return AgentState(
            history=state.history
            + [{"role": "assistant", "content": "The manuscript is empty. Please provide some content first."}]
        )
    elif word_count < 1000:
        return AgentState(
            history=state.history
            + [
                {
                    "role": "assistant",
                    "content": f"The manuscript is quite short (only {word_count} words). "
                              "Consider adding more content before requesting a critique. "
                              "Would you like me to help expand on any particular section?",
                }
            ]
        )
    
    critique = critique_text(text, personality)
    return AgentState(
        history=state.history
        + [{"role": "assistant", "content": f"Critique ({personality}):\n\n{critique}"}]
    )

@weave.op()
def retrieve_documents(state: AgentState, query: str, k: int = 5) -> AgentState:
    """Retrieve relevant documents based on the query.
    
    Args:
        state: Current agent state
        query: Search query
        k: Number of documents to retrieve (default: 5)
    """
    global retriever
    if not retriever:
        raise ValueError("Retriever not initialized. Call setup_retriever first.")
    
    results = retriever.search(query=query, k=k)
    
    return AgentState(
        history=state.history
        + [
            {
                "role": "assistant",
                "content": f"I found {len(results)} relevant documents:\n\n"
                + "\n\n".join(
                    f"From {r['source']}:\n{r['text']}" for r in results
                ),
            }
        ],
    )

@weave.op()
def list_files(directory: str) -> str:
    """List names of all files in a directory.

    Args:
        directory: The directory to list.

    Returns:
        The list of files in the directory.
    """
    result = json.dumps(os.listdir(directory))
    if len(result) > LENGTH_LIMIT:
        result = result[:LENGTH_LIMIT]
        result += "\n... (truncated)"
    return result

@weave.op()
def write_to_file(path: str, content: str) -> str:
    """Write text to a file at the given path.

    Args:
        path: The path to the file.
        content: The content to write to the file.

    Returns:
        A message indicating whether the file was written successfully.
    """
    # Create backup if writing to manuscript
    manuscript_path, _ = find_manuscript()
    if os.path.exists(path) and path == manuscript_path:
        backup_path = get_manuscript_backup_path()
        with open(path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    with open(path, "w") as f:
        f.write(content)
    return f"File written successfully to {path}" + (" (backup created)" if path == manuscript_path else "")

@weave.op()
def read_from_file(path: str) -> str:
    """Read text from a file at the given path.

    Args:
        path: The path to the file.

    Returns:
        The content of the file.
    """
    with open(path, "r") as f:
        result = f.read()
        if len(result) > LENGTH_LIMIT:
            result = result[:LENGTH_LIMIT]
            result += "\n... (truncated)"
        return result

def critique_text(text: str, personality: str) -> str:
    """Provide feedback on a given text based on the selected personality using LLM."""
    prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    prompt_path = os.path.join(prompt_dir, f"{personality}.txt")
    with open(prompt_path, 'r') as f:
        system_prompt = f.read()

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )

    return response.choices[0].message.content
