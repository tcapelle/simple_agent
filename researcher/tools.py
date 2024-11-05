import base64
import json
import os
import subprocess
import openai
import weave

LENGTH_LIMIT = 10000


def critique_text(text: str, personality: str) -> str:
    """Provide feedback on a given text based on the selected personality using LLM.

    Args:
        text: The text to critique.
        personality: The personality type for critique (e.g., "phd_advisor", "reviewer_number_2").

    Returns:
        The critique based on the selected personality.
    """
    # Ensure prompt path is correct
    prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    prompt_path = os.path.join(prompt_dir, f"{personality}.txt")
    with open(prompt_path, 'r') as f:
        system_prompt = f.read()

    # Prepare the messages for the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    # Initialize OpenAI client
    client = openai.OpenAI()

    # Call the OpenAI LLM using the updated chat completion API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    # Extract the assistant's message content
    critique = response.choices[0].message['content']

    return critique


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
    with open(path, "w") as f:
        f.write(content)
    return "File written successfully."


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
