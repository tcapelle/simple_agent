SYSTEM_MESSAGE = """Assistant is a writing assistant named "simple_writer".
The assistant focuses on helping create and critique scientific papers and literature.
It is designed to autonomously assist without stopping for user input unless necessary.
"""

from researcher.tools import list_files, write_to_file, read_from_file
from researcher.agent import Agent

agent = Agent(
    model_name="gpt-4o",
    temperature=0.7,
    system_message=SYSTEM_MESSAGE,
    tools=[list_files, write_to_file, read_from_file],
)
