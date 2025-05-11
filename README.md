# Researcher: Agentic Writing Assistant

Researcher is an autonomous agent powered by Mistral LLMs that helps you iteratively write and improve scientific manuscripts. It uses a set of integrated tools (retrieval, critique, file I/O, thinking) and an interactive loop to generate, revise, and manage your manuscript based on your data.

## Key Features

- 🤖 Autonomous Agentic Workflow: The agent generates content, solicits self-critique, and applies feedback.
- 🧠 Mistral LLM Integration: Chat and embedding models for content generation and retrieval.
- 📚 Retrieval Augmented Generation: Contextual vector search over your document corpus.
- 🔧 Tool-System: `list_files`, `retrieve_relevant_documents`, `critique_content`, `think`, `read_from_file`, `write_to_file`.
- 💾 Persistent State with Weave: Save and resume sessions, maintain conversation history, and manage manuscript file.
- 📄 Manuscript Management: Automatic handling of workdir/manuscript.txt, backups, and versioning.
- 🚀 Document Preprocessing: Chunk PDF documents into JSONL for indexing via `researcher_prepare`.
- 🔄 Interactive REPL: Command-line interface to interact with the agent, provide initial prompts, and exit gracefully.

## Installation

```bash
git clone https://github.com/tcapelle/phd_researcher.git
cd phd_researcher
pip install -e .
```

Make sure to set your Mistral API key:

```bash
export MISTRAL_API_KEY=your_api_key_here
```

## Usage

### Preprocess Documents

Process PDF files into a JSONL corpus:

```bash
researcher_prepare --data_dir path/to/your_pdfs --output_file path/to/processed_documents.jsonl --chunk_size 512
```

### Build and Run the Agent

Ensure your vector database exists (created automatically from preprocessed docs). Then start the agent:

```bash
researcher --data_path path/to/ --database path/to/contextual_vector_db.pkl --model_name mistral-medium-latest --max_tokens 1000
```

- On first run, if no database is found, run `researcher_prepare` to generate preprocessed documents.

- The agent will prompt you for an initial manuscript prompt or resume an existing one. Use commands `exit` or `quit` to end the session.

### Command-line Options

```
researcher [initial prompt]           # Start a new session with an initial prompt
researcher --state weave_state_ref    # Resume from a saved Weave state
researcher --data_path my_data        # Path to document corpus
researcher --database my_data/contextual_vector_db.pkl  # Path to vector DB
researcher --model_name mistral-medium-latest           # LLM model name
researcher --max_tokens 1000          # Max tokens for context in retrieval and critiquing
```

## Project Structure

```
.
├── dev/                       # Workshop materials and notebooks
├── my_data/                   # Example data directory
├── researcher/                # Core agent implementation
│   ├── agent.py               # Agent loop and logic
│   ├── console.py             # CLI interface
│   ├── config.py              # Default settings and tool definitions
│   ├── tools.py               # Tools available to the agent
│   ├── rag.py                 # Contextual vector DB implementation
│   ├── preprocess.py          # Document preprocessing utilities
│   ├── tool_calling.py        # Weave tool integration
│   ├── mistral_helper.py      # Helper functions for Mistral API
│   └── prompts/               # System and personality prompt templates
├── call_openai.py             # Legacy script, may be deprecated
├── bug.py                     # Debug utilities
├── pyproject.toml             # Project metadata and dependencies
└── README.md                  # This file
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

MIT 