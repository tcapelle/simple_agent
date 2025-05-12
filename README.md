# Researcher: Agentic Writing Assistant

![](images/researcher.gif)

Researcher is an autonomous agent powered by Mistral LLMs that helps you iteratively write and improve scientific manuscripts. It uses a set of integrated tools (retrieval, critique, file I/O, thinking) and an interactive loop to generate, revise, and manage your manuscript based on your data.

## Key Features

- 🤖 Autonomous Agentic Workflow: The agent generates content, solicits self-critique, and applies feedback.
- 🧠 Mistral LLM Integration: Chat and embedding models for content generation and retrieval.
- 📚 Retrieval Augmented Generation: Contextual vector search over your document corpus.
- 🔧 Tool-System: `list_files`, `retrieve_relevant_documents`, `critique_content`, `think`, `read_from_file`, `write_to_file`.
- 💾 Persistent State with Weave: Save and resume sessions, maintain conversation history, and manage manuscript file.
- 📄 Manuscript Management: Automatic handling of workdir/manuscript.txt, backups, and versioning.
- 🚀 Document Preprocessing: Chunk PDF documents into JSONL for indexing via `researcher.prepare`.
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

### Preprocess Documents and Build Vector DB

Process all PDFs in your data directory **and** build the contextual vector database:

```bash
researcher.prepare
```

### Start the Agent

Use the `researcher` entrypoint to launch the interactive manuscript assistant:

```bash
researcher
```

- On first run, if no database is found, you can run `researcher.prepare` (as above) to generate the preprocessed documents and vector DB.
- The agent will prompt you for an initial manuscript prompt or resume an existing one. Use `exit` or `quit` to end the session.


## License

MIT 