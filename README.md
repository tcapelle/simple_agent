# AI Research Assistant

An intelligent research assistant that helps with document analysis and research tasks. The assistant can take on different personas (like a PhD advisor or reviewer) and maintains context through conversations while helping you with your research documents. It uses advanced RAG (Retrieval Augmented Generation) techniques including contextual embeddings and hybrid search to provide accurate and relevant assistance.

## Features

- 🤖 Multiple AI Personalities (Mom, PhD Advisor, Nice Brother, Reviewer)
- 📚 Advanced Document Processing & Analysis
- 💾 Conversation State Management
- 🔧 Extensible Tool System
- 📝 Enhanced RAG Support with Contextual Embeddings
- 🔍 Hybrid Search with BM25 and Semantic Search
- 🎯 Document Reranking for Improved Accuracy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tcapelle/phd_researcher.git
   cd phd_researcher
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage

Start a new conversation:
```bash
python -m researcher "What can you tell me about the documents in the data folder?"
```

### Advanced Usage

1. Specify a custom data folder:
   ```bash
   python -m researcher --folder path/to/documents
   ```

2. Resume from a previous state:
   ```bash
   python -m researcher --state weave_state_reference
   ```

3. Preprocess documents:
   ```bash
   python -m researcher.prepare --data_dir path/to/documents
   ```

## Project Structure

```
researcher/
├── prompts/                # AI Personality Templates
│   ├── mom.txt            # Supportive, nurturing persona
│   ├── nice_brother.txt   # Friendly, casual persona
│   ├── phd_advisor.txt    # Academic, mentoring persona
│   └── reviewer_2.txt     # Critical, detailed persona
├── agent.py               # Core agent logic
├── config.py             # Configuration settings
├── console.py            # Console interface
├── rag.py               # Enhanced document retrieval system
├── tools.py             # Tool implementations and definitions
├── tool_calling.py      # Tool calling infrastructure
├── state.py             # Agent state management
└── preprocess.py        # Document preprocessing utilities
```

## Features in Detail

### AI Personalities
Choose from different AI personas to get varying perspectives on your research:
- **Mom**: Supportive and encouraging feedback
- **PhD Advisor**: Academic and methodological guidance
- **Nice Brother**: Casual and friendly discussion
- **Reviewer #2**: Critical analysis and detailed feedback

### Advanced Document Processing
- Sophisticated document preprocessing with customizable chunking
- Contextual embeddings for improved document understanding
- Hybrid search combining BM25 and semantic search
- Document reranking for enhanced retrieval accuracy
- Maintains context across conversations
- Uses advanced RAG techniques for intelligent document retrieval

### State Management
- Save and resume conversations
- Track conversation history
- Reference previous discussions
- Persistent state storage with Weave

### Tool System
- Extensible tool system for various research tasks
- Built-in tools for document analysis and retrieval
- Custom tool support for specialized needs

## Development

### Adding New Tools

Create new tools by adding them to `tools.py`:
```python
@weave.op
def your_new_tool(state: AgentState, param: str) -> AgentState:
    # Tool implementation
    return updated_state
```

### Adding New Personalities

Add new personality templates to the `prompts/` directory as text files.

### Configuring RAG

The system supports advanced RAG configurations:
- Customize embedding models
- Adjust retrieval parameters
- Configure hybrid search weights
- Set up reranking parameters

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# Enhancing RAG with Contextual Retrieval

> Note: For more background information on Contextual Retrieval, including additional performance evaluations on various datasets, we recommend reading our accompanying  [blog post](https://www.anthropic.com/news/contextual-retrieval).

Retrieval Augmented Generation (RAG) enables Claude to leverage your internal knowledge bases, codebases, or any other corpus of documents when providing a response. Enterprises are increasingly building RAG applications to improve workflows in customer support, Q&A over internal company documents, financial & legal analysis, code generation, and much more.

In a [separate guide](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/retrieval_augmented_generation/guide.ipynb), we walked through setting up a basic retrieval system, demonstrated how to evaluate its performance, and then outlined a few techniques to improve performance. In this guide, we present a technique for improving retrieval performance: Contextual Embeddings.

In traditional RAG, documents are typically split into smaller chunks for efficient retrieval. While this approach works well for many applications, it can lead to problems when individual chunks lack sufficient context. Contextual Embeddings solve this problem by adding relevant context to each chunk before embedding. This method improves the quality of each embedded chunk, allowing for more accurate retrieval and thus better overall performance. Averaged across all data sources we tested, Contextual Embeddings reduced the top-20-chunk retrieval failure rate by 35%.

The same chunk-specific context can also be used with BM25 search to further improve retrieval performance. We introduce this technique in the “Contextual BM25” section.

In this guide, we'll demonstrate how to build and optimize a Contextual Retrieval system using a dataset of 9 codebases as our knowledge base. We'll walk through:

```python
class PreprocessingArgs:
    """Arguments for preprocessing PDF documents"""
    data_dir: Path = sp.field(default=Path("my_data/"), help="Directory containing PDF files to process")
    output_file: Path = sp.field(default=None, help="Output JSONL file path. Defaults to 'processed_documents.jsonl' in data_dir")
    chunk_size: int = sp.field(default=512, help="Target size of each chunk in characters")
    max_workers: int = sp.field(default=6, help="Maximum number of worker processes. Defaults to CPU count")
    max_tokens_len: int = sp.field(default=100000, help="Maximum number of tokens allowed per document")
    skipped_docs_file: Path = sp.field(default=None, help="Output file for skipped documents. Defaults to 'skipped_documents.jsonl' in data_dir")
```