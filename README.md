# AI Research Assistant

An intelligent research assistant that helps with document analysis and research tasks. The assistant can take on different personas (like a PhD advisor or reviewer) and maintains context through conversations while helping you with your research documents.

## Features

- 🤖 Multiple AI Personalities (Mom, PhD Advisor, Nice Brother, Reviewer)
- 📚 Document Processing & Analysis
- 💾 Conversation State Management
- 🔧 Extensible Tool System
- 📝 RAG (Retrieval Augmented Generation) Support

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd ai-research-assistant
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
├── rag.py               # Document retrieval system
├── tools.py             # Tool implementations and definitions
├── tool_calling.py      # Tool calling infrastructure
└── state.py             # Agent state management
```

## Features in Detail

### AI Personalities
Choose from different AI personas to get varying perspectives on your research:
- **Mom**: Supportive and encouraging feedback
- **PhD Advisor**: Academic and methodological guidance
- **Nice Brother**: Casual and friendly discussion
- **Reviewer #2**: Critical analysis and detailed feedback

### Document Processing
- Automatically processes documents in the specified folder
- Maintains context across conversations
- Uses RAG for intelligent document retrieval and reference

### State Management
- Save and resume conversations
- Track conversation history
- Reference previous discussions

### Tool System
Extensible tool system for various research tasks (can be customized based on needs)

## Development

### Adding New Tools

Create new tools by adding them to `tool_definitions.py`:
```python
@weave.op()
def your_new_tool(state: AgentState, param: str) -> AgentState:
    # Tool implementation
    return updated_state
```

### Adding New Personalities

Add new personality templates to the `prompts/` directory as text files.

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.