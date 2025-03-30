# Guidelines for Claude when working with this codebase

## Commands
- Run structured app: `python -m src.main`
- Run Jupyter notebooks: `jupyter notebook agents_sdk/app_assistant_voice_agents.ipynb`
- Run legacy voice agent demo: `python -m agents_sdk.app_assistant_voice_agents`
- Run dispute agent demo: `python -m agents_sdk.dispute_agent`

## Code Style
- Use Python type hints for all function parameters and return values
- Follow PEP 8 for Python code formatting
- Import order: standard library, third-party packages, local modules
- For Jupyter notebooks, keep cells concise and include markdown documentation
- Name variables descriptively, using snake_case for variables and functions
- Use docstrings for all functions and classes explaining purpose, params, and returns

## Project Structure
The codebase has been restructured into a modular package:

```
src/
├── agents/           # Each agent has its own directory
│   ├── account/      # Account agent
│   ├── knowledge/    # Product knowledge agent
│   ├── search/       # Web search agent
│   └── triage/       # Main routing agent
├── tools/            # Each tool type has its own directory
│   ├── account/      # Account tools
│   ├── web_search/   # Web search tools
│   └── file_search/  # File search tools
├── voice_pipeline/   # Voice interaction components
├── utils/            # Utility functions
├── config/           # Configuration settings
└── data/             # Data resources
    ├── audio/        # Audio files
    └── knowledge/    # Knowledge files
```

See `plans.md` for detailed structure information.

## Data Files
- Keep audio files in `src/data/audio/` (was: `agents_sdk/voice_agents_audio/`)
- Keep knowledge files in `src/data/knowledge/` (was: `agents_sdk/voice_agents_knowledge/`)
- Maintain separation between agent definitions and execution code

## Error Handling
- Use try/except blocks for API calls and external services
- Log errors appropriately with context
- Include appropriate error messages in agent responses

## Dependencies
- OpenAI Agents SDK (`openai-agents`)
- OpenAI Python SDK (`openai`)
- Audio libraries: sounddevice, numpy