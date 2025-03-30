# Development Guidelines for OpenAI Agents Reliability Extension

## Setup & Commands
- Install: `pip install -e .`
- Run example: `python example.py`
- Run single test: `python -m unittest reliability/tests/test_file.py::TestClass::test_method`
- Lint: `flake8 reliability`
- Type check: `mypy reliability`

## Code Style
- **Typing**: Use Python type hints throughout
- **Docstrings**: Google style format
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Imports**: Standard library first, then third-party, then local
- **Line length**: 88 characters (Black default)
- **Error handling**: Use try/except with specific exceptions
- **Validation**: Use Pydantic models for data validation
- **Async**: Use asyncio for concurrent operations

## Project Structure
- `reliability/`: Main package with validation components
- Modules organized by functionality (validators, prompts, processor)
- Keep validation logic separate from runner/agent implementations
- Maintain compatibility with the OpenAI Agents SDK