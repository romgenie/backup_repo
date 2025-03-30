# Coding Agent Development Guidelines

## Commands
- Run agent: `python src/main.py`
- Run with custom prompt: `python src/main.py --prompt "Your prompt here"`
- Run with specific model: `python src/main.py --model "model-name"`
- Run tests: TBD (no tests currently in codebase)
- Linting: TBD (no linting config found)

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local modules
- **Formatting**: Follow PEP 8 guidelines with 4 spaces for indentation
- **Types**: Use type hints for function parameters and return values
- **Docstrings**: Use Google style docstrings with Args/Returns sections
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Error Handling**: Use try/except blocks with specific exception types
- **Logging**: Use the provided logger for all operations, especially file writes
- **Function Decorators**: Use @function_tool for all agent tools
- **Code Organization**: Group related functions under clear section headers
- **Validation**: Implement validation for all inputs and outputs