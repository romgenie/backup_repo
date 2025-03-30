# OpenAI Agents SDK Reliability Extension

This extension adds comprehensive reliability functionality to the OpenAI Agents SDK, reducing hallucinations in agent responses by validating outputs against their context and removing suspicious content.

## Overview

The reliability functionality works by:

1. Running validator agents to check different aspects of agent responses:
   - URL validation
   - Number validation
   - Information validation
   - Code validation

2. When suspicious content is detected, running an editor agent to clean up the response.

3. Preserving validated information across multiple runs through context and subtask management.

This approach is based on the reliability functionality from the Upsonic framework, but adapted to work with the OpenAI Agents SDK and enhanced with additional features.

## Installation

```bash
# First, install the OpenAI Agents SDK
pip install openai-agents

# Then, clone this repository and install it
git clone https://github.com/yourusername/openai-agents-reliability.git
cd openai-agents-reliability
pip install -e .
```

## Key Features

- **Comprehensive Validation**: Validates URLs, numbers, information, and code against trusted context
- **Context Management**: Maintains a history of interactions for improved validation across multiple runs
- **Subtask Management**: Supports complex workflows with multiple subtasks
- **Caching**: Improves performance by caching validation results
- **Language Transformation**: Transforms responses to different languages while preserving reliability
- **Configurable Behavior**: Offers flexible configuration options for all aspects of reliability processing
- **Error Handling**: Provides robust error handling with retry mechanisms
- **Predefined Configurations**: Includes ready-to-use configurations for different use cases

## Usage

### Using the ReliableRunner

```python
from agents import Agent
from reliability import ReliableRunner

# Create a standard agent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
)

# Run the agent with reliability processing
result = await ReliableRunner.run(
    agent,
    "What is the capital of France?",
    reliability_level=10,  # Full reliability processing
)

print(result.final_output)
```

### Using the ReliableAgent

```python
from reliability import ReliableAgent

# Create a reliable agent
agent = ReliableAgent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    reliability_level=10,  # Full reliability processing
)

# Run the agent
result = await agent.run("What is the capital of France?")

print(result.final_output)
```

### Using Subtasks

```python
from reliability import ReliableAgent

# Create a reliable agent
agent = ReliableAgent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    reliability_level=10,
)

# Define a sequence of subtasks
subtasks = [
    "Generate three potential topics for a blog post.",
    "Choose the most interesting topic and create an outline.",
    "Write the introduction section based on the outline.",
]

# Run subtasks
result = await agent.run_subtasks(subtasks)

print(result)
```

### Using Language Transformation

```python
from reliability import ReliableAgent, SupportedLanguages

# Create a reliable agent with language transformation
agent = ReliableAgent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    reliability_level=10,
    target_language=SupportedLanguages.SPANISH,
)

# Run the agent
result = await agent.run("What is the capital of France?")

print(result.final_output)  # Response in Spanish
```

### Using Custom Configuration

```python
from reliability import ReliableAgent, ReliabilityConfig

# Create a custom configuration
config = ReliabilityConfig()
config.validation.url_validator.confidence_threshold = 0.8
config.editor.additional_instructions = "Replace suspicious content with 'unknown'."
config.cache.enabled = True

# Create a reliable agent with custom configuration
agent = ReliableAgent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    config=config,
)

# Run the agent
result = await agent.run("What is the population of Mars?")

print(result.final_output)
```

### Using Predefined Configurations

```python
from reliability import ReliableAgent, get_high_reliability_config

# Create a reliable agent with high reliability configuration
agent = ReliableAgent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    config=get_high_reliability_config(),
)

# Run the agent
result = await agent.run("What is the population of Mars?")

print(result.final_output)
```

## Components

### Core Components

- `ReliabilityProcessor`: Orchestrates the validation process
- `ReliableRunner`: Wraps the standard Runner with reliability processing
- `ReliableAgent`: Extends the standard Agent with built-in reliability processing

### Data Models

- `ValidationPoint`: Stores validation results for a specific aspect
- `ValidationResult`: Stores overall validation results
- `SourceReliability`: Defines levels of source reliability

### Managers

- `ContextManager`: Maintains context information across multiple runs
- `SubtaskManager`: Manages subtasks for complex workflows
- `CacheManager`: Caches validation results for improved performance
- `ConfigurationManager`: Manages configuration settings

### Configuration

- `ReliabilityConfig`: Comprehensive configuration for reliability processing
- `ValidatorConfig`: Configuration for individual validators
- `EditorConfig`: Configuration for the editor agent
- `CacheConfig`: Configuration for caching
- `ContextConfig`: Configuration for context management
- `SubtaskConfig`: Configuration for subtask management
- `LanguageConfig`: Configuration for language transformation
- `ErrorConfig`: Configuration for error handling

### Validation

- Enhanced validation for URLs, numbers, information, and code
- Content analysis to determine which validators to run
- Confidence scoring for validation results

### Language Transformation

- `LanguageTransformer`: Transforms responses to different languages
- Support for multiple languages via `SupportedLanguages`
- Preservation of reliability metrics across transformations

### Error Handling

- Standardized error types for different failure modes
- Retry logic with configurable parameters
- Error logging for debugging

## How It Works

1. The agent runs normally and produces a response.
2. The response is analyzed to determine which validators to run.
3. Validator agents check different aspects of the response against the context.
4. If suspicious content is detected, an editor agent removes or replaces it.
5. Language transformation is applied if a target language is specified.
6. The processed response is returned, and relevant information is cached.

This process helps reduce hallucinations by ensuring that facts in the response are backed by the context, while maintining efficiency through selective validation and caching.

## Examples

See `example.py` for complete examples of using the reliability functionality.

## Customization

You can customize the reliability functionality by:

- Adjusting the reliability level (0-10)
- Using predefined configurations (high reliability, fast, balanced)
- Creating custom configurations with specific settings
- Adding custom validation rules
- Extending the core components for specialized use cases

## License

This project is licensed under the MIT License - see the LICENSE file for details.
