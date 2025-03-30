# OpenAI Agents SDK Interactive Playground

This directory contains the interactive playground components for exploring the OpenAI Agents SDK.

## Components

### Basic Playground

The basic playground (`playground.py`) provides:
- Model selection from standard models
- System and user message input
- Basic model settings configuration
- Tool configuration options
- Code and API call previews

### Advanced Playground

The advanced playground (`sdk_models/showcase.py`) provides enhanced features:
- **Model Support**
  - Comprehensive model catalog
  - Multiple provider support (OpenAI, Azure OpenAI, Anthropic)
  - Detailed capability visualization
  - Provider configuration

- **Output Schemas**
  - Schema builder with templates
  - Schema visualization
  - Schema validation testing
  - Example generation

- **Advanced Tool Integration**
  - Tool definition builder
  - Mock response testing
  - Tool execution visualization
  - Tool usage previews

- **Response Types & Reasoning Models**
  - Multiple response formats (JSON, reasoning, analytical, etc.)
  - Reasoning settings configuration
  - Dynamic system instruction generation
  - Example outputs

## Usage

The playground is integrated with the main application. You can access it through:
- Basic Playground: Select "Playground" in the navigation menu
- Advanced Playground: Select "Advanced Playground" in the navigation menu

## Installation

To install dependencies for the advanced playground:

```bash
cd interactive_playground/sdk_models
python setup.py
```

## Structure

```
interactive_playground/
├── __init__.py           # Package initialization
├── playground.py         # Basic playground implementation
├── README.md             # This file
└── sdk_models/           # Advanced playground features
    ├── __init__.py       # Package initialization
    ├── __main__.py       # Standalone entry point
    ├── additional_features.md  # Feature roadmap
    ├── feature_integration.py  # Integration with the playground
    ├── model_definitions.py    # Model catalog
    ├── model_selector.py       # Model selection UI
    ├── output_schemas.py       # Schema support
    ├── provider_config.py      # Provider configuration
    ├── README.md               # SDK models documentation
    ├── response_types.py       # Response type configuration
    ├── setup.py                # Dependencies installation
    ├── showcase.py             # Main showcase UI
    ├── showcase_wrapper.py     # Error handling wrapper
    └── tool_support.py         # Tool definition and testing
```