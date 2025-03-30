# SDK Models Module

This module provides enhanced model support for the SDK Playground, including:

1. **Comprehensive Model Catalog**: Support for OpenAI, Azure OpenAI, and Anthropic models
2. **Provider Configuration**: Manage API keys and endpoints for different providers
3. **Model Capability Display**: Show which features each model supports (vision, tool use, etc.)
4. **Custom Provider Support**: Allow specifying custom endpoints for API calls

## Components

The module consists of the following components:

### 1. Model Definitions (`model_definitions.py`)

- `ModelCapability`: Represents a specific capability of a model (e.g., tool use, vision)
- `ModelProvider`: Represents a model provider (e.g., OpenAI, Azure)
- `ModelDefinition`: Complete definition of a model with its capabilities
- Predefined models grouped by provider (OpenAI, Azure, Anthropic)

### 2. Model Selector (`model_selector.py`)

- UI component for selecting models with provider filtering
- Provider configuration UI
- Model capability visualization

### 3. Provider Configuration (`provider_config.py`)

- Manager for model provider configurations
- Load config from environment variables
- Validate configuration requirements
- Create appropriate provider clients

### 4. Playground Integration (`playground_integration.py`)

- Functions to integrate models with the playground
- Generate code previews based on selected model
- Generate API call previews

## Usage

```python
from interactive_playground.sdk_models.model_selector import model_selector
from interactive_playground.sdk_models.provider_config import ProviderConfig
from interactive_playground.sdk_models.playground_integration import (
    initialize_model_support,
    get_model_settings_ui,
    get_tool_settings_ui,
    generate_model_code_preview,
    generate_api_call_preview
)

# Initialize model support
initialize_model_support()

# Use the model selector
selected_model, provider_config = model_selector()

# Get model-specific settings
model_settings = get_model_settings_ui(selected_model)

# Get tool settings
tool_settings = get_tool_settings_ui(selected_model)

# Generate previews
code_preview = generate_model_code_preview(
    selected_model,
    provider_config,
    model_settings,
    tool_settings,
    system_message,
    user_message
)

api_call = generate_api_call_preview(
    selected_model,
    provider_config,
    model_settings,
    tool_settings,
    system_message,
    user_message
)
```

## Environment Variables

The module supports loading configuration from environment variables:

- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_BASE_URL`: Optional custom base URL for OpenAI
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint
- `AZURE_OPENAI_API_VERSION`: Azure OpenAI API version
- `ANTHROPIC_API_KEY`: Anthropic API key

## Extending

To add support for new models or providers:

1. Update `model_definitions.py` with new provider and model definitions
2. Update `provider_config.py` to handle configuration for the new provider
3. Update `playground_integration.py` to generate correct code and API previews for the new provider