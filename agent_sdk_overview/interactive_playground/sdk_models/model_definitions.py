"""
Model definitions and capabilities for the SDK Playground.
This module defines available models and their capabilities.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelCapability:
    """Represents a capability of a model."""
    name: str
    description: str
    supported: bool = False
    details: Optional[str] = None


@dataclass
class ModelProvider:
    """Represents a model provider."""
    id: str
    name: str
    description: str
    url: str
    requires_key: bool = True
    requires_endpoint: bool = False
    custom_base_url: bool = False


@dataclass
class ModelDefinition:
    """Complete definition of a model with capabilities."""
    id: str
    name: str
    provider: ModelProvider
    description: str
    capabilities: Dict[str, ModelCapability]
    is_available: bool = True
    max_tokens: int = 4096
    supports_functions: bool = False
    supports_vision: bool = False
    example_prompt: Optional[str] = None
    default_temperature: float = 0.7
    default_top_p: float = 1.0
    context_window: int = 8192
    token_limit: int = 4096
    version: Optional[str] = None


# Define standard capabilities that models may have
def get_standard_capabilities(all_disabled=False) -> Dict[str, ModelCapability]:
    """Get standard set of capabilities, optionally all disabled."""
    supported = not all_disabled
    return {
        "tool_use": ModelCapability(
            name="Tool Use",
            description="Can use external tools via function calling",
            supported=supported,
        ),
        "vision": ModelCapability(
            name="Vision",
            description="Can process and understand images in the input",
            supported=False,
        ),
        "parallel_tool_calls": ModelCapability(
            name="Parallel Tool Calls",
            description="Can call multiple tools in parallel",
            supported=supported,
        ),
        "streaming": ModelCapability(
            name="Streaming",
            description="Supports streaming responses token by token",
            supported=supported,
        ),
        "json_mode": ModelCapability(
            name="JSON Mode",
            description="Can output well-formed JSON responses",
            supported=supported,
        ),
        "structured_output": ModelCapability(
            name="Structured Output",
            description="Can conform to specified output schemas",
            supported=supported,
        ),
    }


# Define supported model providers
PROVIDERS = {
    "openai": ModelProvider(
        id="openai",
        name="OpenAI",
        description="OpenAI API models",
        url="https://platform.openai.com/",
        requires_key=True,
        custom_base_url=True,
    ),
    "azure_openai": ModelProvider(
        id="azure_openai",
        name="Azure OpenAI",
        description="OpenAI models hosted on Azure",
        url="https://azure.microsoft.com/en-us/products/ai-services/openai-service",
        requires_key=True,
        requires_endpoint=True,
    ),
    "anthropic": ModelProvider(
        id="anthropic",
        name="Anthropic",
        description="Anthropic Claude models",
        url="https://www.anthropic.com/",
        requires_key=True,
    ),
    "custom": ModelProvider(
        id="custom",
        name="Custom Provider",
        description="Custom model provider with your own endpoint",
        url="",
        requires_key=True,
        requires_endpoint=True,
        custom_base_url=True,
    ),
}


# Define the available models
OPENAI_MODELS = [
    ModelDefinition(
        id="gpt-4o",
        name="GPT-4o",
        provider=PROVIDERS["openai"],
        description="OpenAI's latest and most capable multimodal model",
        capabilities={
            "tool_use": ModelCapability(
                name="Tool Use",
                description="Can use external tools via function calling",
                supported=True,
            ),
            "vision": ModelCapability(
                name="Vision",
                description="Can process and understand images in the input",
                supported=True,
            ),
            "parallel_tool_calls": ModelCapability(
                name="Parallel Tool Calls",
                description="Can call multiple tools in parallel",
                supported=True,
            ),
            "streaming": ModelCapability(
                name="Streaming",
                description="Supports streaming responses token by token",
                supported=True,
            ),
            "json_mode": ModelCapability(
                name="JSON Mode",
                description="Can output well-formed JSON responses",
                supported=True,
            ),
            "structured_output": ModelCapability(
                name="Structured Output",
                description="Can conform to specified output schemas",
                supported=True,
            ),
        },
        supports_functions=True,
        supports_vision=True,
        context_window=128000,
        token_limit=4096,
        example_prompt="Describe what makes this model unique in one sentence.",
    ),
    ModelDefinition(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider=PROVIDERS["openai"],
        description="Optimized version of GPT-4 with improved capabilities and lower latency",
        capabilities={
            "tool_use": ModelCapability(
                name="Tool Use",
                description="Can use external tools via function calling",
                supported=True,
            ),
            "vision": ModelCapability(
                name="Vision",
                description="Can process and understand images in the input",
                supported=True,
            ),
            "parallel_tool_calls": ModelCapability(
                name="Parallel Tool Calls",
                description="Can call multiple tools in parallel",
                supported=True,
            ),
            "streaming": ModelCapability(
                name="Streaming",
                description="Supports streaming responses token by token",
                supported=True,
            ),
            "json_mode": ModelCapability(
                name="JSON Mode",
                description="Can output well-formed JSON responses",
                supported=True,
            ),
            "structured_output": ModelCapability(
                name="Structured Output",
                description="Can conform to specified output schemas",
                supported=True,
            ),
        },
        supports_functions=True,
        supports_vision=True,
        context_window=128000,
        token_limit=4096,
    ),
    ModelDefinition(
        id="gpt-4",
        name="GPT-4",
        provider=PROVIDERS["openai"],
        description="OpenAI's original GPT-4 model with strong reasoning capabilities",
        capabilities={
            "tool_use": ModelCapability(
                name="Tool Use",
                description="Can use external tools via function calling",
                supported=True,
            ),
            "vision": ModelCapability(
                name="Vision",
                description="Can process and understand images in the input",
                supported=False,
            ),
            "parallel_tool_calls": ModelCapability(
                name="Parallel Tool Calls",
                description="Can call multiple tools in parallel",
                supported=False,
            ),
            "streaming": ModelCapability(
                name="Streaming",
                description="Supports streaming responses token by token",
                supported=True,
            ),
            "json_mode": ModelCapability(
                name="JSON Mode",
                description="Can output well-formed JSON responses",
                supported=True,
            ),
            "structured_output": ModelCapability(
                name="Structured Output",
                description="Can conform to specified output schemas",
                supported=True,
            ),
        },
        supports_functions=True,
        context_window=8192,
        token_limit=4096,
    ),
    ModelDefinition(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        provider=PROVIDERS["openai"],
        description="Cost-effective model with good performance for many tasks",
        capabilities={
            "tool_use": ModelCapability(
                name="Tool Use",
                description="Can use external tools via function calling",
                supported=True,
            ),
            "vision": ModelCapability(
                name="Vision",
                description="Can process and understand images in the input",
                supported=False,
            ),
            "parallel_tool_calls": ModelCapability(
                name="Parallel Tool Calls",
                description="Can call multiple tools in parallel",
                supported=True,
            ),
            "streaming": ModelCapability(
                name="Streaming",
                description="Supports streaming responses token by token",
                supported=True,
            ),
            "json_mode": ModelCapability(
                name="JSON Mode",
                description="Can output well-formed JSON responses",
                supported=True,
            ),
            "structured_output": ModelCapability(
                name="Structured Output",
                description="Can conform to specified output schemas",
                supported=True,
            ),
        },
        supports_functions=True,
        context_window=16385,
        token_limit=4096,
    ),
]

ANTHROPIC_MODELS = [
    ModelDefinition(
        id="claude-3-opus",
        name="Claude 3 Opus",
        provider=PROVIDERS["anthropic"],
        description="Anthropic's most powerful model with exceptional performance across tasks",
        capabilities={
            "tool_use": ModelCapability(
                name="Tool Use",
                description="Can use external tools via function calling",
                supported=True,
            ),
            "vision": ModelCapability(
                name="Vision",
                description="Can process and understand images in the input",
                supported=True,
            ),
            "parallel_tool_calls": ModelCapability(
                name="Parallel Tool Calls",
                description="Can call multiple tools in parallel",
                supported=False,
                details="Uses sequential tool calls instead",
            ),
            "streaming": ModelCapability(
                name="Streaming",
                description="Supports streaming responses token by token",
                supported=True,
            ),
            "json_mode": ModelCapability(
                name="JSON Mode",
                description="Can output well-formed JSON responses",
                supported=True,
            ),
            "structured_output": ModelCapability(
                name="Structured Output",
                description="Can conform to specified output schemas",
                supported=True,
            ),
        },
        supports_functions=True,
        supports_vision=True,
        context_window=200000,
        token_limit=4096,
    ),
    ModelDefinition(
        id="claude-3-sonnet",
        name="Claude 3 Sonnet",
        provider=PROVIDERS["anthropic"],
        description="Balanced model for most use cases with good performance",
        capabilities={
            "tool_use": ModelCapability(
                name="Tool Use",
                description="Can use external tools via function calling",
                supported=True,
            ),
            "vision": ModelCapability(
                name="Vision",
                description="Can process and understand images in the input",
                supported=True,
            ),
            "parallel_tool_calls": ModelCapability(
                name="Parallel Tool Calls",
                description="Can call multiple tools in parallel",
                supported=False,
                details="Uses sequential tool calls instead",
            ),
            "streaming": ModelCapability(
                name="Streaming",
                description="Supports streaming responses token by token",
                supported=True,
            ),
            "json_mode": ModelCapability(
                name="JSON Mode",
                description="Can output well-formed JSON responses",
                supported=True,
            ),
            "structured_output": ModelCapability(
                name="Structured Output",
                description="Can conform to specified output schemas",
                supported=True,
            ),
        },
        supports_functions=True,
        supports_vision=True,
        context_window=200000,
        token_limit=4096,
    ),
    ModelDefinition(
        id="claude-3-haiku",
        name="Claude 3 Haiku",
        provider=PROVIDERS["anthropic"],
        description="Fast and efficient model for high-volume tasks",
        capabilities={
            "tool_use": ModelCapability(
                name="Tool Use",
                description="Can use external tools via function calling",
                supported=True,
            ),
            "vision": ModelCapability(
                name="Vision",
                description="Can process and understand images in the input",
                supported=True,
            ),
            "parallel_tool_calls": ModelCapability(
                name="Parallel Tool Calls",
                description="Can call multiple tools in parallel",
                supported=False,
                details="Uses sequential tool calls instead",
            ),
            "streaming": ModelCapability(
                name="Streaming",
                description="Supports streaming responses token by token",
                supported=True,
            ),
            "json_mode": ModelCapability(
                name="JSON Mode",
                description="Can output well-formed JSON responses",
                supported=True,
            ),
            "structured_output": ModelCapability(
                name="Structured Output",
                description="Can conform to specified output schemas",
                supported=True,
            ),
        },
        supports_functions=True,
        supports_vision=True,
        context_window=200000,
        token_limit=4096,
    ),
]

AZURE_MODELS = [
    ModelDefinition(
        id="gpt-4-turbo",
        name="Azure GPT-4 Turbo",
        provider=PROVIDERS["azure_openai"],
        description="OpenAI's GPT-4 Turbo model hosted on Azure",
        capabilities=get_standard_capabilities(),
        supports_functions=True,
        supports_vision=True,
        context_window=128000,
        token_limit=4096,
    ),
    ModelDefinition(
        id="gpt-4",
        name="Azure GPT-4",
        provider=PROVIDERS["azure_openai"],
        description="OpenAI's GPT-4 model hosted on Azure",
        capabilities=get_standard_capabilities(),
        # Update vision capability to False
        supports_functions=True,
        context_window=8192,
        token_limit=4096,
    ),
    ModelDefinition(
        id="gpt-35-turbo",  # Note different ID format for Azure
        name="Azure GPT-3.5 Turbo",
        provider=PROVIDERS["azure_openai"],
        description="OpenAI's GPT-3.5 Turbo model hosted on Azure",
        capabilities=get_standard_capabilities(),
        supports_functions=True,
        context_window=16385,
        token_limit=4096,
    ),
]

# Combine all models
ALL_MODELS = OPENAI_MODELS + ANTHROPIC_MODELS + AZURE_MODELS

# Group models by provider
MODELS_BY_PROVIDER = {
    "openai": OPENAI_MODELS,
    "azure_openai": AZURE_MODELS,
    "anthropic": ANTHROPIC_MODELS,
}


def get_model_by_id(model_id: str) -> Optional[ModelDefinition]:
    """Get a model definition by its ID."""
    for model in ALL_MODELS:
        if model.id == model_id:
            return model
    return None


def get_models_for_provider(provider_id: str) -> List[ModelDefinition]:
    """Get all models for a specific provider."""
    return MODELS_BY_PROVIDER.get(provider_id, [])