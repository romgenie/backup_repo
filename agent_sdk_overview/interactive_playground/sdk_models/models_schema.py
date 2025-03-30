"""
Models schema for SDK Playground.
Defines the available models as a schema instead of hardcoded entries.
"""
from typing import Dict, List, Optional, Set, Union, Literal, Any
from dataclasses import dataclass, field
from enum import Enum


class ModelFeature(str, Enum):
    """Features that a model might support."""
    TOOL_USE = "tool_use"
    VISION = "vision"
    PARALLEL_TOOL_CALLS = "parallel_tool_calls"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    STRUCTURED_OUTPUT = "structured_output"
    FUNCTION_CALLING = "function_calling"
    AGENTS_SDK = "agents_sdk"
    TOOL_CHOICE = "tool_choice"


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class ModelProviderRenderInfo:
    """Render information for a model provider."""
    def __init__(
        self,
        id: ModelProvider,
        name: str,
        description: str,
        url: str,
        requires_key: bool = True,
        requires_endpoint: bool = False,
        custom_base_url: bool = False,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.url = url
        self.requires_key = requires_key
        self.requires_endpoint = requires_endpoint
        self.custom_base_url = custom_base_url


# Provider display information
PROVIDER_INFO = {
    ModelProvider.OPENAI: ModelProviderRenderInfo(
        id=ModelProvider.OPENAI,
        name="OpenAI",
        description="OpenAI API models",
        url="https://platform.openai.com/",
        requires_key=True,
        custom_base_url=True,
    ),
    ModelProvider.AZURE_OPENAI: ModelProviderRenderInfo(
        id=ModelProvider.AZURE_OPENAI,
        name="Azure OpenAI",
        description="OpenAI models hosted on Azure",
        url="https://azure.microsoft.com/en-us/products/ai-services/openai-service",
        requires_key=True,
        requires_endpoint=True,
    ),
    ModelProvider.ANTHROPIC: ModelProviderRenderInfo(
        id=ModelProvider.ANTHROPIC,
        name="Anthropic",
        description="Anthropic Claude models",
        url="https://www.anthropic.com/",
        requires_key=True,
    ),
    ModelProvider.CUSTOM: ModelProviderRenderInfo(
        id=ModelProvider.CUSTOM,
        name="Custom Provider",
        description="Custom model provider with your own endpoint",
        url="",
        requires_key=True,
        requires_endpoint=True,
        custom_base_url=True,
    ),
}


@dataclass
class ModelCapability:
    """Details about a specific model capability."""
    feature: ModelFeature  
    supported: bool = False
    details: Optional[str] = None


@dataclass
class ModelSchema:
    """Complete schema of a model with all its properties."""
    id: str
    name: str
    provider: ModelProvider
    description: str
    capabilities: Dict[ModelFeature, ModelCapability] = field(default_factory=dict)
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
    # Fields specific to API schemas
    api_compatibilities: Set[str] = field(default_factory=set)
    supports_agents_sdk: bool = True
    supports_responses_api: bool = True
    
    def to_model_definition(self):
        """Convert schema to ModelDefinition for backward compatibility."""
        from .model_definitions import ModelDefinition, ModelCapability as OldModelCapability, PROVIDERS
        
        # Map capabilities to the old format
        capabilities_dict = {}
        for feature, capability in self.capabilities.items():
            capabilities_dict[feature.value] = OldModelCapability(
                name=feature.value.replace("_", " ").title(),
                description=get_feature_description(feature),
                supported=capability.supported,
                details=capability.details
            )
        
        return ModelDefinition(
            id=self.id,
            name=self.name,
            provider=PROVIDERS[self.provider],
            description=self.description,
            capabilities=capabilities_dict,
            is_available=self.is_available,
            max_tokens=self.max_tokens,
            supports_functions=self.supports_functions,
            supports_vision=self.supports_vision,
            example_prompt=self.example_prompt,
            default_temperature=self.default_temperature,
            default_top_p=self.default_top_p,
            context_window=self.context_window,
            token_limit=self.token_limit,
            version=self.version
        )


def get_feature_description(feature: ModelFeature) -> str:
    """Get standardized description for a model feature."""
    descriptions = {
        ModelFeature.TOOL_USE: "Can use external tools via function calling",
        ModelFeature.VISION: "Can process and understand images in the input",
        ModelFeature.PARALLEL_TOOL_CALLS: "Can call multiple tools in parallel",
        ModelFeature.STREAMING: "Supports streaming responses token by token",
        ModelFeature.JSON_MODE: "Can output well-formed JSON responses",
        ModelFeature.STRUCTURED_OUTPUT: "Can conform to specified output schemas",
        ModelFeature.FUNCTION_CALLING: "Supports function calling API",
        ModelFeature.AGENTS_SDK: "Compatible with Agents SDK",
        ModelFeature.TOOL_CHOICE: "Supports tool_choice parameter to force/require specific tools"
    }
    return descriptions.get(feature, "Unknown feature")


# Define model schemas for different providers
OPENAI_MODEL_SCHEMAS = [
    ModelSchema(
        id="gpt-4o",
        name="GPT-4o",
        provider=ModelProvider.OPENAI,
        description="OpenAI's latest and most capable multimodal model",
        capabilities={
            ModelFeature.TOOL_USE: ModelCapability(feature=ModelFeature.TOOL_USE, supported=True),
            ModelFeature.VISION: ModelCapability(feature=ModelFeature.VISION, supported=True),
            ModelFeature.PARALLEL_TOOL_CALLS: ModelCapability(feature=ModelFeature.PARALLEL_TOOL_CALLS, supported=True),
            ModelFeature.STREAMING: ModelCapability(feature=ModelFeature.STREAMING, supported=True),
            ModelFeature.JSON_MODE: ModelCapability(feature=ModelFeature.JSON_MODE, supported=True),
            ModelFeature.STRUCTURED_OUTPUT: ModelCapability(feature=ModelFeature.STRUCTURED_OUTPUT, supported=True),
            ModelFeature.FUNCTION_CALLING: ModelCapability(feature=ModelFeature.FUNCTION_CALLING, supported=True),
            ModelFeature.AGENTS_SDK: ModelCapability(feature=ModelFeature.AGENTS_SDK, supported=True),
            ModelFeature.TOOL_CHOICE: ModelCapability(feature=ModelFeature.TOOL_CHOICE, supported=True),
        },
        supports_functions=True,
        supports_vision=True,
        context_window=128000,
        token_limit=4096,
        example_prompt="Describe what makes this model unique in one sentence.",
        api_compatibilities={"chat", "assistants", "agents_sdk", "responses"},
    ),
    ModelSchema(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider=ModelProvider.OPENAI,
        description="Optimized version of GPT-4 with improved capabilities and lower latency",
        capabilities={
            ModelFeature.TOOL_USE: ModelCapability(feature=ModelFeature.TOOL_USE, supported=True),
            ModelFeature.VISION: ModelCapability(feature=ModelFeature.VISION, supported=True),
            ModelFeature.PARALLEL_TOOL_CALLS: ModelCapability(feature=ModelFeature.PARALLEL_TOOL_CALLS, supported=True),
            ModelFeature.STREAMING: ModelCapability(feature=ModelFeature.STREAMING, supported=True),
            ModelFeature.JSON_MODE: ModelCapability(feature=ModelFeature.JSON_MODE, supported=True),
            ModelFeature.STRUCTURED_OUTPUT: ModelCapability(feature=ModelFeature.STRUCTURED_OUTPUT, supported=True),
            ModelFeature.FUNCTION_CALLING: ModelCapability(feature=ModelFeature.FUNCTION_CALLING, supported=True),
            ModelFeature.AGENTS_SDK: ModelCapability(feature=ModelFeature.AGENTS_SDK, supported=True),
            ModelFeature.TOOL_CHOICE: ModelCapability(feature=ModelFeature.TOOL_CHOICE, supported=True),
        },
        supports_functions=True,
        supports_vision=True,
        context_window=128000,
        token_limit=4096,
        api_compatibilities={"chat", "assistants", "agents_sdk", "responses"},
    ),
    ModelSchema(
        id="gpt-4",
        name="GPT-4",
        provider=ModelProvider.OPENAI,
        description="OpenAI's original GPT-4 model with strong reasoning capabilities",
        capabilities={
            ModelFeature.TOOL_USE: ModelCapability(feature=ModelFeature.TOOL_USE, supported=True),
            ModelFeature.VISION: ModelCapability(feature=ModelFeature.VISION, supported=False),
            ModelFeature.PARALLEL_TOOL_CALLS: ModelCapability(feature=ModelFeature.PARALLEL_TOOL_CALLS, supported=False),
            ModelFeature.STREAMING: ModelCapability(feature=ModelFeature.STREAMING, supported=True),
            ModelFeature.JSON_MODE: ModelCapability(feature=ModelFeature.JSON_MODE, supported=True),
            ModelFeature.STRUCTURED_OUTPUT: ModelCapability(feature=ModelFeature.STRUCTURED_OUTPUT, supported=True),
            ModelFeature.FUNCTION_CALLING: ModelCapability(feature=ModelFeature.FUNCTION_CALLING, supported=True),
            ModelFeature.AGENTS_SDK: ModelCapability(feature=ModelFeature.AGENTS_SDK, supported=True),
            ModelFeature.TOOL_CHOICE: ModelCapability(feature=ModelFeature.TOOL_CHOICE, supported=True),
        },
        supports_functions=True,
        context_window=8192,
        token_limit=4096,
        api_compatibilities={"chat", "agents_sdk", "responses"},
    ),
    ModelSchema(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        provider=ModelProvider.OPENAI,
        description="Cost-effective model with good performance for many tasks",
        capabilities={
            ModelFeature.TOOL_USE: ModelCapability(feature=ModelFeature.TOOL_USE, supported=True),
            ModelFeature.VISION: ModelCapability(feature=ModelFeature.VISION, supported=False),
            ModelFeature.PARALLEL_TOOL_CALLS: ModelCapability(feature=ModelFeature.PARALLEL_TOOL_CALLS, supported=True),
            ModelFeature.STREAMING: ModelCapability(feature=ModelFeature.STREAMING, supported=True),
            ModelFeature.JSON_MODE: ModelCapability(feature=ModelFeature.JSON_MODE, supported=True),
            ModelFeature.STRUCTURED_OUTPUT: ModelCapability(feature=ModelFeature.STRUCTURED_OUTPUT, supported=True),
            ModelFeature.FUNCTION_CALLING: ModelCapability(feature=ModelFeature.FUNCTION_CALLING, supported=True),
            ModelFeature.AGENTS_SDK: ModelCapability(feature=ModelFeature.AGENTS_SDK, supported=True),
            ModelFeature.TOOL_CHOICE: ModelCapability(feature=ModelFeature.TOOL_CHOICE, supported=True),
        },
        supports_functions=True,
        context_window=16385,
        token_limit=4096,
        api_compatibilities={"chat", "assistants", "agents_sdk", "responses"},
    ),
]

ANTHROPIC_MODEL_SCHEMAS = [
    ModelSchema(
        id="claude-3-opus",
        name="Claude 3 Opus",
        provider=ModelProvider.ANTHROPIC,
        description="Anthropic's most powerful model with exceptional performance across tasks",
        capabilities={
            ModelFeature.TOOL_USE: ModelCapability(feature=ModelFeature.TOOL_USE, supported=True),
            ModelFeature.VISION: ModelCapability(feature=ModelFeature.VISION, supported=True),
            ModelFeature.PARALLEL_TOOL_CALLS: ModelCapability(
                feature=ModelFeature.PARALLEL_TOOL_CALLS, 
                supported=False,
                details="Uses sequential tool calls instead"
            ),
            ModelFeature.STREAMING: ModelCapability(feature=ModelFeature.STREAMING, supported=True),
            ModelFeature.JSON_MODE: ModelCapability(feature=ModelFeature.JSON_MODE, supported=True),
            ModelFeature.STRUCTURED_OUTPUT: ModelCapability(feature=ModelFeature.STRUCTURED_OUTPUT, supported=True),
            ModelFeature.FUNCTION_CALLING: ModelCapability(feature=ModelFeature.FUNCTION_CALLING, supported=True),
            ModelFeature.AGENTS_SDK: ModelCapability(feature=ModelFeature.AGENTS_SDK, supported=True),
            ModelFeature.TOOL_CHOICE: ModelCapability(
                feature=ModelFeature.TOOL_CHOICE, 
                supported=False,
                details="Tool choice is always automatic"
            ),
        },
        supports_functions=True,
        supports_vision=True,
        context_window=200000,
        token_limit=4096,
        api_compatibilities={"messages", "agents_sdk", "responses"},
    ),
    ModelSchema(
        id="claude-3-sonnet",
        name="Claude 3 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        description="Balanced model for most use cases with good performance",
        capabilities={
            ModelFeature.TOOL_USE: ModelCapability(feature=ModelFeature.TOOL_USE, supported=True),
            ModelFeature.VISION: ModelCapability(feature=ModelFeature.VISION, supported=True),
            ModelFeature.PARALLEL_TOOL_CALLS: ModelCapability(
                feature=ModelFeature.PARALLEL_TOOL_CALLS, 
                supported=False,
                details="Uses sequential tool calls instead"
            ),
            ModelFeature.STREAMING: ModelCapability(feature=ModelFeature.STREAMING, supported=True),
            ModelFeature.JSON_MODE: ModelCapability(feature=ModelFeature.JSON_MODE, supported=True),
            ModelFeature.STRUCTURED_OUTPUT: ModelCapability(feature=ModelFeature.STRUCTURED_OUTPUT, supported=True),
            ModelFeature.FUNCTION_CALLING: ModelCapability(feature=ModelFeature.FUNCTION_CALLING, supported=True),
            ModelFeature.AGENTS_SDK: ModelCapability(feature=ModelFeature.AGENTS_SDK, supported=True),
            ModelFeature.TOOL_CHOICE: ModelCapability(
                feature=ModelFeature.TOOL_CHOICE, 
                supported=False,
                details="Tool choice is always automatic"
            ),
        },
        supports_functions=True,
        supports_vision=True,
        context_window=200000,
        token_limit=4096,
        api_compatibilities={"messages", "agents_sdk", "responses"},
    ),
    ModelSchema(
        id="claude-3-haiku",
        name="Claude 3 Haiku",
        provider=ModelProvider.ANTHROPIC,
        description="Fast and efficient model for high-volume tasks",
        capabilities={
            ModelFeature.TOOL_USE: ModelCapability(feature=ModelFeature.TOOL_USE, supported=True),
            ModelFeature.VISION: ModelCapability(feature=ModelFeature.VISION, supported=True),
            ModelFeature.PARALLEL_TOOL_CALLS: ModelCapability(
                feature=ModelFeature.PARALLEL_TOOL_CALLS, 
                supported=False,
                details="Uses sequential tool calls instead"
            ),
            ModelFeature.STREAMING: ModelCapability(feature=ModelFeature.STREAMING, supported=True),
            ModelFeature.JSON_MODE: ModelCapability(feature=ModelFeature.JSON_MODE, supported=True),
            ModelFeature.STRUCTURED_OUTPUT: ModelCapability(feature=ModelFeature.STRUCTURED_OUTPUT, supported=True),
            ModelFeature.FUNCTION_CALLING: ModelCapability(feature=ModelFeature.FUNCTION_CALLING, supported=True),
            ModelFeature.AGENTS_SDK: ModelCapability(feature=ModelFeature.AGENTS_SDK, supported=True),
            ModelFeature.TOOL_CHOICE: ModelCapability(
                feature=ModelFeature.TOOL_CHOICE, 
                supported=False,
                details="Tool choice is always automatic"
            ),
        },
        supports_functions=True,
        supports_vision=True,
        context_window=200000,
        token_limit=4096,
        api_compatibilities={"messages", "agents_sdk", "responses"},
    ),
]

AZURE_MODEL_SCHEMAS = [
    ModelSchema(
        id="gpt-4-turbo",
        name="Azure GPT-4 Turbo",
        provider=ModelProvider.AZURE_OPENAI,
        description="OpenAI's GPT-4 Turbo model hosted on Azure",
        capabilities={
            ModelFeature.TOOL_USE: ModelCapability(feature=ModelFeature.TOOL_USE, supported=True),
            ModelFeature.VISION: ModelCapability(feature=ModelFeature.VISION, supported=True),
            ModelFeature.PARALLEL_TOOL_CALLS: ModelCapability(feature=ModelFeature.PARALLEL_TOOL_CALLS, supported=True),
            ModelFeature.STREAMING: ModelCapability(feature=ModelFeature.STREAMING, supported=True),
            ModelFeature.JSON_MODE: ModelCapability(feature=ModelFeature.JSON_MODE, supported=True),
            ModelFeature.STRUCTURED_OUTPUT: ModelCapability(feature=ModelFeature.STRUCTURED_OUTPUT, supported=True),
            ModelFeature.FUNCTION_CALLING: ModelCapability(feature=ModelFeature.FUNCTION_CALLING, supported=True),
            ModelFeature.AGENTS_SDK: ModelCapability(feature=ModelFeature.AGENTS_SDK, supported=True),
            ModelFeature.TOOL_CHOICE: ModelCapability(feature=ModelFeature.TOOL_CHOICE, supported=True),
        },
        supports_functions=True,
        supports_vision=True,
        context_window=128000,
        token_limit=4096,
        api_compatibilities={"chat", "assistants", "agents_sdk", "responses"},
    ),
    ModelSchema(
        id="gpt-4",
        name="Azure GPT-4",
        provider=ModelProvider.AZURE_OPENAI,
        description="OpenAI's GPT-4 model hosted on Azure",
        capabilities={
            ModelFeature.TOOL_USE: ModelCapability(feature=ModelFeature.TOOL_USE, supported=True),
            ModelFeature.VISION: ModelCapability(feature=ModelFeature.VISION, supported=False),
            ModelFeature.PARALLEL_TOOL_CALLS: ModelCapability(feature=ModelFeature.PARALLEL_TOOL_CALLS, supported=False),
            ModelFeature.STREAMING: ModelCapability(feature=ModelFeature.STREAMING, supported=True),
            ModelFeature.JSON_MODE: ModelCapability(feature=ModelFeature.JSON_MODE, supported=True),
            ModelFeature.STRUCTURED_OUTPUT: ModelCapability(feature=ModelFeature.STRUCTURED_OUTPUT, supported=True),
            ModelFeature.FUNCTION_CALLING: ModelCapability(feature=ModelFeature.FUNCTION_CALLING, supported=True),
            ModelFeature.AGENTS_SDK: ModelCapability(feature=ModelFeature.AGENTS_SDK, supported=True),
            ModelFeature.TOOL_CHOICE: ModelCapability(feature=ModelFeature.TOOL_CHOICE, supported=True),
        },
        supports_functions=True,
        context_window=8192,
        token_limit=4096,
        api_compatibilities={"chat", "agents_sdk", "responses"},
    ),
    ModelSchema(
        id="gpt-35-turbo",  # Note different ID format for Azure
        name="Azure GPT-3.5 Turbo",
        provider=ModelProvider.AZURE_OPENAI,
        description="OpenAI's GPT-3.5 Turbo model hosted on Azure",
        capabilities={
            ModelFeature.TOOL_USE: ModelCapability(feature=ModelFeature.TOOL_USE, supported=True),
            ModelFeature.VISION: ModelCapability(feature=ModelFeature.VISION, supported=False),
            ModelFeature.PARALLEL_TOOL_CALLS: ModelCapability(feature=ModelFeature.PARALLEL_TOOL_CALLS, supported=True),
            ModelFeature.STREAMING: ModelCapability(feature=ModelFeature.STREAMING, supported=True),
            ModelFeature.JSON_MODE: ModelCapability(feature=ModelFeature.JSON_MODE, supported=True),
            ModelFeature.STRUCTURED_OUTPUT: ModelCapability(feature=ModelFeature.STRUCTURED_OUTPUT, supported=True),
            ModelFeature.FUNCTION_CALLING: ModelCapability(feature=ModelFeature.FUNCTION_CALLING, supported=True),
            ModelFeature.AGENTS_SDK: ModelCapability(feature=ModelFeature.AGENTS_SDK, supported=True),
            ModelFeature.TOOL_CHOICE: ModelCapability(feature=ModelFeature.TOOL_CHOICE, supported=True),
        },
        supports_functions=True,
        context_window=16385,
        token_limit=4096,
        api_compatibilities={"chat", "agents_sdk", "responses"},
    ),
]

# Combine all model schemas
ALL_MODEL_SCHEMAS = OPENAI_MODEL_SCHEMAS + ANTHROPIC_MODEL_SCHEMAS + AZURE_MODEL_SCHEMAS

# Group model schemas by provider
MODEL_SCHEMAS_BY_PROVIDER = {
    ModelProvider.OPENAI: OPENAI_MODEL_SCHEMAS,
    ModelProvider.AZURE_OPENAI: AZURE_MODEL_SCHEMAS,
    ModelProvider.ANTHROPIC: ANTHROPIC_MODEL_SCHEMAS,
}

# Helper functions to work with model schemas
def get_model_schema_by_id(model_id: str) -> Optional[ModelSchema]:
    """Get a model schema by its ID."""
    for model in ALL_MODEL_SCHEMAS:
        if model.id == model_id:
            return model
    return None

def get_model_schemas_for_provider(provider_id: Union[str, ModelProvider]) -> List[ModelSchema]:
    """Get all model schemas for a specific provider."""
    if isinstance(provider_id, str):
        provider_id = ModelProvider(provider_id)
    return MODEL_SCHEMAS_BY_PROVIDER.get(provider_id, [])

def get_model_schemas_by_api_compatibility(api_type: str) -> List[ModelSchema]:
    """Get all model schemas compatible with a specific API type."""
    return [model for model in ALL_MODEL_SCHEMAS if api_type in model.api_compatibilities]

def get_model_schemas_for_agents_sdk() -> List[ModelSchema]:
    """Get all model schemas compatible with the Agents SDK."""
    return [model for model in ALL_MODEL_SCHEMAS if model.supports_agents_sdk]

def get_model_schemas_for_responses_api() -> List[ModelSchema]:
    """Get all model schemas compatible with the Responses API."""
    return [model for model in ALL_MODEL_SCHEMAS if model.supports_responses_api]