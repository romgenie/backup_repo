"""
SDK Models package for interactive playground.
Provides model schema definitions and selection components.
"""

# Import from models_schema for new code
from .models_schema import (
    ModelSchema,
    ModelProvider,
    ModelCapability,
    PROVIDER_INFO,
    MODEL_SCHEMAS_BY_PROVIDER,
    ALL_MODEL_SCHEMAS,
    get_model_schema_by_id,
    get_model_schemas_for_provider,
    get_model_schemas_by_api_compatibility,
    get_model_schemas_for_agents_sdk,
    get_model_schemas_for_responses_api
)

# Import from model_definitions for backward compatibility
from .model_definitions import (
    ModelDefinition,
    ModelProvider as OldModelProvider,
    ModelCapability as OldModelCapability,
    PROVIDERS,
    ALL_MODELS,
    MODELS_BY_PROVIDER,
    get_model_by_id,
    get_models_for_provider
)

# Provide compatibility functions
def convert_model_schema_to_definition(model_schema: ModelSchema) -> ModelDefinition:
    """Convert a ModelSchema to the old ModelDefinition format for backward compatibility."""
    return model_schema.to_model_definition()

def get_compatible_models():
    """Get models compatible with Agents SDK and Responses API."""
    return [
        convert_model_schema_to_definition(model) 
        for model in ALL_MODEL_SCHEMAS 
        if model.supports_agents_sdk and model.supports_responses_api
    ]