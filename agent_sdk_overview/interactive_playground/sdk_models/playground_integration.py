"""
Integration module to connect the model selector to the playground.
This module provides functions for using the selected model in the playground.
"""
import streamlit as st
from typing import Dict, Any, Optional, List

# Import from models_schema instead of model_definitions
from .models_schema import ModelSchema, ModelFeature
from .model_selector import model_selector
from .provider_config import ProviderConfig


def initialize_model_support():
    """Initialize model support in the session state."""
    if "model_support_initialized" not in st.session_state:
        # Load provider configs from environment
        ProviderConfig.load_from_env()
        st.session_state.model_support_initialized = True


def get_model_settings_ui(model: ModelSchema) -> Dict[str, Any]:
    """
    Render UI for model-specific settings based on its capabilities.
    Returns a dictionary of model settings.
    """
    settings = {}
    
    # Temperature and randomness settings
    temp_col, top_p_col = st.columns(2)
    with temp_col:
        settings["temperature"] = st.slider(
            "Temperature:", 
            min_value=0.0, 
            max_value=1.0, 
            value=model.default_temperature, 
            step=0.1
        )
    
    with top_p_col:
        settings["top_p"] = st.slider(
            "Top_p:", 
            min_value=0.0, 
            max_value=1.0, 
            value=model.default_top_p, 
            step=0.1
        )
    
    # Token limit settings
    token_col, presence_col, freq_col = st.columns(3)
    with token_col:
        settings["max_tokens"] = st.number_input(
            "Max tokens:", 
            min_value=1, 
            max_value=model.token_limit, 
            value=min(1000, model.token_limit),
            step=100
        )
    
    with presence_col:
        settings["presence_penalty"] = st.slider(
            "Presence penalty:", 
            min_value=0.0, 
            max_value=2.0, 
            value=0.0, 
            step=0.1
        )
    
    with freq_col:
        settings["frequency_penalty"] = st.slider(
            "Frequency penalty:", 
            min_value=0.0, 
            max_value=2.0, 
            value=0.0, 
            step=0.1
        )
    
    # Model-specific capabilities settings
    if ModelFeature.JSON_MODE in model.capabilities and model.capabilities[ModelFeature.JSON_MODE].supported:
        settings["response_format"] = st.selectbox(
            "Response Format:",
            options=["auto", "json_object"],
            format_func=lambda x: "JSON" if x == "json_object" else "Auto",
        )
    
    settings["store"] = st.checkbox("Store response", True)
    
    return settings


def get_tool_settings_ui(model: ModelSchema) -> Dict[str, Any]:
    """Render UI for tool-related settings based on model capabilities."""
    tool_settings = {}
    
    # Only show tool settings if the model supports tools
    if model.supports_functions:
        st.subheader("Tools")
        
        use_tools = st.checkbox("Enable tools")
        tool_settings["use_tools"] = use_tools
        
        if use_tools:
            # Only show tool choice if the model supports it
            if ModelFeature.TOOL_CHOICE in model.capabilities and model.capabilities[ModelFeature.TOOL_CHOICE].supported:
                tool_choice = st.selectbox(
                    "Tool choice:",
                    options=["auto", "required", "none"]
                )
                tool_settings["tool_choice"] = tool_choice
            else:
                tool_settings["tool_choice"] = "auto"
            
            # Only show parallel tool calls option if the model supports it
            if (ModelFeature.PARALLEL_TOOL_CALLS in model.capabilities and 
                model.capabilities[ModelFeature.PARALLEL_TOOL_CALLS].supported):
                parallel_tool_calls = st.checkbox("Enable parallel tool calls")
                tool_settings["parallel_tool_calls"] = parallel_tool_calls
    
    return tool_settings


def generate_model_code_preview(
    model: ModelSchema,
    provider_config: Dict,
    model_settings: Dict,
    tool_settings: Dict,
    system_message: str,
    user_message: str
) -> str:
    """
    Generate code preview for using the selected model with the SDK.
    Returns Python code as a string.
    """
    # Get the provider type
    provider_type = model.provider
    
    # Handle different provider types
    if provider_type == "openai":
        client_creation = f"""
# Create OpenAI client
from openai import OpenAI
client = OpenAI(api_key="{provider_config.get('api_key', 'YOUR_API_KEY')}")
"""
        if provider_config.get("base_url"):
            client_creation = f"""
# Create OpenAI client with custom base URL
from openai import OpenAI
client = OpenAI(
    api_key="{provider_config.get('api_key', 'YOUR_API_KEY')}",
    base_url="{provider_config.get('base_url')}"
)
"""
    elif provider_type == "azure_openai":
        client_creation = f"""
# Create Azure OpenAI client
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key="{provider_config.get('api_key', 'YOUR_API_KEY')}",
    azure_endpoint="{provider_config.get('endpoint', 'YOUR_ENDPOINT')}",
    api_version="{provider_config.get('api_version', '2023-12-01-preview')}"
)
"""
    elif provider_type == "anthropic":
        client_creation = f"""
# Create Anthropic client
from anthropic import Anthropic
client = Anthropic(api_key="{provider_config.get('api_key', 'YOUR_API_KEY')}")
"""
    else:
        # Generic custom provider
        client_creation = f"""
# Create custom client - replace with appropriate client for your provider
client = YourProviderClient(
    api_key="{provider_config.get('api_key', 'YOUR_API_KEY')}",
    base_url="{provider_config.get('base_url', 'YOUR_BASE_URL')}"
)
"""
    
    # Construct model settings code
    settings_args = []
    for key, value in model_settings.items():
        if isinstance(value, str):
            settings_args.append(f"{key}='{value}'")
        else:
            settings_args.append(f"{key}={value}")
    
    # Add tool settings
    if tool_settings.get("use_tools", False):
        if tool_settings.get("tool_choice") != "auto":
            settings_args.append(f"tool_choice='{tool_settings.get('tool_choice')}'")
        if tool_settings.get("parallel_tool_calls", False):
            settings_args.append(f"parallel_tool_calls={tool_settings.get('parallel_tool_calls')}")
    
    settings_str = "ModelSettings()" if not settings_args else f"ModelSettings(\n    {',\\n    '.join(settings_args)}\n)"
    
    # Generate sample code
    code = f"""import asyncio
from agents.models.{provider_type.lower()}_provider import {provider_type.capitalize()}Provider
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing

{client_creation}

async def main():
    # Initialize provider with client
    provider = {provider_type.capitalize()}Provider(client=client)
    
    # Get the model
    model = provider.get_model("{model.id}")
    
    # Define model settings
    settings = {settings_str}
    
    # Call the model
    response = await model.get_response(
        system_instructions="{system_message}",
        input="{user_message}",
        model_settings=settings,
        tools=[],  # Add your tools here if using tools
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED
    )
    
    # Process response
    print(response.output)
    print(f"Usage: {{response.usage}}")

asyncio.run(main())
"""
    return code


def generate_api_call_preview(
    model: ModelSchema,
    provider_config: Dict,
    model_settings: Dict,
    tool_settings: Dict,
    system_message: str,
    user_message: str
) -> Dict:
    """
    Generate a preview of the API call that would be made.
    Returns a dictionary representing the API call.
    """
    provider_type = model.provider
    
    if provider_type == "openai" or provider_type == "azure_openai":
        # OpenAI-style API call
        api_call = {
            "model": model.id,
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }
        
        # Add model settings
        for key, value in model_settings.items():
            if key != "store" or not value:  # Only add store if it's False
                api_call[key] = value
        
        # Add tool settings
        if tool_settings.get("use_tools", False):
            if tool_settings.get("tool_choice") != "auto":
                api_call["tool_choice"] = tool_settings.get("tool_choice")
            if tool_settings.get("parallel_tool_calls", False):
                api_call["parallel_tool_calls"] = tool_settings.get("parallel_tool_calls")
        
        return api_call
    
    elif provider_type == "anthropic":
        # Anthropic-style API call
        api_call = {
            "model": model.id,
            "system": system_message,
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "max_tokens": model_settings.get("max_tokens", 1000),
        }
        
        # Add temperature
        if "temperature" in model_settings:
            api_call["temperature"] = model_settings["temperature"]
        
        # Anthropic doesn't support all the same parameters
        if "top_p" in model_settings:
            api_call["top_p"] = model_settings["top_p"]
        
        return api_call
    
    else:
        # Generic API call format
        return {
            "model": model.id,
            "provider": provider_type,
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "settings": model_settings,
            "tool_settings": tool_settings if tool_settings.get("use_tools", False) else None
        }