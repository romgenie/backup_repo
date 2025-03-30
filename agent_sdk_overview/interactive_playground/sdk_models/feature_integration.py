"""
Integration module for connecting enhanced SDK features to the playground.
This module provides functions for using the advanced features in the playground.
"""
import streamlit as st
import json
from typing import Dict, Any, Optional, List, Union

# Update imports to use models_schema
from .models_schema import ModelSchema
from .output_schemas import schema_support_ui, SchemaDefinition
from .tool_support import tool_support_ui, ToolDefinition
from .response_types import response_types_ui, generate_reasoning_prompt


def initialize_enhanced_features():
    """Initialize all enhanced features in the session state."""
    # Ensure session state is initialized for all features
    if "enhanced_features_initialized" not in st.session_state:
        # Initialize schemas
        if "output_schema" not in st.session_state:
            st.session_state.output_schema = None
        
        # Initialize tools
        if "tools" not in st.session_state:
            st.session_state.tools = []
        
        # Initialize response configuration
        if "response_config" not in st.session_state:
            st.session_state.response_config = {"type": "default", "config": {}}
        
        # Initialize reasoning settings
        if "reasoning_settings" not in st.session_state:
            st.session_state.reasoning_settings = {
                "reasoning_depth": "standard",
                "think_step_by_step": True,
                "show_work": True,
                "use_numbered_steps": True,
                "max_reasoning_steps": 5,
                "evidence_threshold": "moderate"
            }
        
        # Initialize reasoning instruction
        if "reasoning_instruction" not in st.session_state:
            st.session_state.reasoning_instruction = ""
        
        # Mark initialization as complete
        st.session_state.enhanced_features_initialized = True


def enhanced_features_selector():
    """UI for selecting and configuring enhanced features."""
    st.header("Enhanced Features")
    
    # Initialize features
    initialize_enhanced_features()
    
    feature_tabs = st.tabs([
        "Output Schemas", 
        "Tool Support", 
        "Response Types"
    ])
    
    with feature_tabs[0]:
        schema_support_ui()
    
    with feature_tabs[1]:
        tool_support_ui()
    
    with feature_tabs[2]:
        response_types_ui()


def get_enhanced_system_instruction(base_instruction: str) -> str:
    """
    Enhance the system instruction with reasoning and response format guidance.
    
    Args:
        base_instruction: The original system instruction
        
    Returns:
        Enhanced system instruction
    """
    enhanced_instruction = base_instruction
    
    # Add reasoning instruction if available
    if "reasoning_instruction" in st.session_state and st.session_state.reasoning_instruction:
        enhanced_instruction += f"\n\n{st.session_state.reasoning_instruction}"
    
    # Add response type formatting instruction
    if "response_config" in st.session_state:
        response_config = st.session_state.response_config
        response_type = response_config.get("type", "default")
        
        if response_type == "json_mode":
            format_type = response_config.get("config", {}).get("format", "object")
            enhanced_instruction += f"\n\nProvide your response as a valid JSON {format_type}."
        
        elif response_type == "reasoning":
            verbosity = response_config.get("config", {}).get("verbosity", "detailed")
            format_type = response_config.get("config", {}).get("format", "numbered")
            
            format_instruction = ""
            if format_type == "numbered":
                format_instruction = "in a numbered list"
            elif format_type == "bulleted":
                format_instruction = "with bullet points"
            elif format_type == "paragraph":
                format_instruction = "in paragraph form"
            
            enhanced_instruction += f"\n\nProvide {verbosity} reasoning {format_instruction}."
        
        elif response_type == "analytical":
            criteria = response_config.get("config", {}).get("evaluation_criteria", [])
            include_uncertainty = response_config.get("config", {}).get("include_uncertainty", True)
            
            if criteria:
                criteria_str = ", ".join(criteria)
                enhanced_instruction += f"\n\nProvide an analytical assessment considering: {criteria_str}."
            
            if include_uncertainty:
                enhanced_instruction += "\n\nAcknowledge areas of uncertainty where appropriate."
        
        elif response_type == "instructional":
            detail_level = response_config.get("config", {}).get("detail_level", "intermediate")
            include_warnings = response_config.get("config", {}).get("include_warnings", True)
            
            enhanced_instruction += f"\n\nProvide {detail_level}-level step-by-step instructions."
            
            if include_warnings:
                enhanced_instruction += " Include relevant warnings and cautions."
        
        elif response_type == "creative":
            style = response_config.get("config", {}).get("style", "narrative")
            tone = response_config.get("config", {}).get("tone", "casual")
            
            enhanced_instruction += f"\n\nProvide a {tone} response in a {style} style."
    
    return enhanced_instruction


def get_enhanced_model_settings(base_settings: Dict) -> Dict:
    """
    Enhance model settings with additional configuration based on features.
    
    Args:
        base_settings: Basic model settings
        
    Returns:
        Enhanced model settings
    """
    enhanced_settings = base_settings.copy()
    
    # Add response format settings
    if "response_config" in st.session_state:
        response_config = st.session_state.response_config
        response_type = response_config.get("type", "default")
        
        if response_type == "json_mode":
            enhanced_settings["response_format"] = {"type": "json_object"}
    
    return enhanced_settings


def get_tools_for_model() -> List:
    """
    Get configured tools for use with the model.
    
    Returns:
        List of tools in SDK format
    """
    tool_list = []
    
    if "tools" in st.session_state and st.session_state.tools:
        for tool_def in st.session_state.tools:
            tool_list.append(tool_def.to_function_tool())
    
    return tool_list


def get_output_schema() -> Optional[Dict]:
    """
    Get configured output schema for use with the model.
    
    Returns:
        Output schema or None
    """
    if "output_schema" in st.session_state and st.session_state.output_schema:
        return st.session_state.output_schema
    
    return None


def generate_enhanced_api_preview(
    model: ModelSchema,
    provider_config: Dict,
    model_settings: Dict,
    tool_settings: Dict,
    system_message: str,
    user_message: str
) -> Dict:
    """
    Generate a preview of the API call with enhanced features.
    
    Args:
        model: The model schema
        provider_config: Provider configuration
        model_settings: Model settings
        tool_settings: Tool settings
        system_message: System message
        user_message: User message
        
    Returns:
        API call preview
    """
    # Start with base API call
    api_call = {
        "model": model.id,
        "messages": [
            {
                "role": "system",
                "content": get_enhanced_system_instruction(system_message)
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    }
    
    # Add enhanced model settings
    enhanced_settings = get_enhanced_model_settings(model_settings)
    for key, value in enhanced_settings.items():
        api_call[key] = value
    
    # Add tools if enabled
    if tool_settings.get("use_tools", False):
        tools = get_tools_for_model()
        if tools:
            api_call["tools"] = tools
            
            if tool_settings.get("tool_choice") != "auto":
                api_call["tool_choice"] = tool_settings.get("tool_choice")
            
            if tool_settings.get("parallel_tool_calls", False):
                api_call["parallel_tool_calls"] = tool_settings.get("parallel_tool_calls")
    
    # Add output schema if available
    output_schema = get_output_schema()
    if output_schema:
        api_call["response_format"] = {
            "type": "json_object",
            "schema": output_schema
        }
    
    return api_call


def generate_enhanced_code_preview(
    model: ModelSchema,
    provider_config: Dict,
    model_settings: Dict,
    tool_settings: Dict,
    system_message: str,
    user_message: str
) -> str:
    """
    Generate SDK code with enhanced features.
    
    Args:
        model: The model schema
        provider_config: Provider configuration
        model_settings: Model settings
        tool_settings: Tool settings
        system_message: System message
        user_message: User message
        
    Returns:
        Python code preview
    """
    provider_type = model.provider
    
    # Generate client creation code
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
    else:
        # Generic client creation
        client_creation = f"""
# Create client
client = YourProviderClient(api_key="{provider_config.get('api_key', 'YOUR_API_KEY')}")
"""
    
    # Construct enhanced settings
    enhanced_settings = get_enhanced_model_settings(model_settings)
    settings_items = []
    
    for key, value in enhanced_settings.items():
        if isinstance(value, str):
            settings_items.append(f"{key}='{value}'")
        elif isinstance(value, dict):
            settings_items.append(f"{key}={value}")
        else:
            settings_items.append(f"{key}={value}")
    
    settings_str = "ModelSettings()" if not settings_items else f"ModelSettings(\n    {',\\n    '.join(settings_items)}\n)"
    
    # Tool setup code
    tools_code = ""
    tool_list = "[]"
    
    if "tools" in st.session_state and st.session_state.tools and tool_settings.get("use_tools", False):
        tools_code = "# Define tools\ntools = []\n"
        
        for i, tool_def in enumerate(st.session_state.tools):
            tool_schema = json.dumps(tool_def.schema, indent=4)
            
            tools_code += f"""
# {tool_def.name} tool
{tool_def.name}_tool = FunctionTool(
    name="{tool_def.name}",
    description="{tool_def.description}",
    params_json_schema={tool_schema},
    func=lambda params: {{
        # Implementation would go here
        return {{"result": "Implementation result"}}
    }}
)
tools.append({tool_def.name}_tool)
"""
        
        tool_list = "tools"
    
    # Output schema code
    schema_code = ""
    output_schema_var = "None"
    
    if "output_schema" in st.session_state and st.session_state.output_schema:
        schema_str = json.dumps(st.session_state.output_schema, indent=4)
        schema_code = f"""
# Define output schema
output_schema = {schema_str}
"""
        output_schema_var = "output_schema"
    
    # Enhanced system instruction
    enhanced_system = get_enhanced_system_instruction(system_message)
    
    # Generate the full code
    code = f"""import asyncio
from agents.models.{provider_type.lower()}_provider import {provider_type.capitalize()}Provider
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.tool import FunctionTool

{client_creation}
{tools_code}
{schema_code}
async def main():
    # Initialize provider with client
    provider = {provider_type.capitalize()}Provider(client=client)
    
    # Get the model
    model = provider.get_model("{model.id}")
    
    # Define enhanced system instruction
    system_instruction = \"\"\"
{enhanced_system}
\"\"\"
    
    # Define model settings
    settings = {settings_str}
    
    # Call the model
    response = await model.get_response(
        system_instructions=system_instruction,
        input="{user_message}",
        model_settings=settings,
        tools={tool_list},
        output_schema={output_schema_var},
        handoffs=[],
        tracing=ModelTracing.DISABLED
    )
    
    # Process response
    print(response.output)
    print(f"Usage: {{response.usage}}")

asyncio.run(main())
"""
    return code