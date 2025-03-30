import streamlit as st

def api_config_module():
    st.header("API Configuration")
    
    # Initialize session state
    if 'openai_api' not in st.session_state:
        st.session_state.openai_api = "responses"
    if 'api_version' not in st.session_state:
        st.session_state.api_version = ""
    if 'use_custom_provider' not in st.session_state:
        st.session_state.use_custom_provider = False
    if 'custom_provider_name' not in st.session_state:
        st.session_state.custom_provider_name = ""
    if 'custom_provider_base_url' not in st.session_state:
        st.session_state.custom_provider_base_url = ""
    if 'custom_provider_model_map' not in st.session_state:
        st.session_state.custom_provider_model_map = "{}"
    
    # Create form
    with st.form(key="api_config_form"):
        # OpenAI API Type
        st.subheader("OpenAI API Configuration")
        
        openai_api = st.selectbox(
            "OpenAI API Type",
            options=["responses", "chat_completions"],
            index=["responses", "chat_completions"].index(st.session_state.openai_api),
            format_func=lambda x: f"{x.replace('_', ' ').title()} API" + (" (default)" if x == "responses" else ""),
            help="Choose which OpenAI API to use for agent interactions"
        )
        
        # API Version
        api_version = st.text_input(
            "API Version (optional)",
            value=st.session_state.api_version,
            placeholder="2023-05-15",
            help="Specify a particular OpenAI API version if needed (e.g., '2023-05-15')"
        )
        
        # Custom Model Provider
        st.subheader("Custom Model Provider")
        
        use_custom_provider = st.checkbox(
            "Use Custom Model Provider",
            value=st.session_state.use_custom_provider,
            help="Enable to configure a custom model provider with an OpenAI-compatible API"
        )
        
        if use_custom_provider:
            custom_provider_name = st.text_input(
                "Provider Name",
                value=st.session_state.custom_provider_name,
                placeholder="AnthropicProvider",
                help="Name for your custom provider class"
            )
            
            custom_provider_base_url = st.text_input(
                "Provider Base URL",
                value=st.session_state.custom_provider_base_url,
                placeholder="https://api.anthropic.com",
                help="Base URL for the custom provider's API"
            )
            
            custom_provider_model_map = st.text_area(
                "Model Mapping (JSON)",
                value=st.session_state.custom_provider_model_map,
                placeholder="""{"claude-3-opus-20240229": "claude-3-opus-20240229", "o3-mini": "claude-3-haiku-20240307"}""",
                help="JSON mapping from OpenAI model names to provider model names"
            )
            
            # Validate JSON
            try:
                import json
                if custom_provider_model_map:
                    json.loads(custom_provider_model_map)
            except json.JSONDecodeError:
                st.error("Invalid JSON format for model mapping")
        else:
            custom_provider_name = ""
            custom_provider_base_url = ""
            custom_provider_model_map = "{}"
        
        # Submit button
        submitted = st.form_submit_button("Save API Configuration")
        
        if submitted:
            st.session_state.openai_api = openai_api
            st.session_state.api_version = api_version
            st.session_state.use_custom_provider = use_custom_provider
            
            if use_custom_provider:
                st.session_state.custom_provider_name = custom_provider_name
                st.session_state.custom_provider_base_url = custom_provider_base_url
                st.session_state.custom_provider_model_map = custom_provider_model_map
            
            st.success("API configuration saved!")
    
    # Show generated code
    st.subheader("API Configuration Code")
    
    code = ""
    
    # Add imports
    if st.session_state.use_custom_provider:
        code += """from agents import Agent, set_default_openai_api, set_default_model_provider
from agents.models import ModelProvider
from typing import Dict, Any, Optional
import json
"""
    else:
        code += "from agents import set_default_openai_api\n"
    
    # Add API type setting
    code += f"""
# Set OpenAI API type
set_default_openai_api("{st.session_state.openai_api}")
"""
    
    # Add custom provider if enabled
    if st.session_state.use_custom_provider and st.session_state.custom_provider_name and st.session_state.custom_provider_base_url:
        try:
            import json
            model_map = json.loads(st.session_state.custom_provider_model_map) if st.session_state.custom_provider_model_map else {}
            model_map_str = json.dumps(model_map, indent=4)
            
            code += f"""
# Custom Model Provider
class {st.session_state.custom_provider_name}(ModelProvider):
    def __init__(self):
        self.base_url = "{st.session_state.custom_provider_base_url}"
        self.model_map = {model_map_str}
    
    def get_model_name(self, model_name: str) -> str:
        # Map OpenAI model names to provider model names
        return self.model_map.get(model_name, model_name)
    
    def create_client(self, api_key: Optional[str] = None) -> Any:
        # Create and return a client for your provider
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            {"api_version='" + st.session_state.api_version + "'," if st.session_state.api_version else ""}
        )

# Set the custom provider as default
set_default_model_provider({st.session_state.custom_provider_name}())
"""
        except Exception as e:
            code += f"# Error in custom provider configuration: {str(e)}"
    elif st.session_state.api_version:
        code += f"""
# Set API version
from openai import OpenAI, AsyncOpenAI
OpenAI.api_version = "{st.session_state.api_version}"
AsyncOpenAI.api_version = "{st.session_state.api_version}"
"""
    
    st.code(code, language="python")
    
    # Return the current configuration
    return {
        "openai_api": st.session_state.openai_api,
        "api_version": st.session_state.api_version if st.session_state.api_version else None,
        "custom_provider": {
            "enabled": st.session_state.use_custom_provider,
            "name": st.session_state.custom_provider_name if st.session_state.use_custom_provider else None,
            "base_url": st.session_state.custom_provider_base_url if st.session_state.use_custom_provider else None,
            "model_map": st.session_state.custom_provider_model_map if st.session_state.use_custom_provider else None
        }
    }

# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="API Configuration", layout="wide")
    api_config = api_config_module()
    
    # Show the current configuration
    with st.expander("Current Configuration"):
        st.json(api_config)