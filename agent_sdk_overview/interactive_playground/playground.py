"""
Interactive playground for experimenting with model settings and API calls.
This module provides the main UI for the playground.
"""
import streamlit as st
import json
from agents.models.interface import ModelTracing
from agents.model_settings import ModelSettings

# Import the refactored model support
from .sdk_models.model_selector import model_selector
from .sdk_models.provider_config import ProviderConfig
from .sdk_models.playground_integration import (
    initialize_model_support,
    get_model_settings_ui,
    get_tool_settings_ui,
    generate_model_code_preview,
    generate_api_call_preview
)


def display_playground():
    """
    Interactive playground for experimenting with model settings and
    seeing how they translate to API calls.
    """
    st.header("Interactive Playground")
    
    st.markdown("""
    Use this playground to experiment with model settings and see how they translate to API calls.
    This will help you understand how the SDK formats requests and processes responses.
    """)
    
    # Initialize model support
    initialize_model_support()
    
    # Model selection section
    st.subheader("Model Selection")
    selected_model, provider_config = model_selector()
    
    # Check if provider configuration is valid
    if not ProviderConfig.validate_config(selected_model, provider_config):
        st.warning(
            f"⚠️ Please provide the required configuration for {selected_model.provider.name} "
            "in the Model Selection section."
        )
    
    # System message
    system_message = st.text_area(
        "System instructions:",
        "You are a helpful assistant that provides concise answers."
    )
    
    # User message
    user_message = st.text_area(
        "User message:",
        "Explain the difference between neural networks and traditional algorithms in 2-3 sentences."
    )
    
    # Model settings
    st.subheader("Model Settings")
    
    # Get model-specific settings
    model_settings = get_model_settings_ui(selected_model)
    
    # Get tool settings if applicable
    tool_settings = get_tool_settings_ui(selected_model)
    
    # Generate API call preview
    if st.button("Generate Preview"):
        # Check if the provider configuration is valid first
        if not ProviderConfig.validate_config(selected_model, provider_config):
            st.error(
                f"⚠️ Missing required configuration for {selected_model.provider.name}. "
                "Please complete the configuration in the Model Selection section."
            )
        else:
            # Generate code preview
            st.subheader("SDK Code")
            code_preview = generate_model_code_preview(
                selected_model,
                provider_config,
                model_settings,
                tool_settings,
                system_message,
                user_message
            )
            st.code(code_preview, language="python")
            
            # Generate API call preview
            st.subheader(f"Equivalent {selected_model.provider.name} API Call")
            api_call = generate_api_call_preview(
                selected_model,
                provider_config,
                model_settings,
                tool_settings,
                system_message,
                user_message
            )
            st.code(json.dumps(api_call, indent=2), language="json")