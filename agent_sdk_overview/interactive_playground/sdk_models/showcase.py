"""
Showcase for demonstrating enhanced SDK features.
This module provides a standalone UI for showcasing all the enhanced SDK features.
"""
import streamlit as st
import json
from typing import Dict, Any, Optional, List, Union

from .model_selector import model_selector
from .provider_config import ProviderConfig
from .feature_integration import (
    initialize_enhanced_features,
    enhanced_features_selector,
    get_enhanced_system_instruction,
    get_enhanced_model_settings,
    get_tools_for_model,
    get_output_schema,
    generate_enhanced_api_preview,
    generate_enhanced_code_preview
)


def showcase_ui():
    """Main UI for the enhanced features showcase."""
    st.title("Advanced Playground")
    
    st.markdown("""
    This advanced playground demonstrates enhanced features of the OpenAI Agents SDK,
    providing a more comprehensive environment for exploring and prototyping with the SDK.
    
    ### New Features
    
    - **Enhanced Model Support**: Support for OpenAI, Azure OpenAI, and Anthropic models with detailed capabilities
    - **Output Schemas**: Define and test structured output formats
    - **Advanced Tool Integration**: Create, test and visualize custom tools
    - **Response Types & Reasoning**: Configure specific response formats and reasoning approaches
    
    Each feature can be explored in its dedicated tab on the right side of the screen.
    The generated preview will incorporate all selected features and configurations.
    """)
    
    # Initialize features
    initialize_enhanced_features()
    
    # Create two columns for the main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Model selection
        st.header("Model Selection")
        selected_model, provider_config = model_selector()
        
        # Check if provider configuration is valid
        if not ProviderConfig.validate_config(selected_model, provider_config):
            st.warning(
                f"⚠️ Please provide the required configuration for {selected_model.provider.name}."
            )
        
        # Basic input section
        st.header("Input")
        
        system_message = st.text_area(
            "System Instructions:",
            "You are a helpful assistant that provides concise answers.",
            height=100
        )
        
        user_message = st.text_area(
            "User Message:",
            "Explain the difference between neural networks and traditional algorithms in 2-3 sentences.",
            height=100
        )
        
        # Basic model settings
        st.header("Basic Model Settings")
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.number_input("Max tokens:", 1, 4000, 1000, 100)
        
        with col1b:
            top_p = st.slider("Top_p:", 0.0, 1.0, 1.0, 0.1)
            presence_penalty = st.slider("Presence penalty:", 0.0, 2.0, 0.0, 0.1)
        
        # Collect basic settings
        basic_settings = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty
        }
        
        # Basic tool settings
        use_tools = st.checkbox("Enable tools")
        
        tool_settings = {"use_tools": use_tools}
        
        if use_tools:
            col1c, col1d = st.columns(2)
            
            with col1c:
                tool_choice = st.selectbox(
                    "Tool choice:",
                    ["auto", "required", "none"]
                )
                tool_settings["tool_choice"] = tool_choice
            
            with col1d:
                if selected_model.capabilities.get("parallel_tool_calls", None) and selected_model.capabilities["parallel_tool_calls"].supported:
                    parallel_tool_calls = st.checkbox("Enable parallel tool calls")
                    tool_settings["parallel_tool_calls"] = parallel_tool_calls
    
    with col2:
        # Enhanced features
        enhanced_features_selector()
    
    # Generate previews
    st.header("Preview")
    
    if st.button("Generate Preview"):
        # Check if the provider configuration is valid first
        if not ProviderConfig.validate_config(selected_model, provider_config):
            st.error(
                f"⚠️ Missing required configuration for {selected_model.provider.name}. "
                "Please complete the configuration."
            )
        else:
            tabs = st.tabs(["SDK Code", "API Call", "System Instruction"])
            
            with tabs[0]:
                # Generate enhanced code preview
                code_preview = generate_enhanced_code_preview(
                    selected_model,
                    provider_config,
                    basic_settings,
                    tool_settings,
                    system_message,
                    user_message
                )
                st.code(code_preview, language="python")
            
            with tabs[1]:
                # Generate enhanced API call preview
                api_call = generate_enhanced_api_preview(
                    selected_model,
                    provider_config,
                    basic_settings,
                    tool_settings,
                    system_message,
                    user_message
                )
                st.code(json.dumps(api_call, indent=2), language="json")
            
            with tabs[2]:
                # Show enhanced system instruction
                enhanced_system = get_enhanced_system_instruction(system_message)
                st.subheader("Enhanced System Instruction")
                st.markdown(enhanced_system)


if __name__ == "__main__":
    showcase_ui()