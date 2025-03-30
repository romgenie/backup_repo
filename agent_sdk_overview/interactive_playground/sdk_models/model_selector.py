"""
Model selector component for the SDK Playground.
Provides a UI for selecting models with provider filtering.
"""
import streamlit as st
from typing import Dict, List, Optional, Tuple, Callable

# Import from new models_schema instead of model_definitions
from .models_schema import (
    ModelSchema,
    ModelProvider,
    PROVIDER_INFO,
    MODEL_SCHEMAS_BY_PROVIDER,
    get_model_schema_by_id,
    get_model_schemas_for_agents_sdk,
    get_model_schemas_for_responses_api
)


def model_selector(
    on_model_select: Optional[Callable[[ModelSchema], None]] = None
) -> Tuple[ModelSchema, Dict]:
    """
    Display a model selector UI component.
    
    Args:
        on_model_select: Optional callback when a model is selected
        
    Returns:
        Tuple of (selected_model, provider_config)
    """
    st.subheader("Model Selection")
    
    # Filter for models compatible with Agents SDK and Responses API
    compatible_models = get_model_schemas_for_responses_api()
    
    # Rebuild provider groups based on compatible models
    compatible_provider_ids = set(model.provider for model in compatible_models)
    compatible_providers = {p_id: PROVIDER_INFO[p_id] for p_id in compatible_provider_ids}
    
    # Create tabs for each provider
    provider_tabs = st.tabs([p.name for p in compatible_providers.values()])
    
    # Session state to keep track of selected provider and model
    if "selected_provider_id" not in st.session_state:
        st.session_state.selected_provider_id = ModelProvider.OPENAI
    
    if "selected_model_id" not in st.session_state:
        st.session_state.selected_model_id = "gpt-4-turbo"
    
    # Provider configs
    if "provider_configs" not in st.session_state:
        st.session_state.provider_configs = {
            ModelProvider.OPENAI: {"api_key": "", "base_url": ""},
            ModelProvider.AZURE_OPENAI: {"api_key": "", "endpoint": "", "api_version": "2023-12-01-preview"},
            ModelProvider.ANTHROPIC: {"api_key": ""},
            ModelProvider.CUSTOM: {"api_key": "", "base_url": ""},
        }
    
    selected_model = None
    provider_config = {}
    
    # Create a tab for each provider
    for i, (provider_id, provider) in enumerate(compatible_providers.items()):
        with provider_tabs[i]:
            # Provider description
            st.markdown(f"**{provider.description}**")
            
            # Provider configuration
            st.subheader("Provider Configuration")
            config = st.session_state.provider_configs[provider_id]
            
            if provider.requires_key:
                config["api_key"] = st.text_input(
                    f"{provider.name} API Key", 
                    value=config.get("api_key", ""),
                    type="password",
                    key=f"{provider_id}_api_key"
                )
            
            if provider.requires_endpoint:
                config["endpoint"] = st.text_input(
                    f"{provider.name} Endpoint", 
                    value=config.get("endpoint", ""),
                    key=f"{provider_id}_endpoint"
                )
                
                if provider_id == ModelProvider.AZURE_OPENAI:
                    config["api_version"] = st.text_input(
                        "API Version",
                        value=config.get("api_version", "2023-12-01-preview"),
                        key=f"{provider_id}_api_version"
                    )
            
            if provider.custom_base_url:
                config["base_url"] = st.text_input(
                    "Custom Base URL (optional)", 
                    value=config.get("base_url", ""),
                    key=f"{provider_id}_base_url"
                )
                
            st.session_state.provider_configs[provider_id] = config
            
            # Get models for this provider that are compatible
            provider_models = [m for m in MODEL_SCHEMAS_BY_PROVIDER.get(provider_id, []) 
                               if m.supports_agents_sdk and m.supports_responses_api]
            
            if provider_models:
                # Create model selection
                model_options = {model.name: model.id for model in provider_models}
                
                # Get default model for this provider if none selected yet
                if provider_id == st.session_state.selected_provider_id:
                    default_model_name = next(
                        (name for name, mid in model_options.items() 
                         if mid == st.session_state.selected_model_id),
                        next(iter(model_options.keys()))
                    )
                else:
                    default_model_name = next(iter(model_options.keys()))
                
                selected_model_name = st.selectbox(
                    "Select a model:",
                    options=model_options.keys(),
                    index=list(model_options.keys()).index(default_model_name),
                    key=f"{provider_id}_model_selector"
                )
                
                # Update selected model and provider when changed
                selected_model_id = model_options[selected_model_name]
                
                # If this is the active provider tab, update the selection
                if st.button("Use This Model", key=f"use_{provider_id}"):
                    st.session_state.selected_provider_id = provider_id
                    st.session_state.selected_model_id = selected_model_id
                    selected_model = get_model_schema_by_id(selected_model_id)
                    provider_config = st.session_state.provider_configs[provider_id]
                    
                    # Call the callback if provided
                    if on_model_select and selected_model:
                        on_model_select(selected_model)
                    
                    st.rerun()
                
                # If this is the active provider, show model details
                if provider_id == st.session_state.selected_provider_id and selected_model_id == st.session_state.selected_model_id:
                    display_model_details(get_model_schema_by_id(selected_model_id))
            else:
                st.info(f"No compatible models defined for {provider.name}.")
    
    # Get the currently selected model and provider config
    current_model = get_model_schema_by_id(st.session_state.selected_model_id)
    current_provider_config = st.session_state.provider_configs[st.session_state.selected_provider_id]
    
    # Return the current model if no new selection was made
    return selected_model or current_model, provider_config or current_provider_config


def display_model_details(model: ModelSchema):
    """Display detailed information about a model."""
    if not model:
        return
    
    st.markdown("---")
    st.markdown(f"### {model.name}")
    st.markdown(model.description)
    
    # Model specs in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Context Window", f"{model.context_window:,} tokens")
    with col2:
        st.metric("Max Output", f"{model.token_limit:,} tokens")
    with col3:
        st.metric("Provider", PROVIDER_INFO[model.provider].name)
    
    # Display model capabilities
    st.markdown("#### Capabilities")
    
    capability_cols = st.columns(3)
    for i, (feature, capability) in enumerate(model.capabilities.items()):
        col = capability_cols[i % 3]
        with col:
            status = "✅" if capability.supported else "❌"
            details = f" ({capability.details})" if capability.details else ""
            feature_name = feature.value.replace("_", " ").title()
            col.markdown(f"{status} **{feature_name}**{details}")
    
    # Display API compatibilities
    st.markdown("#### API Compatibilities")
    api_cols = st.columns(4)
    for i, api in enumerate(sorted(model.api_compatibilities)):
        col = api_cols[i % 4]
        col.markdown(f"• {api}")
            
    # Example prompt if available
    if model.example_prompt:
        with st.expander("Example Prompt"):
            st.markdown(model.example_prompt)