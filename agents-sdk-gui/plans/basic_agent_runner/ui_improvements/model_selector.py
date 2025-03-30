"""
Model Selector Component for Agent Runner

This module provides a lightweight model selection component that can be
integrated into the Configure tab and other parts of the application.
"""

import streamlit as st
from typing import Dict, List, Optional, Tuple, Callable, Any
import os
from pathlib import Path

from ui_improvements.model_management import (
    initialize_model_config, 
    save_model_config,
    get_default_model,
    get_current_model_info,
    ModelSettings,
    SUPPORTED_PROVIDERS
)

def render_compact_model_selector(
    on_model_change: Optional[Callable[[str, str, ModelSettings], None]] = None,
    show_settings: bool = False
) -> Tuple[str, str, ModelSettings]:
    """
    Render a compact model selection component for use within other UI components.
    
    This provides a simpler interface than the full model management page, designed
    to be embedded within configure tabs and other components.
    
    Args:
        on_model_change: Optional callback when model selection changes
        show_settings: Whether to show a subset of model settings
        
    Returns:
        Tuple[str, str, ModelSettings]: Provider name, model ID, and model settings
    """
    # Initialize or load configuration
    if "model_config" not in st.session_state:
        st.session_state.model_config = initialize_model_config()
    
    config = st.session_state.model_config
    
    # Get current selections
    provider, model_id, settings = get_default_model()
    
    # Create columns for provider and model selection
    col1, col2 = st.columns([1, 2])
    
    # Provider selection
    with col1:
        st.markdown("### Provider")
        available_providers = list(SUPPORTED_PROVIDERS.keys())
        selected_provider = st.selectbox(
            "Select Provider",
            available_providers,
            index=available_providers.index(provider) if provider in available_providers else 0,
            key="compact_provider_select",
            label_visibility="collapsed"
        )
    
    # Model selection
    with col2:
        st.markdown("### Model")
        provider_info = SUPPORTED_PROVIDERS.get(selected_provider, {})
        models = provider_info.get("models", [])
        
        if models:
            model_options = [(m["id"], f"{m['name']} - {m['tokens']:,} tokens") for m in models]
            selected_model_idx = next((i for i, (id, _) in enumerate(model_options) if id == model_id), 0)
            
            selected_model = st.selectbox(
                "Select Model",
                options=[id for id, _ in model_options],
                format_func=lambda x: next((desc for id, desc in model_options if id == x), x),
                index=selected_model_idx if model_options else 0,
                key="compact_model_select",
                label_visibility="collapsed"
            )
            
            # Get current model info
            model_info = get_current_model_info(selected_provider, selected_model)
        else:
            st.warning(f"No models available for {selected_provider}")
            selected_model = ""
            model_info = {"id": "", "name": "None", "tokens": 0, "description": ""}
            
    # Show basic model info
    if selected_model:
        # Display model information in an info box
        st.info(
            f"**{model_info['name']}**: {model_info['description']}\n\n"
            f"Max context size: {model_info['tokens']:,} tokens"
        )
        
        # Check if this is a different selection than the default
        is_default = (selected_provider == provider and selected_model == model_id)
        
        if not is_default:
            if st.button("Set as Default", key="set_compact_default"):
                # Update default provider and model
                config["default_provider"] = selected_provider
                
                if "default_models" not in config:
                    config["default_models"] = {}
                config["default_models"][selected_provider] = selected_model
                
                # Save configuration
                save_model_config(config)
                st.session_state.model_config = config
                
                st.success(f"{model_info['name']} set as default model")
                
                # If we have a callback, notify about the change
                if on_model_change:
                    updated_provider, updated_model, updated_settings = get_default_model()
                    on_model_change(updated_provider, updated_model, updated_settings)
                    
                # Rerun to update the UI
                st.rerun()
    
    # Show simplified settings UI if requested
    if show_settings and selected_model:
        st.markdown("### Quick Settings")
        
        # Get model settings
        from ui_improvements.model_management import get_model_settings
        model_settings = get_model_settings(config, selected_provider, selected_model)
        
        # Create columns for temperature and top_p
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=model_settings.temperature if model_settings.temperature is not None else 0.7,
                step=0.05,
                help="Controls randomness. Lower values are more deterministic, higher values more creative."
            )
            
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=model_info.get("tokens", 4096),
                value=model_settings.max_tokens if model_settings.max_tokens is not None else 1000,
                help="Maximum number of output tokens to generate."
            )
        
        with col2:
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=model_settings.top_p if model_settings.top_p is not None else 1.0,
                step=0.05,
                help="Controls diversity via nucleus sampling. 1.0 considers all tokens."
            )
            
            # Tool choice options
            tool_choice_options = [None, "auto", "required", "none"]
            tool_choice_index = 0
            
            # Find index for the current value
            if model_settings.tool_choice is not None:
                try:
                    tool_choice_index = tool_choice_options.index(model_settings.tool_choice)
                except ValueError:
                    # If it's a custom value not in the list
                    tool_choice_options.append(model_settings.tool_choice)
                    tool_choice_index = len(tool_choice_options) - 1
            
            tool_choice = st.selectbox(
                "Tool Choice",
                options=tool_choice_options,
                index=tool_choice_index,
                format_func=lambda x: "Default" if x is None else x,
                help="Controls how the model uses tools. 'auto' lets the model decide, 'required' forces tool use, 'none' prevents tool use."
            )
        
        # Save button for settings
        if st.button("Save Settings", key="save_compact_settings"):
            # Update the settings in the config
            updated_settings = ModelSettings(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                # Keep other settings as they were
                frequency_penalty=model_settings.frequency_penalty,
                presence_penalty=model_settings.presence_penalty,
                parallel_tool_calls=model_settings.parallel_tool_calls,
                truncation=model_settings.truncation,
                store=model_settings.store,
                tracing=model_settings.tracing
            )
            
            # Ensure model_settings structure exists in config
            if "model_settings" not in config:
                config["model_settings"] = {}
            if selected_provider not in config["model_settings"]:
                config["model_settings"][selected_provider] = {}
            
            # Save settings for the provider and model
            config["model_settings"][selected_provider][selected_model] = updated_settings.to_dict()
            
            # Save configuration
            save_model_config(config)
            st.session_state.model_config = config
            
            st.success(f"Settings for {model_info['name']} saved")
            
            # If we have a callback, notify about the change
            if on_model_change:
                updated_provider, updated_model, updated_settings = get_default_model()
                on_model_change(updated_provider, updated_model, updated_settings)
                
            # Rerun to update the UI
            st.rerun()
        
        # Link to full settings page
        st.write("For more advanced settings, visit the [Model Management](#) page.")
    
    # Return current selection (provider, model, settings)
    if selected_model:
        from ui_improvements.model_management import get_model_settings
        current_settings = get_model_settings(config, selected_provider, selected_model)
        return selected_provider, selected_model, current_settings
    else:
        # If no model is selected, return the current defaults
        return get_default_model()

def render_provider_card(provider_name: str, is_selected: bool = False, on_click: Optional[Callable] = None):
    """
    Render a provider card for visual selection UI.
    
    Args:
        provider_name: The name of the provider
        is_selected: Whether this provider is currently selected
        on_click: Callback when the card is clicked
    """
    # Determine styling based on selection state
    card_border = "3px solid #4169E1" if is_selected else "1px solid rgba(65, 105, 225, 0.2)"
    card_bg = "rgba(65, 105, 225, 0.1)" if is_selected else "rgba(65, 105, 225, 0.02)"
    
    # Provider logo/icon based on name
    if provider_name == "OpenAI":
        icon = "🟢"  # Green circle for OpenAI
    elif provider_name == "Anthropic":
        icon = "🟣"  # Purple circle for Anthropic
    elif provider_name == "Cohere":
        icon = "🔵"  # Blue circle for Cohere
    else:
        icon = "⚪"  # Default
    
    # Create a clickable provider card
    html = f"""
    <div class="provider-card" style="border: {card_border}; background-color: {card_bg};">
        <div class="provider-icon">{icon}</div>
        <div class="provider-name">{provider_name}</div>
        <div class="provider-status">{"✓ Selected" if is_selected else ""}</div>
    </div>
    <style>
    .provider-card {{
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        cursor: pointer;
        height: 100%;
        transition: all 0.2s ease;
        box-shadow: {f"0 4px 8px rgba(65, 105, 225, 0.2)" if is_selected else "none"};
    }}
    .provider-card:hover {{
        background-color: rgba(65, 105, 225, 0.15);
        border-color: rgba(65, 105, 225, 0.4);
    }}
    .provider-icon {{
        font-size: 24px;
        margin-bottom: 8px;
    }}
    .provider-name {{
        font-weight: 500;
        margin-bottom: 4px;
    }}
    .provider-status {{
        font-size: 12px;
        color: #4169E1;
    }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)
    
    # Button to select this provider
    button_label = "Selected" if is_selected else "Select"
    if st.button(button_label, key=f"select_provider_{provider_name}", disabled=is_selected):
        if on_click:
            on_click(provider_name)

def render_model_card(model: Dict, is_selected: bool = False, on_click: Optional[Callable] = None):
    """
    Render a model card for visual selection UI.
    
    Args:
        model: The model information dictionary
        is_selected: Whether this model is currently selected
        on_click: Callback when the card is clicked
    """
    # Determine styling based on selection state
    card_border = "3px solid #4169E1" if is_selected else "1px solid rgba(65, 105, 225, 0.2)"
    card_bg = "rgba(65, 105, 225, 0.1)" if is_selected else "rgba(65, 105, 225, 0.02)"
    
    # Create model card with consistent styling
    html = f"""
    <div class="model-card" style="border: {card_border}; background-color: {card_bg};">
        <div class="model-name">{model["name"]}</div>
        <div class="model-tokens">{model["tokens"]:,} tokens</div>
        <div class="model-description">{model["description"]}</div>
        <div class="model-status">{"✓ Selected" if is_selected else ""}</div>
    </div>
    <style>
    .model-card {{
        padding: 16px;
        border-radius: 8px;
        height: 100%;
        transition: all 0.2s ease;
        box-shadow: {f"0 4px 8px rgba(65, 105, 225, 0.2)" if is_selected else "none"};
        margin-bottom: 16px;
    }}
    .model-card:hover {{
        background-color: rgba(65, 105, 225, 0.15);
        border-color: rgba(65, 105, 225, 0.4);
    }}
    .model-name {{
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 4px;
        color: #343D4A;
    }}
    .model-tokens {{
        color: #626E7F;
        font-size: 14px;
        margin-bottom: 8px;
    }}
    .model-description {{
        font-size: 13px;
        color: #626E7F;
        margin-bottom: 12px;
        min-height: 40px;
    }}
    .model-status {{
        font-size: 12px;
        color: #4169E1;
        font-weight: 500;
    }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)
    
    # Button to select this model
    button_label = "Selected" if is_selected else "Select"
    if st.button(button_label, key=f"select_model_{model['id']}", disabled=is_selected):
        if on_click:
            on_click(model["id"])

def render_visual_model_selector(
    on_model_change: Optional[Callable[[str, str, ModelSettings], None]] = None
) -> Tuple[str, str, ModelSettings]:
    """
    Render a visual model selection component with cards.
    
    This provides a more visually appealing interface than the compact selector,
    showing provider and model cards for selection.
    
    Args:
        on_model_change: Optional callback when model selection changes
        
    Returns:
        Tuple[str, str, ModelSettings]: Provider name, model ID, and model settings
    """
    # Initialize or load configuration
    if "model_config" not in st.session_state:
        st.session_state.model_config = initialize_model_config()
    
    config = st.session_state.model_config
    
    # Get current selections
    provider, model_id, settings = get_default_model()
    
    st.subheader("Select AI Provider")
    
    # Provider selection with cards
    available_providers = list(SUPPORTED_PROVIDERS.keys())
    
    # Calculate columns for providers (max 4 per row)
    cols_per_row = min(4, len(available_providers))
    provider_cols = st.columns(cols_per_row)
    
    # Render provider cards
    for i, provider_name in enumerate(available_providers):
        col_idx = i % cols_per_row
        with provider_cols[col_idx]:
            # Determine if this provider is selected
            is_selected = provider_name == provider
            
            # Callback for when provider is selected
            def select_provider(selected_provider):
                nonlocal provider
                # Update the default provider
                config["default_provider"] = selected_provider
                save_model_config(config)
                st.session_state.model_config = config
                
                # If we have a callback, notify about the change
                if on_model_change:
                    updated_provider, updated_model, updated_settings = get_default_model()
                    on_model_change(updated_provider, updated_model, updated_settings)
                
                st.rerun()
            
            # Render the provider card
            render_provider_card(
                provider_name=provider_name,
                is_selected=is_selected,
                on_click=select_provider
            )
    
    st.subheader(f"Select Model for {provider}")
    
    # Model selection with cards
    provider_info = SUPPORTED_PROVIDERS.get(provider, {})
    models = provider_info.get("models", [])
    
    if models:
        # Calculate columns for models (max 3 per row)
        cols_per_row = min(3, len(models))
        rows = (len(models) + cols_per_row - 1) // cols_per_row  # Calculate rows needed
        
        for row in range(rows):
            # Create columns for this row
            model_cols = st.columns(cols_per_row)
            
            # Fill columns with models
            for col_idx in range(cols_per_row):
                model_idx = row * cols_per_row + col_idx
                
                # Check if we have a model for this slot
                if model_idx < len(models):
                    model = models[model_idx]
                    
                    with model_cols[col_idx]:
                        # Determine if this model is selected
                        is_selected = model["id"] == model_id
                        
                        # Callback for when model is selected
                        def select_model(selected_model_id):
                            # Update the default model for this provider
                            if "default_models" not in config:
                                config["default_models"] = {}
                            config["default_models"][provider] = selected_model_id
                            save_model_config(config)
                            st.session_state.model_config = config
                            
                            # If we have a callback, notify about the change
                            if on_model_change:
                                updated_provider, updated_model, updated_settings = get_default_model()
                                on_model_change(updated_provider, updated_model, updated_settings)
                            
                            st.rerun()
                        
                        # Render the model card
                        render_model_card(
                            model=model,
                            is_selected=is_selected,
                            on_click=select_model
                        )
    else:
        st.warning(f"No models available for {provider}")
    
    # Return current selection (provider, model, settings)
    return provider, model_id, settings

if __name__ == "__main__":
    st.set_page_config(page_title="Model Selector Demo", layout="wide")
    
    st.title("Model Selector Demo")
    
    tab1, tab2 = st.tabs(["Compact Selector", "Visual Selector"])
    
    with tab1:
        st.header("Compact Model Selector")
        st.write("This is a lightweight version for embedding in other UI components.")
        
        def on_model_change(provider, model, settings):
            st.write(f"Selected: {provider} - {model}")
            st.write(f"Settings: {settings}")
        
        provider, model, settings = render_compact_model_selector(
            on_model_change=on_model_change,
            show_settings=True
        )
        
        st.divider()
        st.markdown("### Selected Model")
        st.json({
            "provider": provider,
            "model": model,
            "settings": settings.to_dict() if settings else {}
        })
    
    with tab2:
        st.header("Visual Model Selector")
        st.write("This is a more visually appealing version for dedicated model selection pages.")
        
        provider, model, settings = render_visual_model_selector()
        
        st.divider()
        st.markdown("### Selected Model")
        st.json({
            "provider": provider,
            "model": model,
            "settings": settings.to_dict() if settings else {}
        })