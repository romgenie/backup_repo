"""
Integration Utilities for Model Management System

This module provides helper functions for integrating the model management
system with the existing Agent Runner application.
"""

import streamlit as st
import os
import sys
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from pathlib import Path

def integrate_model_management_css():
    """
    Integrate model management CSS styles into the main application.
    
    This should be called from the main UI initialization.
    """
    # Get the path to the CSS file
    ui_improvements_dir = Path(__file__).parent
    css_path = ui_improvements_dir / "enhanced_styles.css"
    
    # Add model management specific styles
    model_management_css = """
    /* Model Management Specific Styles */
    .provider-card, .model-card {
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        transition: all 0.2s ease;
    }
    
    .provider-card:hover, .model-card:hover {
        background-color: rgba(65, 105, 225, 0.15);
        transform: translateY(-2px);
    }
    
    .provider-icon {
        font-size: 24px;
        margin-bottom: 8px;
    }
    
    .model-name {
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 4px;
    }
    
    .model-tokens, .model-description {
        color: #626E7F;
        font-size: 14px;
    }
    
    /* Parameter form styling */
    .parameter-form {
        border: 1px solid rgba(65, 105, 225, 0.2);
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
    }
    
    /* Tab styling for model parameters */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 8px 16px;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #28a745;
    }
    
    .status-inactive {
        background-color: #dc3545;
    }
    
    .status-warning {
        background-color: #ffc107;
    }
    """
    
    # Check if the CSS file exists
    if css_path.exists():
        # Read existing CSS
        with open(css_path, "r") as f:
            css_content = f.read()
        
        # Append model management CSS if it's not already there
        if "Model Management Specific Styles" not in css_content:
            with open(css_path, "a") as f:
                f.write("\n\n" + model_management_css)
    else:
        # Create the CSS file with model management styles
        with open(css_path, "w") as f:
            f.write(model_management_css)
    
    # Apply the CSS to the current page
    st.markdown(f"""
    <style>
    {model_management_css}
    </style>
    """, unsafe_allow_html=True)

def integrate_with_agent_creation(
    create_agent_func: Callable,
    config_container: Any = None
) -> Callable:
    """
    Enhance an existing agent creation function with model selection capabilities.
    
    Args:
        create_agent_func: The original agent creation function
        config_container: Optional container to render the UI in
        
    Returns:
        Callable: Enhanced function that includes model selection
    """
    from ui_improvements.model_selector import render_compact_model_selector
    from ui_improvements.model_management import create_agent_with_model
    
    def enhanced_create_agent(*args, **kwargs):
        # If we have a specific container, use it; otherwise use st directly
        ui = config_container or st
        
        with ui.expander("Model Selection", expanded=True):
            ui.markdown("### Select AI Model")
            ui.write("Choose the AI model to use for this agent")
            
            # Render the compact model selector
            provider, model, settings = render_compact_model_selector(
                show_settings=True
            )
            
            # Add these to kwargs if not already present
            if "model_provider" not in kwargs:
                kwargs["model_provider"] = provider
            if "model_name" not in kwargs:
                kwargs["model_name"] = model
            if "model_settings" not in kwargs:
                kwargs["model_settings"] = settings
        
        # Call the original function with enhanced arguments
        return create_agent_func(*args, **kwargs)
    
    return enhanced_create_agent

def integrate_with_agent_runner(
    runner_module_path: str
) -> bool:
    """
    Integrate model management with the agent runner module.
    
    This patches the agent_runner.py module to use the selected model
    when creating agents.
    
    Args:
        runner_module_path: Path to the agent_runner.py module
        
    Returns:
        bool: True if integration was successful, False otherwise
    """
    try:
        # Import the necessary modules
        sys.path.append(os.path.dirname(os.path.dirname(runner_module_path)))
        from agent_management.mcp import configuration
        from ui_improvements.model_management import (
            get_default_model,
            setup_global_openai_client,
            create_custom_model_provider,
            create_agent_with_model
        )
        
        # Get the current model configuration
        if "model_config" not in st.session_state:
            from ui_improvements.model_management import initialize_model_config
            st.session_state.model_config = initialize_model_config()
        
        config = st.session_state.model_config
        integration_method = config.get("integration_method", "agent_model")
        
        # Apply the appropriate integration method
        if integration_method == "global_client":
            # Set up global OpenAI client
            success = setup_global_openai_client(config)
            if not success:
                st.warning("Failed to set up global OpenAI client")
                return False
            
        return True
    except Exception as e:
        st.error(f"Failed to integrate with agent runner: {str(e)}")
        return False

def add_to_configure_tab(
    tab_renderer: Callable,
    tab_key: str = "model_tab"
) -> Callable:
    """
    Add model management to the configure tab.
    
    Args:
        tab_renderer: The original tab rendering function
        tab_key: Key for the new model tab
        
    Returns:
        Callable: Enhanced tab rendering function
    """
    from ui_improvements.model_selector import render_visual_model_selector
    
    def enhanced_tab_renderer(*args, **kwargs):
        # Call the original renderer
        tabs = tab_renderer(*args, **kwargs)
        
        # Check if the result is a list of tab objects
        if isinstance(tabs, (list, tuple)) and len(tabs) > 0:
            # Add a new tab for model management
            model_tab = st.tabs(["Model Selection"])[0]
            tabs.append(model_tab)
            
            with model_tab:
                st.header("AI Model Selection")
                st.write("Select the AI model to use for your agents")
                
                # Render the visual model selector
                provider, model, settings = render_visual_model_selector()
                
                # Show current selection
                st.divider()
                st.subheader("Current Default Model")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Provider", provider)
                with col2:
                    st.metric("Model", model)
                
                # Show a link to the full model management page
                st.markdown("""
                For advanced model configuration, visit the full
                [Model Management](#) page.
                """)
        
        return tabs
    
    return enhanced_tab_renderer

def render_model_tab():
    """
    Render a standalone model management tab.
    
    This can be used as a new tab in the main application.
    """
    from ui_improvements.model_management import render_model_management
    
    st.title("Model Management")
    st.write("Configure and manage AI models for your agents")
    
    # Render the full model management UI
    render_model_management()

def get_model_info_for_agent(
    agent_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get model information for an agent.
    
    Args:
        agent_name: Optional name of the agent to get model info for
        
    Returns:
        Dict: Model information
    """
    from ui_improvements.model_management import get_default_model
    
    # Get the default model
    provider, model_id, settings = get_default_model()
    
    # Get provider and model display info
    from ui_improvements.model_management import (
        SUPPORTED_PROVIDERS,
        get_current_model_info
    )
    
    provider_info = SUPPORTED_PROVIDERS.get(provider, {})
    model_info = get_current_model_info(provider, model_id)
    
    return {
        "provider": provider,
        "provider_display": provider,
        "model_id": model_id,
        "model_display": model_info.get("name", model_id),
        "context_tokens": model_info.get("tokens", 0),
        "description": model_info.get("description", ""),
        "settings": settings.to_dict() if settings else {}
    }

def show_model_info(container: Any = None):
    """
    Display model information in a sidebar or container.
    
    Args:
        container: Optional container to render the info in
    """
    # Use the provided container or st directly
    ui = container or st
    
    # Get model info
    model_info = get_model_info_for_agent()
    
    # Show in an expandable section
    with ui.expander("Current AI Model", expanded=False):
        # Display provider and model
        col1, col2 = ui.columns(2)
        with col1:
            ui.metric("Provider", model_info["provider_display"])
        with col2:
            ui.metric("Model", model_info["model_display"])
        
        # Show context window
        ui.write(f"Context window: {model_info['context_tokens']:,} tokens")
        
        # Show basic settings
        settings = model_info.get("settings", {})
        if settings:
            ui.write("Settings:")
            settings_markdown = ""
            if "temperature" in settings:
                settings_markdown += f"- Temperature: {settings['temperature']:.2f}  \n"
            if "top_p" in settings:
                settings_markdown += f"- Top P: {settings['top_p']:.2f}  \n"
            if "max_tokens" in settings:
                settings_markdown += f"- Max tokens: {settings['max_tokens']:,}  \n"
            
            ui.markdown(settings_markdown)
        
        # Link to change model
        ui.write("[Change model](#)")

def apply_model_to_agent_config(
    agent_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply the current model configuration to an agent config.
    
    Args:
        agent_config: The agent configuration dictionary
        
    Returns:
        Dict: Updated agent configuration
    """
    from ui_improvements.model_management import get_default_model
    
    # Get the default model
    provider, model_id, settings = get_default_model()
    
    # Update the agent config
    if "model" not in agent_config:
        agent_config["model"] = model_id
        
    if settings and "model_settings" not in agent_config:
        # Convert our ModelSettings to a dict for the agent config
        agent_config["model_settings"] = settings.to_dict()
    
    return agent_config