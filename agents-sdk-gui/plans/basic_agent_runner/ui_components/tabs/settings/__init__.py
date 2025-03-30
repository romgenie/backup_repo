"""
Settings module for the application.
"""

import streamlit as st
from typing import Optional, Callable

from .agent_settings import render_api_key_settings, render_model_settings
from .advanced_settings import render_debug_settings
from .ui_settings import render_ui_settings
from .mcp_settings import render_mcp_settings
from .about import render_about_section

def render_settings_tab(
    api_key: str,
    env_file_path: str,
    on_api_key_save: Callable[[str], str],
    on_api_key_load: Callable[[str], tuple]
) -> None:
    """
    Render the settings tab of the application
    
    Args:
        api_key: Current API key
        env_file_path: Path to the .env file
        on_api_key_save: Callback to save API key
        on_api_key_load: Callback to load API key
    """
    # Import required modules
    from services.session_manager import SessionManager
    
    st.markdown("""
    <div class="form-section">
        <h2>Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different settings categories
    settings_tabs = st.tabs(["API Keys", "Debug", "UI", "MCP", "About"])
    
    # API Keys tab
    with settings_tabs[0]:
        render_api_key_settings(api_key, env_file_path, on_api_key_save, on_api_key_load)
    
    # Debug tab
    with settings_tabs[1]:
        render_debug_settings()
    
    # UI Settings tab
    with settings_tabs[2]:
        render_ui_settings()
    
    # MCP Settings tab
    with settings_tabs[3]:
        render_mcp_settings()
    
    # About tab
    with settings_tabs[4]:
        render_about_section()