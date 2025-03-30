"""
Main UI rendering for the Agent Runner application.
"""

import streamlit as st
from typing import Dict, Any, Callable

from ui_components.tabs import (
    render_chat_tab, render_configure_tab, render_tools_tab,
    render_history_tab, render_settings_tab, render_mcp_tab
)


def render_main_ui(
    app: Any,
    api_key: str,
    env_file_path: str,
    running_agents: Dict[str, Any],
    callbacks: Dict[str, Callable]
) -> None:
    """
    Render the main UI of the application.
    
    Args:
        app: The application instance
        api_key: OpenAI API key
        env_file_path: Path to the .env file
        running_agents: Dictionary of running agents
        callbacks: Dictionary of callback functions
    """
    st.title("Agent Runner")
    
    # Store the app instance in session state for access from other components
    st.session_state._app = app
    
    # Create tabs to organize the UI
    chat_tab, config_tab, mcp_tab, history_tab, settings_tab = st.tabs([
        "Chat", "Configure", "MCP", "History", "Settings"
    ])
    
    # Render content in each tab
    with chat_tab:
        # Render chat tab
        agent = st.session_state.get("agent", None)
        run_config = st.session_state.get("run_config", None)
        render_chat_tab(
            agent=agent,
            run_config=run_config,
            running_agents=running_agents,
            on_clear_chat=callbacks.get("on_clear_chat", lambda: None)
        )
    
    with config_tab:
        # Render the configuration tab
        render_configure_tab(
            api_key=api_key,
            env_file_path=env_file_path,
            on_agent_create=callbacks.get("on_agent_create", lambda *args: None),
            on_api_key_save=callbacks.get("on_api_key_save", lambda *args: None),
            on_api_key_load=callbacks.get("on_api_key_load", lambda *args: None)
        )
    
    with mcp_tab:
        # Render the MCP tab
        render_mcp_tab(
            on_server_add=callbacks.get("on_mcp_server_add", lambda *args: None),
            on_server_remove=callbacks.get("on_mcp_server_remove", lambda *args: None),
            on_server_test=callbacks.get("on_mcp_server_test", lambda *args: None),
            on_server_select=callbacks.get("on_mcp_server_select", lambda *args: None)
        )
    
    with history_tab:
        # Render the history tab
        render_history_tab()
    
    with settings_tab:
        # Render the settings tab
        render_settings_tab(
            api_key=api_key,
            env_file_path=env_file_path,
            on_api_key_save=callbacks.get("on_api_key_save", lambda *args: None),
            on_api_key_load=callbacks.get("on_api_key_load", lambda *args: None)
        )