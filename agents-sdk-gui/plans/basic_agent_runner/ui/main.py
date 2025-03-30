"""
Main UI rendering for the Agent Runner application.
"""

import streamlit as st
from typing import Dict, Any, Callable
from pathlib import Path

from ui_components.tabs import (
    render_chat_tab, render_configure_tab, render_tools_tab,
    render_history_tab, render_settings_tab, render_mcp_tab
)
from ui_components.session_header import render_session_header, render_session_footer
from services.session_manager import SessionManager


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
    # Initialize session state
    SessionManager.init()
    
    # Load CSS
    css_path = Path(__file__).parent / "styles.css"
    with open(css_path) as f:
        st.markdown(f"""
        <style>
        {f.read()}
        </style>
        """, unsafe_allow_html=True)
    
    # Store the app instance in session state for access from other components
    SessionManager.set("_app", app)
    
    # Render the application title and session header
    st.title("Agent Runner")
    render_session_header()
    
    # Check if we need to switch to the chat tab
    active_tab_index = 0  # Default to chat tab
    
    # If we're coming from agent creation, we want to switch to the chat tab
    if st.session_state.get("switch_to_chat_tab", False):
        active_tab_index = 0  # Chat tab index
        # Clear the flag so we don't keep switching
        st.session_state.switch_to_chat_tab = False
        print("DEBUG: Switching to Chat tab due to switch_to_chat_tab flag")
    
    # Create tabs to organize the UI - try to set the active tab
    try:
        # In some Streamlit versions, the 'index' parameter works
        chat_tab, config_tab, mcp_tab, history_tab, settings_tab = st.tabs(
            ["Chat", "Configure", "MCP", "History", "Settings"],
            index=active_tab_index
        )
    except:
        # Fall back to simpler version if 'index' is not supported
        chat_tab, config_tab, mcp_tab, history_tab, settings_tab = st.tabs([
            "Chat", "Configure", "MCP", "History", "Settings"
        ])
    
    # If agent was just created, show a message and guide user to chat tab
    if SessionManager.check_flag("agent_creating"):
        # Reset the flag so we don't show this message repeatedly
        SessionManager.clear_flag("agent_creating")
        
        # Show a success message with an arrow pointing to the Chat tab
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="background-color: #D4EDDA; color: #155724; padding: 10px; border-radius: 5px; flex-grow: 1;">
                <b>Agent created successfully!</b> You can start chatting now.
            </div>
            <div style="margin-left: 10px; font-size: 24px; color: #155724;">
                ← Click the Chat tab to start using your agent
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Set a flag to indicate we should refresh the header
        SessionManager.set_flag("refresh_header", True)
    
    # Render content in each tab
    with chat_tab:
        # Render chat tab
        agent = SessionManager.get_agent()
        run_config = SessionManager.get_run_config()
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
    
    # Render the session footer at the bottom of the page
    render_session_footer()