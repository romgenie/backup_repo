"""
Advanced and debug settings components for settings tab.
"""

import streamlit as st
import os
import base64
import json
from typing import Dict, Any, Optional

def render_debug_settings() -> None:
    """Render debug settings section"""
    st.markdown("""
    <div class="form-section">
        <h3>Debug Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize debug flags if not present
    if "debug_flags" not in st.session_state:
        st.session_state.debug_flags = {
            "show_session_debug": False,
            "verbose_logging": False,
            "show_raw_responses": False,
            "tool_call_debugging": True
        }
    
    # Get the current settings from session state
    debug_flags = st.session_state.debug_flags
    
    # Session debug toggle
    debug_flags["show_session_debug"] = st.checkbox(
        "Show Session State Debug Panel",
        value=debug_flags.get("show_session_debug", False),
        help="Show a debug panel with session state information"
    )
    
    # Verbose logging toggle
    debug_flags["verbose_logging"] = st.checkbox(
        "Enable Verbose Logging",
        value=debug_flags.get("verbose_logging", False),
        help="Print detailed logs to the console"
    )
    
    # Show raw responses toggle
    debug_flags["show_raw_responses"] = st.checkbox(
        "Show Raw API Responses",
        value=debug_flags.get("show_raw_responses", False),
        help="Display raw API responses in the UI"
    )
    
    # Tool call debugging toggle
    debug_flags["tool_call_debugging"] = st.checkbox(
        "Tool Call Debugging",
        value=debug_flags.get("tool_call_debugging", True),
        help="Show detailed information about tool calls"
    )
    
    # Session state management
    with st.expander("Session State Management", expanded=False):
        # Clear session state button with confirmation
        if "confirm_clear_session" not in st.session_state:
            st.session_state.confirm_clear_session = False
            
        if not st.session_state.confirm_clear_session:
            if st.button("Clear All Session State"):
                st.session_state.confirm_clear_session = True
        else:
            st.warning("This will reset all settings and conversations. Are you sure?")
            col1, col2 = st.columns(2)
            if col1.button("Yes, Clear Session State"):
                # Save some keys we want to preserve
                api_key = st.session_state.get("api_key", "")
                env_file_path = st.session_state.get("env_file_path", "")
                
                # Clear everything
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                    
                # Restore preserved values
                st.session_state.api_key = api_key
                st.session_state.env_file_path = env_file_path
                
                # Add success message and reset confirmation
                st.session_state.confirm_clear_session = False
                st.success("Session state cleared successfully")
                st.rerun()
                
            if col2.button("Cancel"):
                st.session_state.confirm_clear_session = False
                st.rerun()
    
    # Export/Import settings
    with st.expander("Export/Import Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Settings"):
                # Collect settings to export
                settings_to_export = {
                    "debug_flags": st.session_state.get("debug_flags", {}),
                    "model_settings": st.session_state.get("model_settings", {}),
                    "ui_settings": st.session_state.get("ui_settings", {}),
                    "experimental_settings": st.session_state.get("experimental_settings", {}),
                    "mcp_settings": st.session_state.get("mcp_settings", {})
                }
                
                # Convert to JSON
                settings_json = json.dumps(settings_to_export, indent=2)
                
                # Create a download link
                b64_settings = base64.b64encode(settings_json.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64_settings}" download="agent_runner_settings.json">Download Settings JSON</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Import settings
            uploaded_file = st.file_uploader("Import Settings", type="json")
            if uploaded_file is not None:
                try:
                    # Read and parse the file
                    settings_content = uploaded_file.read()
                    imported_settings = json.loads(settings_content)
                    
                    # Apply the imported settings
                    for key, value in imported_settings.items():
                        if key in ["debug_flags", "model_settings", "ui_settings", "experimental_settings", "mcp_settings"]:
                            st.session_state[key] = value
                    
                    st.success("Settings imported successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing settings: {str(e)}")
    
    # Developer tools
    with st.expander("Developer Tools", expanded=False):
        st.markdown("""
        <div class="warning-message">
            These tools are for developer use and may impact application performance or stability.
        </div>
        """, unsafe_allow_html=True)
        
        # Add API request logging
        if "log_api_requests" not in st.session_state:
            st.session_state.log_api_requests = False
            
        st.session_state.log_api_requests = st.checkbox(
            "Log API Requests",
            value=st.session_state.log_api_requests,
            help="Log all API requests to the console (may contain sensitive information)"
        )
        
        # Add token usage tracking
        if "track_token_usage" not in st.session_state:
            st.session_state.track_token_usage = False
            
        st.session_state.track_token_usage = st.checkbox(
            "Track Token Usage",
            value=st.session_state.track_token_usage,
            help="Track and display token usage information"
        )
        
        # Add button to view available environment variables
        if st.button("View Environment Variables"):
            env_vars = {key: value for key, value in os.environ.items() 
                      if not key.startswith("OPENAI_") and not key.startswith("SECRET_")}
            st.json(env_vars)