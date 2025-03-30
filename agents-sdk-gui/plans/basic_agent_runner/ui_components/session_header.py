"""
UI component for displaying session state information in a header.
"""

import streamlit as st
from typing import Dict, Any, Optional

from services.session_manager import SessionManager

def render_session_header():
    """
    Render a header with key session information.
    """
    # Get debug settings
    debug_settings = st.session_state.get("debug_settings", {})
    show_debug_header = debug_settings.get("show_debug_header", False)
    enable_auto_refresh = debug_settings.get("enable_auto_refresh", True)
    verbose_logging = debug_settings.get("verbose_logging", False)
    
    # Check if we need to run an automatic refresh
    if "last_header_update" not in st.session_state:
        st.session_state.last_header_update = 0
        
    # Add a counter to track when to check for agent
    if "header_counter" not in st.session_state:
        st.session_state.header_counter = 0
    else:
        st.session_state.header_counter += 1
    
    # Add a hidden auto-refresh button - clicks itself periodically
    if "auto_refresh_pressed" not in st.session_state:
        st.session_state.auto_refresh_pressed = False
        
    # Check if we should do a refresh based on counter and auto-refresh setting
    if enable_auto_refresh and st.session_state.header_counter % 3 == 0 and "agent" in st.session_state and not "header_agent_found" in st.session_state:
        # Set flag to avoid infinite refreshes
        st.session_state.header_agent_found = True
        # Directly check session state
        if verbose_logging:
            print("Auto-refreshing header due to agent detection...")
        st.rerun()
    
    # Get agent and conversation info from both session state and SessionManager
    # Check direct session state first
    agent = st.session_state.get("agent")
    # Fall back to SessionManager if not in direct session state
    if agent is None:
        agent = SessionManager.get_agent()
        # If found through SessionManager, update direct session state too
        if agent is not None:
            st.session_state.agent = agent
            
    conversation_history = SessionManager.get_conversation_history()
    msg_count = len(conversation_history)
    
    # Create a container for the header with styling
    with st.container():
        # Standard header view - apply styling through markdown
        st.markdown('<div class="session-header">', unsafe_allow_html=True)
        with st.container():
            cols = st.columns([1, 2, 1])
            
            with cols[0]:
                # Show agent status
                if agent:
                    st.markdown("""
                    <div class="success-message">
                        <strong>Agent Ready</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    # Mark in session state that we've found the agent
                    st.session_state.header_agent_found = True
                else:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("""
                        <div class="warning-message">
                            <strong>No Agent</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        # Small refresh button
                        if st.button("🔄", key="small_header_refresh", help="Refresh agent status"):
                            st.rerun()
            
            with cols[1]:
                # Show agent details if available
                if agent:
                    agent_name = getattr(agent, "name", "Unnamed")
                    agent_model = getattr(agent, "model", "Unknown")
                    tools_count = len(getattr(agent, "tools", [])) if hasattr(agent, "tools") else 0
                    mcp_count = len(getattr(agent, "mcp_servers", [])) if hasattr(agent, "mcp_servers") else 0
                    
                    st.markdown(f"""
                    <div class="info-message text-center">
                        <strong>Agent:</strong> {agent_name} | <strong>Model:</strong> {agent_model} | 
                        <strong>Tools:</strong> {tools_count} | <strong>MCP:</strong> {mcp_count}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-message text-center">
                        <strong>Agent:</strong> Not configured - Go to Configure tab to create an agent
                    </div>
                    """, unsafe_allow_html=True)
            
            with cols[2]:
                # Show conversation status
                if msg_count > 0:
                    st.markdown(f"""
                    <div class="info-message text-center">
                        <strong>Messages:</strong> {msg_count}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="neutral-message text-center">
                        <strong>Messages:</strong> {msg_count}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Close the header div we opened earlier
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Debug header (shown only when enabled)
        if show_debug_header:
            st.divider()
            st.subheader("Debug Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Session State:**")
                st.write(f"Keys in session state: {len(st.session_state)}")
                st.write(f"Agent in session state: {'Yes' if 'agent' in st.session_state else 'No'}")
                st.write(f"Run config in session state: {'Yes' if 'run_config' in st.session_state else 'No'}")
                st.write(f"Conversation messages: {len(conversation_history)}")
                
            with col2:
                st.write("**Debug Settings:**")
                for key, value in st.session_state.get("debug_settings", {}).items():
                    st.write(f"{key}: {value}")
                
            # Add a button to print session state to console
            if st.button("Print Session State", key="header_print_session"):
                SessionManager.print_debug()
                st.success("Session state printed to console")
                    
    # Check if we need to update our knowledge of agent presence
    if enable_auto_refresh and "agent" in st.session_state and agent is None:
        # Agent is in session state but not detected - force refresh
        if verbose_logging:
            print("Force refreshing due to agent in session state but not detected")
        st.rerun()

def render_session_footer():
    """
    Render a footer with debug information.
    """
    # Get debug settings
    debug_settings = st.session_state.get("debug_settings", {})
    show_debug_footer = debug_settings.get("show_debug_footer", False)
    
    # Only render if debug footer is enabled
    if show_debug_footer:
        with st.expander("Session State Debug", expanded=False):
            # Show session state keys in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Session Keys")
                # Display simplified session state
                session_debug = SessionManager.debug()
                st.json(session_debug)
            
            with col2:
                st.subheader("Key Stats")
                
                # Number of session state keys
                st.markdown(f"**Total Keys:** {len(st.session_state)}")
                
                # Conversation info
                messages = len(SessionManager.get_conversation_history())
                st.markdown(f"**Messages:** {messages}")
                
                # Agent info  
                agent = SessionManager.get_agent()
                if agent:
                    st.markdown(f"**Agent Name:** {getattr(agent, 'name', 'Unnamed')}")
                    st.markdown(f"**Agent Model:** {getattr(agent, 'model', 'Unknown')}")
                    
                    # Only show detailed agent info if enabled
                    if debug_settings.get("show_agent_details", False):
                        st.subheader("Agent Details")
                        # Tools
                        if hasattr(agent, 'tools') and agent.tools:
                            st.markdown("**Tools:**")
                            for tool in agent.tools:
                                st.markdown(f"- {getattr(tool, 'name', tool.__class__.__name__)}")
                        
                        # MCP Servers
                        if hasattr(agent, 'mcp_servers') and agent.mcp_servers:
                            st.markdown("**MCP Servers:**")
                            for server in agent.mcp_servers:
                                st.markdown(f"- {getattr(server, 'name', 'Unnamed')}")
                else:
                    st.markdown("**Agent:** Not configured")
                
                # MCP server info
                mcp_servers = SessionManager.get_mcp_servers()
                selected_servers = SessionManager.get_selected_mcp_servers()
                st.markdown(f"**MCP Servers:** {len(mcp_servers)} configured, {len(selected_servers)} selected")
                
                # Debug actions
                col1, col2 = st.columns(2)
                if col1.button("Force Refresh", key="debug_refresh"):
                    st.rerun()
                
                if col2.button("Clear Cache", key="debug_clear_cache"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Cache cleared")
                    st.rerun()
            
            # Show full session state if requested
            if debug_settings.get("show_session_keys", False):
                st.subheader("Full Session State")
                # Show all session state keys
                st.write(st.session_state)