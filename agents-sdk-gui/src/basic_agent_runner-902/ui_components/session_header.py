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
        
    # Check if we should do a refresh based on counter
    if st.session_state.header_counter % 3 == 0 and "agent" in st.session_state and not "header_agent_found" in st.session_state:
        # Set flag to avoid infinite refreshes
        st.session_state.header_agent_found = True
        # Directly check session state
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
    
    # Create a container for the header with a light background 
    with st.container():
        st.markdown("""
        <style>
        .header-container {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            cols = st.columns([1, 2, 1])
            
            with cols[0]:
                # Show agent status
                if agent:
                    st.success("Agent Ready")
                    # Mark in session state that we've found the agent
                    st.session_state.header_agent_found = True
                else:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.warning("No Agent")
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
                    <div style='text-align: center;'>
                        <b>Agent:</b> {agent_name} | <b>Model:</b> {agent_model} | <b>Tools:</b> {tools_count} | <b>MCP:</b> {mcp_count}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='text-align: center;'>
                        <b>Agent:</b> Not configured - Go to Configure tab to create an agent
                    </div>
                    """, unsafe_allow_html=True)
            
            with cols[2]:
                # Show conversation status
                if msg_count > 0:
                    st.info(f"Messages: {msg_count}")
                else:
                    st.markdown(f"Messages: {msg_count}")
                    
    # Check if we need to update our knowledge of agent presence
    if "agent" in st.session_state and agent is None:
        # Agent is in session state but not detected - force refresh
        st.rerun()

def render_session_footer():
    """
    Render a footer with debug information.
    """
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
            else:
                st.markdown("**Agent:** Not configured")
            
            # MCP server info
            mcp_servers = SessionManager.get_mcp_servers()
            selected_servers = SessionManager.get_selected_mcp_servers()
            st.markdown(f"**MCP Servers:** {len(mcp_servers)} configured, {len(selected_servers)} selected")
            
            # Debug actions
            if st.button("Force Refresh", key="debug_refresh"):
                st.rerun()