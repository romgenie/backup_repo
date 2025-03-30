"""
UI component for displaying session state debugging information.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import json

def render_session_debug_footer():
    """
    Render a collapsible footer with session state debugging information.
    """
    st.markdown("---")
    with st.expander("Session State Debugger", expanded=False):
        col1, col2 = st.columns([1, 1])
        
        # Display basic session info
        with col1:
            st.subheader("Session Info")
            st.write(f"Keys in session state: {len(st.session_state)}")
            
            # Create a filtered view of session state that's easier to read
            simple_state = {}
            for key, value in st.session_state.items():
                if key == "conversation_history":
                    simple_state[key] = f"List with {len(value)} messages"
                elif key == "agent":
                    agent_name = getattr(value, "name", "Unnamed") if value else "None"
                    agent_model = getattr(value, "model", "Unknown") if value else "None"
                    simple_state[key] = f"Agent(name='{agent_name}', model='{agent_model}')"
                elif key == "run_config":
                    if value:
                        workflow = getattr(value, "workflow_name", "Unknown")
                        tracing = "Disabled" if getattr(value, "tracing_disabled", False) else "Enabled"
                        simple_state[key] = f"RunConfig(workflow='{workflow}', tracing='{tracing}')"
                    else:
                        simple_state[key] = "None"
                elif key == "mcp_servers":
                    simple_state[key] = f"Dict with {len(value)} servers"
                elif key == "agent_mcp_servers":
                    simple_state[key] = f"List with {len(value) if value else 0} servers"
                elif key == "running_agents":
                    simple_state[key] = f"Dict with {len(value) if value else 0} agents"
                elif isinstance(value, (list, dict, set)):
                    simple_state[key] = f"{type(value).__name__} with {len(value)} items"
                elif callable(value):
                    simple_state[key] = f"Function: {value.__name__ if hasattr(value, '__name__') else 'Anonymous'}"
                else:
                    # For other types, show a preview
                    value_str = str(value)
                    simple_state[key] = f"{value_str[:30]}..." if len(value_str) > 30 else value_str
            
            # Display the simplified session state
            st.json(simple_state)
        
        with col2:
            st.subheader("Session Actions")
            
            # Chat-related info
            st.markdown("**Conversation State**")
            st.write(f"Messages: {len(st.session_state.get('conversation_history', []))}")
            st.write(f"Chat ID: {st.session_state.get('current_chat_id', 'None')}")
            
            # Agent-related info
            st.markdown("**Agent State**")
            agent = st.session_state.get("agent")
            if agent:
                st.write(f"Agent: {agent.name} ({agent.model})")
                st.write(f"Tools: {len(agent.tools) if hasattr(agent, 'tools') else 0}")
                st.write(f"MCP Servers: {len(agent.mcp_servers) if hasattr(agent, 'mcp_servers') else 0}")
            else:
                st.write("No agent configured")
            
            # Status flags
            st.markdown("**Status Flags**")
            flags = {k: v for k, v in st.session_state.items() if isinstance(v, bool)}
            if flags:
                st.json(flags)
            else:
                st.write("No boolean flags set")
            
            # Action buttons
            st.markdown("**Debug Actions**")
            if st.button("Clear Conversation", key="debug_clear_conversation"):
                st.session_state.conversation_history = []
                st.rerun()
            
            if st.button("Force Reload", key="debug_force_reload"):
                st.rerun()

def render_session_header():
    """
    Render a header with key session information.
    """
    agent = st.session_state.get("agent")
    
    # Create a container for the header
    with st.container():
        cols = st.columns([1, 3, 1])
        
        with cols[0]:
            # Show agent status
            if agent:
                st.success("Agent Ready")
            else:
                st.error("No Agent")
        
        with cols[1]:
            # Show agent details if available
            if agent:
                st.markdown(f"**Agent:** {agent.name} ({agent.model})")
            else:
                st.markdown("**Agent:** Not configured")
        
        with cols[2]:
            # Show conversation status
            msg_count = len(st.session_state.get('conversation_history', []))
            st.markdown(f"**Messages:** {msg_count}")
    
    # Add a divider
    st.markdown("---")