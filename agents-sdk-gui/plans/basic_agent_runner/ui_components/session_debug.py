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
    st.markdown('<hr class="mt-lg mb-lg" />', unsafe_allow_html=True)
    
    with st.expander("Session State Debugger", expanded=False):
        st.markdown("""
        <div class="form-section">
            <h3>Debug Tools</h3>
            <div class="text-sm text-neutral-600 mb-xs">Information and actions for debugging</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        # Display basic session info
        with col1:
            st.markdown("""
            <div class="form-section">
                <h4 class="text-primary font-semibold">Session Info</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="neutral-message">
                <strong>Keys in session state:</strong> {len(st.session_state)}
            </div>
            """, unsafe_allow_html=True)
            
            # Create a filtered view of session state that's easier to read
            simple_state = {}
            # Create a copy of session state items to avoid modification during iteration
            session_state_items = list(st.session_state.items())
            for key, value in session_state_items:
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
            st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
            st.json(simple_state)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="form-section">
                <h4 class="text-primary font-semibold">Session Actions</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Chat-related info
            st.markdown('<div class="mb-sm">', unsafe_allow_html=True)
            st.markdown('<div class="font-semibold">Conversation State</div>', unsafe_allow_html=True)
            
            message_count = len(st.session_state.get('conversation_history', []))
            chat_id = st.session_state.get('current_chat_id', 'None')
            
            st.markdown(f"""
            <div class="text-sm mb-xs">
                <span class="font-medium">Messages:</span> {message_count}
            </div>
            <div class="text-sm mb-xs">
                <span class="font-medium">Chat ID:</span> {chat_id}
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Agent-related info
            st.markdown('<div class="mb-sm">', unsafe_allow_html=True)
            st.markdown('<div class="font-semibold">Agent State</div>', unsafe_allow_html=True)
            
            agent = st.session_state.get("agent")
            if agent:
                tool_count = len(agent.tools) if hasattr(agent, 'tools') else 0
                server_count = len(agent.mcp_servers) if hasattr(agent, 'mcp_servers') else 0
                
                st.markdown(f"""
                <div class="text-sm mb-xs">
                    <span class="font-medium">Agent:</span> {agent.name} ({agent.model})
                </div>
                <div class="text-sm mb-xs">
                    <span class="font-medium">Tools:</span> {tool_count}
                </div>
                <div class="text-sm mb-xs">
                    <span class="font-medium">MCP Servers:</span> {server_count}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="neutral-message">
                    No agent configured
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Status flags
            st.markdown('<div class="mb-sm">', unsafe_allow_html=True)
            st.markdown('<div class="font-semibold">Status Flags</div>', unsafe_allow_html=True)
            
            # Create a stable copy of session state to avoid modification during iteration
            session_state_copy = dict(st.session_state)
            flags = {k: v for k, v in session_state_copy.items() if isinstance(v, bool)}
            if flags:
                st.json(flags)
            else:
                st.markdown("""
                <div class="neutral-message">
                    No boolean flags set
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Action buttons
            st.markdown('<div class="mb-sm">', unsafe_allow_html=True)
            st.markdown('<div class="font-semibold">Debug Actions</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Conversation", key="debug_clear_conversation", 
                           help="Clear all messages in the current conversation"):
                    st.session_state.conversation_history = []
                    st.rerun()
            
            with col2:
                if st.button("Force Reload", key="debug_force_reload",
                           help="Force a complete UI refresh"):
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

def render_session_header():
    """
    Render a header with key session information.
    """
    agent = st.session_state.get("agent")
    
    # Create a container for the header
    st.markdown('<div class="session-header">', unsafe_allow_html=True)
    with st.container():
        cols = st.columns([1, 3, 1])
        
        with cols[0]:
            # Show agent status with styled badges
            if agent:
                st.markdown("""
                <div class="success-message p-xs">
                    <span class="font-semibold">Agent Ready</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-message p-xs">
                    <span class="font-semibold">No Agent</span>
                </div>
                """, unsafe_allow_html=True)
        
        with cols[1]:
            # Show agent details if available
            if agent:
                st.markdown(f"""
                <div class="text-center">
                    <span class="font-semibold">Agent:</span> {agent.name} 
                    <span class="text-sm text-neutral-600">({agent.model})</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="text-center">
                    <span class="font-semibold">Agent:</span> 
                    <span class="text-neutral-600">Not configured</span>
                </div>
                """, unsafe_allow_html=True)
        
        with cols[2]:
            # Show conversation status
            msg_count = len(st.session_state.get('conversation_history', []))
            message_badge = f"""
            <span style="
                background-color: var(--primary-light); 
                color: white; 
                padding: 2px 6px; 
                border-radius: var(--radius-md); 
                font-size: var(--font-xs); 
                margin-left: var(--space-xs);
                font-weight: var(--font-weight-medium);
            ">{msg_count}</span>
            """
            st.markdown(f"""
            <div class="text-right">
                <span class="font-semibold">Messages:</span> 
                {message_badge}
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a divider with proper spacing
    st.markdown('<hr class="mt-xs mb-md" />', unsafe_allow_html=True)