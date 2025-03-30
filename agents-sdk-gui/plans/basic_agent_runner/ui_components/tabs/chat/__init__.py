"""
Chat tab component for Streamlit interface.

This module provides the main tab renderer and integrates the various
sub-components for chat functionality.
"""

import streamlit as st
from typing import Dict, Any, Optional, Callable

from agents import Agent, RunConfig, RunResult

from ui_components.tabs.chat.conversation_display import display_chat_history
from ui_components.tabs.chat.input_panel import render_input_area
from ui_components.tabs.chat.session_controls import render_action_buttons
from ui_components.tabs.chat.tool_integration import render_tool_calls_area
from ui_components.tabs.chat.handlers import handle_chat_interactions

def render_chat_tab(
    agent: Optional[Agent],
    run_config: Optional[RunConfig],
    running_agents: Dict[str, RunResult],
    on_clear_chat: Callable = None
):
    """
    Render the chat tab of the application.
    
    This is the main entry point for the chat tab. It handles agent validation,
    rendering the conversation interface, and managing user interactions.
    
    Args:
        agent: The configured agent or None if not created yet
        run_config: The run configuration or None if not created
        running_agents: Dictionary of running agent results
        on_clear_chat: Callback function when clearing chat
    """
    from services.session_manager import SessionManager
    
    # Check if we already have an agent in session state even if it wasn't passed directly
    if agent is None and SessionManager.has_agent():
        # Use the agent from session state
        agent = SessionManager.get_agent()
        # Also get the run config if available
        if run_config is None:
            run_config = SessionManager.get_run_config()
    
    # Check if we've been redirected from agent creation
    if st.session_state.get("switch_to_chat_tab", False) or st.session_state.get("agent_creating", False):
        print("DEBUG: Detected redirect from agent creation, checking for agent...")
        # If we're switching tabs, ensure we have the latest agent
        # First check direct session state
        if "agent" in st.session_state:
            agent = st.session_state.agent
            # If a run_config is also needed, get it from session state
            if run_config is None and "run_config" in st.session_state:
                run_config = st.session_state.run_config
            print("DEBUG: Found agent in session state during tab switch")
    
    # Check direct session state and ensure refresh happens if needed
    if agent is None and "agent" in st.session_state:
        agent = st.session_state.agent
        # If a run_config is also needed, get it from session state
        if run_config is None and "run_config" in st.session_state:
            run_config = st.session_state.run_config
        print("DEBUG: Found agent in session state, will refresh UI")
        # Force refresh to update UI with the agent information
        st.rerun()
    
    # Display appropriate message or interface based on agent creation status
    if agent is None:
        _render_no_agent_message()
        return
    
    # Add debug information
    _render_debug_information(agent, run_config)
    
    # Display conversation history with a more chat-like appearance
    st.subheader("Conversation")
    
    # Debug check for conversation history
    if "conversation_history" not in st.session_state:
        from agent_management import init_conversation
        init_conversation()
        st.warning("Conversation history was not initialized. Initializing now.")
    
    # Debug info about conversation history
    with st.expander("Debug Conversation History", expanded=False):
        st.write(f"History type: {type(st.session_state.conversation_history)}")
        st.write(f"History length: {len(st.session_state.conversation_history)}")
        st.json(st.session_state.conversation_history)
    
    # Create a container with a fixed height for the chat history to make it scrollable
    chat_container = st.container(height=400)
    
    # Display the chat using our utility function
    display_chat_history(chat_container, st.session_state.conversation_history)
    
    # Status area with more space
    status_placeholder = st.empty()
    
    # Tool calls area (collapsible)
    tool_calls_container = render_tool_calls_area()
    
    # Input area and control buttons
    user_input, send_clicked = render_input_area()
    
    # Action buttons row
    clear_clicked = render_action_buttons()
    
    # Handle user interactions
    handle_chat_interactions(
        agent=agent,
        run_config=run_config,
        running_agents=running_agents,
        user_input=user_input,
        send_clicked=send_clicked,
        clear_clicked=clear_clicked,
        status_placeholder=status_placeholder,
        chat_container=chat_container,
        tool_calls_container=tool_calls_container,
        on_clear_chat=on_clear_chat
    )

def _render_no_agent_message():
    """Display a message when no agent is available."""
    import time
    
    # Use a single column layout without the refresh button
    st.markdown("""
    <div class="info-message">
        Please create an agent in the Configure tab before starting a conversation.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we've just created an agent, in which case we should refresh
    if st.session_state.get("agent_creating", False) or "agent" in st.session_state:
        print("Detected agent_creating flag, refreshing the chat view...")
        # Set a time tag to prevent infinite refreshes
        current_time = int(time.time())
        if "last_chat_refresh" not in st.session_state or current_time - st.session_state.get("last_chat_refresh", 0) > 2:
            st.session_state.last_chat_refresh = current_time
            st.rerun()
    
    # Add a hint about agent creation
    st.markdown("""
    <div class="neutral-message mt-sm">
        <strong>Hint</strong>: After creating an agent in the Configure tab, the chat will automatically update.
    </div>
    """, unsafe_allow_html=True)

def _render_debug_information(agent, run_config):
    """Display debug information about the agent and run configuration."""
    with st.expander("Debug Info", expanded=False):
        st.write("Agent Configuration:")
        st.write(f"- Name: {agent.name}")
        st.write(f"- Model: {agent.model}")
        st.write(f"- Tools: {len(agent.tools) if hasattr(agent, 'tools') and agent.tools else 0}")
        st.write("Run Configuration:")
        if run_config:
            st.write(f"- Tracing: {'Disabled' if run_config.tracing_disabled else 'Enabled'}")
            st.write(f"- Workflow: {run_config.workflow_name}")
            st.write(f"- Max Turns: {st.session_state.get('max_turns', 10)}")
        else:
            st.write("No run configuration available")