import streamlit as st
from typing import Dict, Any, Optional, Callable

from agents import Agent, RunConfig, RunResult

from ui_components import display_chat
from ui_components.chat import (
    render_input_area,
    render_tool_calls_area,
    render_action_buttons,
    handle_chat_interactions
)

from agent_management import clear_conversation

def render_chat_tab(
    agent: Optional[Agent],
    run_config: Optional[RunConfig],
    running_agents: Dict[str, RunResult],
    on_clear_chat: Callable = None
):
    """
    Render the chat tab of the application
    
    Args:
        agent: The configured agent or None if not created yet
        run_config: The run configuration or None if not created
        running_agents: Dictionary of running agent results
        on_clear_chat: Callback function when clearing chat
    """
    # Display appropriate message or interface based on agent creation status
    if agent is None:
        st.info("Please create an agent in the Configure tab before starting a conversation.")
        return
    
    # Display conversation history with a more chat-like appearance
    st.subheader("Conversation")
    
    # Check if conversation_history exists
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Create a container with a fixed height for the chat history to make it scrollable
    chat_container = st.container(height=400)
    
    # Display the chat using our utility function
    display_chat(chat_container, st.session_state.conversation_history)
    
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