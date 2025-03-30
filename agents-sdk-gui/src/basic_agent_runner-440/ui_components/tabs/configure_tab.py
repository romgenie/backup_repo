import streamlit as st
from typing import Dict, Any, Callable, Optional

from agents import Agent, RunConfig

from ui_components.config import (
    render_model_settings,
    render_run_settings,
    handle_agent_creation,
    display_agent_info
)

def render_configure_tab(
    api_key: str,
    env_file_path: str,
    on_agent_create: Callable[[Agent, RunConfig], None],
    on_api_key_save: Callable[[str], str],
    on_api_key_load: Callable[[str], tuple]
) -> None:
    """
    Render the configure tab of the application
    
    Args:
        api_key: Current API key
        env_file_path: Path to the .env file
        on_agent_create: Callback when agent is created
        on_api_key_save: Callback to save API key
        on_api_key_load: Callback to load API key
    """
    st.header("Configure Agent")
    
    # Agent creation section
    agent_name = st.text_input("Agent Name", "Assistant")
    
    default_instructions = """You are a helpful assistant with access to tools.

When asked about things like weather, news, or other information that might benefit from using tools, you should use the appropriate tool rather than saying you don't know.

For example:
- If asked about weather, use the get_weather tool
- If asked about news, use the search_news tool

Always provide thoughtful, accurate responses."""

    agent_instructions = st.text_area(
        "Instructions", 
        default_instructions,
        height=150,
        help="Provide detailed instructions for the agent's behavior"
    )
    
    # Model settings section
    model_name, temperature, max_tokens = render_model_settings()
    
    # Run settings section
    max_turns, enable_tracing = render_run_settings()
    
    # Agent creation handler
    handle_agent_creation(
        agent_name=agent_name,
        agent_instructions=agent_instructions,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_turns=max_turns,
        enable_tracing=enable_tracing,
        api_key=api_key,
        on_agent_create=on_agent_create
    )
    
    # If agent is already created, show its configuration
    if "agent" in st.session_state:
        display_agent_info(st.session_state.agent)