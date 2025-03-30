import streamlit as st
import time
import requests
import json
from typing import Dict, Any, List, Optional
from agents import Agent, Runner, RunConfig, RunResult
from openai.types.responses import ResponseTextDeltaEvent

from async_utils import async_to_sync
from ui_components import display_tool_calls

# Import tools from the tools module
from agent_management.tools import get_weather, search_news, get_all_tools

async def _run_agent_async(agent, user_input, max_turns, run_config):
    """
    Async method to run the agent
    
    Args:
        agent: The agent to run
        user_input: The user input or context
        max_turns: Maximum turns for the conversation
        run_config: Configuration for the run
        
    Returns:
        The result of the agent run
    """
    result = await Runner.run(
        agent, 
        input=user_input, 
        max_turns=max_turns,
        run_config=run_config
    )
    return result
    
@async_to_sync
async def _run_agent_sync(agent, user_input, max_turns, run_config):
    """
    Synchronous wrapper for the async agent runner
    
    Args:
        agent: The agent to run
        user_input: The user input or context
        max_turns: Maximum turns for the conversation
        run_config: Configuration for the run
        
    Returns:
        The result of the agent run
    """
    return await _run_agent_async(agent, user_input, max_turns, run_config)

def run_agent_with_history(
    agent: Agent, 
    user_input: str, 
    chat_container: Any, 
    tool_calls_container: Any,
    max_turns: int,
    run_config: RunConfig,
    running_agents: Dict[str, RunResult],
    chat_id: str
) -> Optional[RunResult]:
    """
    Run the agent with conversation history and display results
    
    Args:
        agent: The agent to run
        user_input: The user's input
        chat_container: Streamlit container for chat display
        tool_calls_container: Streamlit container for tool calls display
        max_turns: Maximum turns for the conversation
        run_config: Configuration for the run
        running_agents: Dictionary of running agents keyed by chat ID
        chat_id: Current chat ID
        
    Returns:
        The result of the agent run or None if there was an error
    """
    output_text = ""
    tool_calls = []
    
    # Create placeholder for tool calls
    with tool_calls_container:
        tool_calls_placeholder = st.empty()
    
    # Clear previous outputs
    tool_calls_placeholder.markdown("*No tool calls yet*")
    
    try:
        # Prepare input with conversation history
        agent_input = user_input
        if chat_id in running_agents:
            previous_result = running_agents[chat_id]
            # Use the to_input_list method to create proper context
            agent_input = previous_result.to_input_list() + [{"role": "user", "content": user_input}]
        
        # Call the wrapped async method that creates its own event loop
        result = _run_agent_sync(agent, agent_input, max_turns, run_config)
        
        # Process results
        if result:
            # Display any tool calls
            if hasattr(result, 'steps') and result.steps:
                for step in result.steps:
                    if step.tool_calls:
                        for tool_call in step.tool_calls:
                            tool_calls.append({
                                "name": tool_call.name,
                                "time": time.strftime("%H:%M:%S"),
                                "args": tool_call.args
                            })
                
                # Display tool calls
                display_tool_calls(tool_calls_placeholder, tool_calls)
        
        return result
        
    except Exception as e:
        # Display error in chat container
        with chat_container:
            st.error(f"Error: {str(e)}")
        
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None