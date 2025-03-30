import streamlit as st
import time
import requests
import datetime
import json
from typing import Dict, Any, List, Optional
from agents import Agent, Runner, RunConfig, RunResult, function_tool
from openai.types.responses import ResponseTextDeltaEvent

from async_utils import async_to_sync
from ui_components import display_tool_calls

# Example function tools that can be used in the application

@function_tool
def get_weather(location: str, units: Optional[str] = None) -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and country (e.g., "London,UK")
        units: The units to use (metric or imperial)
        
    Returns:
        A description of the current weather
    """
    # This is a mock implementation - in a real application, you would call a weather API
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    # Default to metric if not specified
    units = units or "metric"
    temperature = 20 if units == "metric" else 68
    condition = weather_conditions[hash(location + str(datetime.date.today())) % len(weather_conditions)]
    
    return f"The weather in {location} is currently {condition} with a temperature of {temperature}°{'C' if units == 'metric' else 'F'}"

@function_tool
def search_news(query: str, max_results: int = 3) -> str:
    """
    Search for news articles matching a query.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (1-10)
        
    Returns:
        A list of news headlines and snippets
    """
    # This is a mock implementation - in a real application, you would call a news API
    mock_results = [
        {"title": f"Latest developments in {query}", "snippet": f"New research shows promising results related to {query}..."},
        {"title": f"Analysis: The impact of {query} on industry", "snippet": f"Experts weigh in on how {query} is changing the landscape..."},
        {"title": f"Interview with {query} specialist", "snippet": f"We spoke with leading researchers about their work on {query}..."},
        {"title": f"{query} breakthrough announced", "snippet": f"A major discovery in the field of {query} was announced today..."},
    ]
    
    results = mock_results[:min(max_results, len(mock_results))]
    formatted_results = "\n\n".join([f"**{r['title']}**\n{r['snippet']}" for r in results])
    
    return f"Found {len(results)} results for '{query}':\n\n{formatted_results}"

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