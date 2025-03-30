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
from ui_components.tools.custom_tools import get_current_time_custom

# Example function tools that can be used in the application

@function_tool
def get_weather(location: str, units: str) -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and country (e.g., "London,UK")
        units: The units to use (metric or imperial)
        
    Returns:
        A description of the current weather
    """
    # This is a mock implementation - in a real application, you would call a weather API
    
    # Default to metric if units value is not valid
    if not units or (units != "metric" and units != "imperial"):
        print(f"DEBUG: Invalid units value '{units}', defaulting to metric")
        units = "metric"
        
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temperature = 20 if units == "metric" else 68
    
    # Handle empty location
    if not location:
        location = "Unknown location"
        
    condition = weather_conditions[hash(location + str(datetime.date.today())) % len(weather_conditions)]
    
    return f"The weather in {location} is currently {condition} with a temperature of {temperature}°{'C' if units == 'metric' else 'F'}"

@function_tool
def search_news(query: str, max_results: int) -> str:
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
    
    # Ensure max_results is an integer (since we removed the default)
    try:
        max_results_int = int(max_results)
    except (ValueError, TypeError):
        max_results_int = 3  # Fallback if conversion fails
        
    results = mock_results[:min(max_results_int, len(mock_results))]
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
    print(f"Running agent with input: {user_input[:100] if isinstance(user_input, str) else 'complex input'}...")
    print(f"Agent configuration: model={agent.model}")
    
    # Check if agent has MCP servers
    if hasattr(agent, 'mcp_servers') and agent.mcp_servers:
        print(f"Agent has {len(agent.mcp_servers)} MCP servers")
        # Log details about each MCP server
        for i, server in enumerate(agent.mcp_servers):
            print(f"MCP Server {i+1} details:")
            print(f"  Type: {type(server).__name__}")
            print(f"  Name: {getattr(server, 'name', 'Unnamed')}")
            # Check if the server has required methods
            has_list_tools = hasattr(server, 'list_tools') and callable(getattr(server, 'list_tools'))
            has_call_tool = hasattr(server, 'call_tool') and callable(getattr(server, 'call_tool'))
            print(f"  Has list_tools: {has_list_tools}")
            print(f"  Has call_tool: {has_call_tool}")
            
            # Try to list tools
            if has_list_tools:
                try:
                    tools = await server.list_tools()
                    print(f"  Available tools: {[t.name for t in tools]}")
                    # Print details about each tool
                    for tool in tools:
                        print(f"    Tool: {tool.name}")
                        print(f"    Description: {getattr(tool, 'description', 'No description')}")
                        params = getattr(tool, 'parameters', [])
                        print(f"    Parameters: {len(params)} total")
                        for param in params:
                            print(f"      - {param.get('name', 'unnamed')}: {param.get('description', 'No description')}")
                except Exception as e:
                    print(f"  Error listing tools: {str(e)}")
    
    # Define callback to log tool calls
    async def tool_callback(event):
        if event.get('type') == 'tool_call':
            tool_name = event.get('name', 'unknown')
            tool_args = event.get('args', {})
            print(f"TOOL CALL: {tool_name}")
            print(f"TOOL ARGS: {tool_args}")
        elif event.get('type') == 'tool_result':
            tool_name = event.get('name', 'unknown')
            tool_result = event.get('result', None)
            print(f"TOOL RESULT: {tool_name}")
            print(f"RESULT VALUE: {tool_result}")
            print(f"RESULT TYPE: {type(tool_result).__name__}")
    
    # Add the callback to run_config if possible
    if hasattr(run_config, 'add_event_callback'):
        run_config.add_event_callback(tool_callback)
    
    try:    
        result = await Runner.run(
            agent, 
            input=user_input, 
            max_turns=max_turns,
            run_config=run_config
        )
        
        print(f"Agent run completed. Result type: {type(result)}")
        
        # Log details about the result
        print("DEBUG - Result object properties:")
        for attr_name in dir(result):
            if not attr_name.startswith('_'):  # Skip private attributes
                try:
                    attr_value = getattr(result, attr_name)
                    if not callable(attr_value):  # Skip methods
                        print(f"  {attr_name}: <class '{type(attr_value).__name__}'> = {str(attr_value)[:20]}...")
                except Exception as e:
                    print(f"  Error accessing {attr_name}: {str(e)}")
        
        # Check for tool calls in result
        if hasattr(result, 'steps'):
            print("DEBUG - Analyzing steps in result:")
            for i, step in enumerate(result.steps):
                print(f"Step {i+1}:")
                
                # Check for regular tool calls
                if hasattr(step, 'tool_calls') and step.tool_calls:
                    print(f"  Regular tool calls: {len(step.tool_calls)}")
                    for tool_call in step.tool_calls:
                        print(f"    Tool: {tool_call.name}")
                        print(f"    Args: {tool_call.args}")
                        if hasattr(tool_call, 'response'):
                            print(f"    Response: {tool_call.response}")
                
                # Check for MCP tool calls
                if hasattr(step, 'mcp_tool_calls') and step.mcp_tool_calls:
                    print(f"  MCP tool calls: {len(step.mcp_tool_calls)}")
                    for mcp_call in step.mcp_tool_calls:
                        print(f"    MCP Tool: {mcp_call.name}")
                        print(f"    Server: {getattr(mcp_call, 'server_name', 'Unknown')}")
                        print(f"    Args: {getattr(mcp_call, 'arguments', {})}")
                        if hasattr(mcp_call, 'result'):
                            print(f"    Result: {mcp_call.result}")
                            print(f"    Result type: {type(mcp_call.result).__name__}")
                        elif hasattr(mcp_call, 'error'):
                            print(f"    Error: {mcp_call.error}")
                        else:
                            # List all attributes
                            print("    All attributes:")
                            for attr in dir(mcp_call):
                                if not attr.startswith('_'):
                                    try:
                                        val = getattr(mcp_call, attr)
                                        if not callable(val):
                                            print(f"      {attr}: {val}")
                                    except:
                                        pass
        
        return result
    except Exception as e:
        print(f"ERROR running agent: {str(e)}")
        import traceback
        print(f"ERROR TRACE: {traceback.format_exc()}")
        raise
    
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
        
        # Debug: check MCP servers in the agent
        if hasattr(agent, 'mcp_servers') and agent.mcp_servers:
            print(f"Running agent with {len(agent.mcp_servers)} MCP servers")
            for server in agent.mcp_servers:
                print(f"  Server: {getattr(server, 'name', 'Unnamed')}")
                # Try to get tools from the server
                try:
                    # Use a separate event loop to avoid nesting issues
                    import asyncio
                    tools_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(tools_loop)
                    tools = tools_loop.run_until_complete(server.list_tools())
                    print(f"  Available MCP tools: {[t.name for t in tools]}")
                    tools_loop.close()
                except Exception as e:
                    print(f"  Error listing MCP tools: {str(e)}")
        else:
            print("No MCP servers available for the agent")
        
        # Call the wrapped async method that creates its own event loop
        print("\n=== Starting agent execution with the following configuration ===")
        print(f"Agent model: {agent.model}")
        print(f"Max turns: {max_turns}")
        print(f"Using {len(agent.tools) if hasattr(agent, 'tools') and agent.tools else 0} standard tools")
        
        if hasattr(agent, 'mcp_servers') and agent.mcp_servers:
            print(f"Using {len(agent.mcp_servers)} MCP servers:")
            for i, server in enumerate(agent.mcp_servers):
                print(f"  Server {i+1}: {getattr(server, 'name', 'Unnamed')}")
        else:
            print("No MCP servers available")
            
        print("=======================================================")
        
        result = _run_agent_sync(agent, agent_input, max_turns, run_config)
        
        # Process results
        if result:
            # Display any tool calls
            if hasattr(result, 'steps') and result.steps:
                for step in result.steps:
                    # Regular tool calls
                    if hasattr(step, 'tool_calls') and step.tool_calls:
                        for tool_call in step.tool_calls:
                            # Create tool info dictionary
                            tool_info = {
                                "name": tool_call.name,
                                "time": time.strftime("%H:%M:%S"),
                                "args": tool_call.args,
                                "is_mcp": False
                            }
                            
                            # Add response if available
                            if hasattr(tool_call, 'response'):
                                tool_info["result"] = tool_call.response
                            
                            tool_calls.append(tool_info)
                    
                    # MCP tool calls
                    if hasattr(step, 'mcp_tool_calls') and step.mcp_tool_calls:
                        for mcp_call in step.mcp_tool_calls:
                            # Start with basic tool info
                            mcp_tool = {
                                "name": mcp_call.name,
                                "server": getattr(mcp_call, 'server_name', "MCP Server"),
                                "time": time.strftime("%H:%M:%S"),
                                "args": getattr(mcp_call, 'arguments', {}),
                                "is_mcp": True,
                                "status": "unknown"  # Default status
                            }
                            
                            # Handle different result scenarios
                            try:
                                # Check for error first
                                if hasattr(mcp_call, 'error') and mcp_call.error:
                                    mcp_tool["error"] = str(mcp_call.error)
                                    mcp_tool["status"] = "error"
                                    print(f"MCP tool error: {mcp_call.name} - {mcp_call.error}")
                                # Then check for result
                                elif hasattr(mcp_call, 'result'):
                                    result = mcp_call.result
                                    mcp_tool["result"] = result
                                    mcp_tool["status"] = "success"
                                    
                                    # More detailed debug info
                                    result_type = type(result).__name__
                                    result_repr = repr(result)
                                    print(f"MCP tool success: {mcp_call.name}")
                                    print(f"Result type: {result_type}")
                                    print(f"Result repr: {result_repr}")
                                    
                                    # Additional time tool specific debug
                                    if mcp_call.name == "get_current_time":
                                        print("TIME TOOL DEBUGGING:")
                                        # Check all attributes of mcp_call
                                        for attr_name in dir(mcp_call):
                                            if not attr_name.startswith('_'):  # Skip private attributes
                                                try:
                                                    attr_value = getattr(mcp_call, attr_name)
                                                    if not callable(attr_value):  # Skip methods
                                                        print(f"  {attr_name}: {repr(attr_value)}")
                                                except Exception as attr_err:
                                                    print(f"  Error getting {attr_name}: {str(attr_err)}")
                                else:
                                    # No result and no error - unclear what happened
                                    mcp_tool["status"] = "no_result"
                                    print(f"MCP tool {mcp_call.name} executed but returned no result or error")
                                    
                                    # More debugging - check for other attributes that might contain data
                                    print(f"MCP call dir: {dir(mcp_call)}")
                                    # Try to check response
                                    if hasattr(mcp_call, 'response'):
                                        print(f"Found response attribute: {repr(mcp_call.response)}")
                                        mcp_tool["result"] = mcp_call.response
                                    # Try to check output
                                    if hasattr(mcp_call, 'output'):
                                        print(f"Found output attribute: {repr(mcp_call.output)}")
                                        mcp_tool["result"] = mcp_call.output
                            except Exception as e:
                                # Something went wrong when processing the result
                                mcp_tool["error"] = f"Error processing tool result: {str(e)}"
                                mcp_tool["status"] = "processing_error"
                                import traceback
                                mcp_tool["stack"] = traceback.format_exc()
                                print(f"Error processing MCP tool result: {str(e)}")
                                
                            tool_calls.append(mcp_tool)
                
                # Display tool calls in the tool calls container
                display_tool_calls(tool_calls_placeholder, tool_calls)
                
                # Important: Tool calls are now shown inline in the message through the chat_display.py changes
        
        return result
        
    except Exception as e:
        # Display error in chat container
        with chat_container:
            st.error(f"Error: {str(e)}")
        
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None