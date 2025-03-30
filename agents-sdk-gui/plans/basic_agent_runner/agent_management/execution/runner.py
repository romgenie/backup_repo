"""
Core execution logic for running agents.
"""

import traceback
from typing import Dict, Any, List, Optional, Callable
from agents import Runner, RunConfig, RunResult, Agent

from async_utils import async_to_sync
from .tool_handler import create_tool_callback, process_mcp_servers, process_tool_calls

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
    
    # Check and log MCP server details
    if hasattr(agent, 'mcp_servers') and agent.mcp_servers:
        process_mcp_servers(agent.mcp_servers)
    
    # Define and add tool callback for security and monitoring
    tool_callback = create_tool_callback()
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
        
        # Process tool calls in the result
        process_tool_calls(result)
        
        return result
    except Exception as e:
        print(f"ERROR running agent: {str(e)}")
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

def prepare_agent_input(user_input: str, running_agents: Dict[str, RunResult], chat_id: str) -> Any:
    """
    Prepare the agent input with conversation history if available
    
    Args:
        user_input: The user input text
        running_agents: Dictionary of running agents keyed by chat ID
        chat_id: Current chat ID
        
    Returns:
        The prepared input for the agent (string or list)
    """
    # Start with just the user input
    agent_input = user_input
    
    # If we have conversation history, add it
    if chat_id in running_agents:
        previous_result = running_agents[chat_id]
        # Use the to_input_list method to create proper context
        input_list = previous_result.to_input_list()
        
        # Check and correct message format
        for idx, msg in enumerate(input_list):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                print(f"WARNING: Invalid message format at index {idx}, correcting")
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    input_list[idx] = {"role": role, "content": content}
                else:
                    # Skip this item by replacing with empty user message
                    input_list[idx] = {"role": "user", "content": ""}
        
        # Add the current user input to the conversation
        agent_input = input_list + [{"role": "user", "content": user_input}]
    
    return agent_input