"""
Agent execution module providing core functionality for running agents.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from agents import Agent, RunConfig, RunResult

from .runner import _run_agent_sync, prepare_agent_input
from .result_processor import process_agent_result
from .example_tools import get_weather, search_news

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
    tool_calls = []
    
    # Create placeholder for tool calls
    with tool_calls_container:
        tool_calls_placeholder = st.empty()
    
    # Clear previous outputs
    tool_calls_placeholder.markdown("*No tool calls yet*")
    
    try:
        # Prepare input with conversation history
        agent_input = prepare_agent_input(user_input, running_agents, chat_id)
        
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
        
        # Start agent execution with debug information
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
        
        # Run the agent
        result = _run_agent_sync(agent, agent_input, max_turns, run_config)
        
        # Process results and extract tool calls
        if result:
            process_agent_result(result, tool_calls_placeholder)
        
        return result
        
    except Exception as e:
        # Display error in chat container
        with chat_container:
            st.error(f"Error: {str(e)}")
        
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None
        
__all__ = ['run_agent_with_history', 'get_weather', 'search_news']