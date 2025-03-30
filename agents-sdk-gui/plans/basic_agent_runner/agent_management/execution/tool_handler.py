"""
Tool call processing functionality for agent executions.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable

from agent_management.security import (
    validate_timezone, validate_time_string, sanitize_mcp_parameters, log_security_event
)

async def create_tool_callback():
    """
    Create a callback function to handle tool calls for monitoring and security
    
    Returns:
        Callback function for monitoring tool calls
    """
    async def tool_callback(event):
        """Event callback for tool calls and results"""
        if event.get('type') == 'tool_call':
            tool_name = event.get('name', 'unknown')
            tool_args = event.get('args', {})
            
            # Apply security validation for MCP tools
            if tool_name in ["get_current_time", "convert_time"]:
                sanitized_args = sanitize_mcp_parameters(tool_name, tool_args)
                if sanitized_args != tool_args:
                    print(f"SECURITY: Sanitized MCP tool args for {tool_name}")
                    print(f"  Original: {tool_args}")
                    print(f"  Sanitized: {sanitized_args}")
                    # Update the event with sanitized args
                    event['args'] = sanitized_args
                    tool_args = sanitized_args
            
            print(f"TOOL CALL: {tool_name}")
            print(f"TOOL ARGS: {tool_args}")
            
        elif event.get('type') == 'tool_result':
            tool_name = event.get('name', 'unknown')
            tool_result = event.get('result', None)
            print(f"TOOL RESULT: {tool_name}")
            print(f"RESULT VALUE: {tool_result}")
            print(f"RESULT TYPE: {type(tool_result).__name__}")
    
    return tool_callback

def process_mcp_servers(mcp_servers):
    """
    Process and log MCP server details
    
    Args:
        mcp_servers: List of MCP server instances
    """
    print(f"Agent has {len(mcp_servers)} MCP servers")
    
    # Log details about each MCP server
    for i, server in enumerate(mcp_servers):
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
                # Use a separate event loop to avoid nesting issues
                tools_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(tools_loop)
                tools = tools_loop.run_until_complete(server.list_tools())
                tools_loop.close()
                
                print(f"  Available tools: {[t.name for t in tools]}")
                # Print details about each tool
                for tool in tools:
                    print(f"    Tool: {tool.name}")
                    print(f"    Description: {getattr(tool, 'description', 'No description')}")
                    
                    # Handle known parameter inconsistencies
                    params = getattr(tool, 'parameters', [])
                    
                    # Special handling for known tools with parameter reporting issues
                    if tool.name == "get_current_time" and not params:
                        # Add expected parameters for get_current_time
                        params = [
                            {"name": "timezone", "description": "Timezone (e.g., 'America/New_York', 'Europe/London')", "type": "string", "required": False}
                        ]
                        print(f"    Parameters: {len(params)} total (added missing parameter definitions)")
                    elif tool.name == "convert_time" and not params:
                        # Add expected parameters for convert_time
                        params = [
                            {"name": "time", "description": "Time to convert (ISO format or natural language)", "type": "string", "required": True},
                            {"name": "from_timezone", "description": "Source timezone", "type": "string", "required": True},
                            {"name": "to_timezone", "description": "Target timezone", "type": "string", "required": True}
                        ]
                        print(f"    Parameters: {len(params)} total (added missing parameter definitions)")
                    else:
                        print(f"    Parameters: {len(params)} total")
                        
                    # Display all parameters
                    for param in params:
                        print(f"      - {param.get('name', 'unnamed')}: {param.get('description', 'No description')}")
            except Exception as e:
                print(f"  Error listing tools: {str(e)}")

def process_tool_calls(result):
    """
    Process and log tool calls from an agent run result
    
    Args:
        result: The agent run result object
    """
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
                    
                    # Get the original arguments
                    original_args = getattr(mcp_call, 'arguments', {})
                    print(f"    Args: {original_args}")
                    
                    # Apply security sanitization to arguments
                    if hasattr(mcp_call, 'arguments'):
                        # Sanitize the arguments and check if they were modified
                        sanitized_args = sanitize_mcp_parameters(mcp_call.name, mcp_call.arguments)
                        if sanitized_args != mcp_call.arguments:
                            print(f"    SECURITY: Sanitized MCP tool arguments")
                            print(f"      Original: {mcp_call.arguments}")
                            print(f"      Sanitized: {sanitized_args}")
                            # Update the arguments with sanitized values
                            mcp_call.arguments = sanitized_args
                    
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