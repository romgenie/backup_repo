"""
Result processing functionality for agent executions.
"""

import time
import json
import datetime
import re
import traceback
from typing import Dict, Any, List, Optional
import streamlit as st
from agents import RunResult

from ui_components import display_tool_calls
from agent_management.security import (
    validate_timezone, validate_time_string, sanitize_mcp_parameters, log_security_event
)

def process_agent_result(
    result: RunResult,
    tool_calls_placeholder: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Process an agent run result and extract tool calls
    
    Args:
        result: The agent run result
        tool_calls_placeholder: Optional Streamlit placeholder for displaying tool calls
        
    Returns:
        List of processed tool calls
    """
    tool_calls = []
    
    if not result:
        return tool_calls
        
    # Debug information about the result
    print("DEBUG: Successfully received result from agent")
    print(f"Result type: {type(result).__name__}")
    
    # Check for conversation content
    if hasattr(result, 'final_output'):
        print(f"Final output: {result.final_output[:100]}...")
    
    if hasattr(result, 'conversation_history') and result.conversation_history:
        print(f"Conversation history in result: {len(result.conversation_history)} messages")
    else:
        print("No conversation history in result")
        
    # Extract tool calls from the result
    if hasattr(result, 'steps') and result.steps:
        print(f"Result has {len(result.steps)} steps")
        tool_calls = extract_tool_calls(result.steps)
        
        # Display tool calls if a placeholder was provided
        if tool_calls_placeholder and tool_calls:
            display_tool_calls(tool_calls_placeholder, tool_calls)
    
    return tool_calls

def extract_tool_calls(steps: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract and process tool calls from result steps
    
    Args:
        steps: List of steps from the result
        
    Returns:
        List of processed tool call dictionaries
    """
    tool_calls = []
    
    for step in steps:
        # Process regular tool calls
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
        
        # Process MCP tool calls
        if hasattr(step, 'mcp_tool_calls') and step.mcp_tool_calls:
            for mcp_call in step.mcp_tool_calls:
                # Process each MCP tool call
                mcp_tool = process_mcp_tool_call(mcp_call)
                tool_calls.append(mcp_tool)
    
    return tool_calls

def process_mcp_tool_call(mcp_call: Any) -> Dict[str, Any]:
    """
    Process an MCP tool call and handle special cases
    
    Args:
        mcp_call: The MCP tool call object
        
    Returns:
        Dictionary with processed tool information
    """
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
            
            # Special case handling for time tools to improve output
            if mcp_call.name == "get_current_time":
                mcp_tool = process_time_tool_result(mcp_call, mcp_tool)
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
        mcp_tool["stack"] = traceback.format_exc()
        print(f"Error processing MCP tool result: {str(e)}")
    
    return mcp_tool

def process_time_tool_result(mcp_call: Any, mcp_tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Special processing for time tool results
    
    Args:
        mcp_call: The MCP tool call object
        mcp_tool: The tool dictionary to update
        
    Returns:
        Updated tool dictionary
    """
    print("TIME TOOL DEBUGGING:")
    # Log original parameters for security analysis
    original_args = getattr(mcp_call, 'arguments', {})
    print(f"Original time tool arguments: {original_args}")
    
    result = mcp_call.result
    
    # Try to ensure the result is properly structured
    if isinstance(result, str) and result.strip().startswith('{'):
        try:
            # Parse JSON string result
            parsed_result = json.loads(result)
            mcp_tool["result"] = parsed_result
            print(f"Parsed get_current_time JSON result: {parsed_result}")
        except Exception as json_err:
            print(f"Error parsing JSON: {json_err}")
    
    # If result is not a dict yet, try to structure it
    if not isinstance(mcp_tool["result"], dict):
        try:
            # Create a structured result
            # Extract timezone if present and validate it
            timezone = "UTC"  # Default to UTC for safety
            if "arguments" in dir(mcp_call):
                args = mcp_call.arguments
                if isinstance(args, dict) and "timezone" in args:
                    raw_timezone = args["timezone"]
                    # Validate the timezone
                    is_valid, validated_timezone = validate_timezone(raw_timezone)
                    if is_valid:
                        timezone = validated_timezone
                    else:
                        print(f"SECURITY WARNING: Invalid timezone requested: {raw_timezone}")
                        # Log security event
                        log_security_event("INVALID_TIMEZONE", raw_timezone)
            
            # Format current time
            now = datetime.datetime.now()
            formatted_result = {
                "time": now.isoformat(),
                "timezone": timezone,
                "formatted": now.strftime("%B %d, %Y, %I:%M %p"),
                "note": "Timezone was validated for security"
            }
            
            # Only replace if we don't have a better format
            if not isinstance(mcp_tool["result"], dict):
                mcp_tool["result"] = formatted_result
                print(f"Created structured time result: {formatted_result}")
        except Exception as e:
            print(f"Error creating structured time: {e}")
    
    # Check all attributes of mcp_call
    for attr_name in dir(mcp_call):
        if not attr_name.startswith('_'):  # Skip private attributes
            try:
                attr_value = getattr(mcp_call, attr_name)
                if not callable(attr_value):  # Skip methods
                    print(f"  {attr_name}: {repr(attr_value)}")
            except Exception as attr_err:
                print(f"  Error getting {attr_name}: {str(attr_err)}")
                
    return mcp_tool