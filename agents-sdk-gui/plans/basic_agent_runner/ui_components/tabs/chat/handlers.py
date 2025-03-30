"""
Handlers for chat interaction with agents.
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Callable
import traceback

from agents import Agent, RunConfig, RunResult

from ui_components.tabs.chat.input_panel import validate_user_input, clear_input_field
from ui_components.tabs.chat.conversation_display import auto_scroll_chat
from ui_components.tabs.chat.session_controls import handle_clear_conversation
from agent_management import add_user_message, add_assistant_message, run_agent_with_history

def handle_chat_interactions(
    agent: Agent,
    run_config: Optional[RunConfig],
    running_agents: Dict[str, RunResult],
    user_input: str,
    send_clicked: bool,
    clear_clicked: bool,
    status_placeholder: Any,
    chat_container: Any,
    tool_calls_container: Any,
    on_clear_chat: Optional[Callable] = None
):
    """
    Handle user interactions with the chat interface.
    
    Args:
        agent: The configured agent
        run_config: The run configuration
        running_agents: Dictionary of running agent results
        user_input: Text from the input field
        send_clicked: Whether send button was clicked
        clear_clicked: Whether clear button was clicked
        status_placeholder: Placeholder for status messages
        chat_container: Container for chat display
        tool_calls_container: Container for tool calls display
        on_clear_chat: Optional callback when chat is cleared
    """
    # Get the max turns from session state or use default
    max_turns = st.session_state.get("max_turns", 10)
    
    # Handle clear history button
    if clear_clicked:
        handle_clear_conversation(on_clear_chat)
        # Force a rerun to update the UI
        st.rerun()
    
    # Handle sending a message
    if send_clicked and user_input:
        # Debug info about the message submission
        print(f"Processing message submission: '{user_input[:30]}...'")
        print(f"Agent available: {agent is not None}")
        print(f"Run config available: {run_config is not None}")
        print(f"Current conversation history: {len(st.session_state.conversation_history)} messages")
        
        # Validate configuration
        if not agent:
            status_placeholder.error("Agent is not properly configured")
            return
            
        if not run_config:
            status_placeholder.warning("Using default run configuration")
            from agents import RunConfig
            run_config = RunConfig(workflow_name="Default")
        
        # Create new tool calls container
        tool_calls_container = st.container()
        
        # Use Streamlit's native info message for processing indication
        input_preview = user_input[:50] + ("..." if len(user_input) > 50 else "")
        status_placeholder.info(f"Processing input: \"{input_preview}\"...")
        
        # Validate the user input
        is_valid, error_message = validate_user_input(user_input)
        if not is_valid:
            status_placeholder.warning(error_message)
            return
        
        # Add user message to history (sanitization happens inside add_user_message)
        add_user_message(user_input)
        
        # Clear the input field after submission
        clear_input_field()
        
        # Create placeholder for tool calls
        with tool_calls_container:
            tool_calls_placeholder = st.empty()
        
        # Clear previous outputs
        tool_calls_placeholder.markdown("*No tool calls yet*")
        
        # Get the current chat ID
        chat_id = st.session_state.current_chat_id
        
        # Run the agent with history
        try:
            # Debug agent information
            _debug_agent_info(agent)
            
            # Run the agent using our utility function
            result = run_agent_with_history(
                agent=agent,
                user_input=user_input,
                chat_container=chat_container,
                tool_calls_container=tool_calls_container,
                max_turns=max_turns,
                run_config=run_config,
                running_agents=running_agents,
                chat_id=chat_id
            )
            
            # Handle the result
            handle_agent_result(
                result=result,
                running_agents=running_agents,
                chat_id=chat_id,
                status_placeholder=status_placeholder
            )
            
            # Rerun to update UI
            st.rerun()
            
        except Exception as e:
            handle_agent_error(e, status_placeholder)

def handle_agent_result(
    result: Any, 
    running_agents: Dict[str, RunResult],
    chat_id: str,
    status_placeholder: Any
):
    """
    Handle the result from running an agent.
    
    Args:
        result: The result from running the agent
        running_agents: Dictionary of running agent results
        chat_id: The current chat ID
        status_placeholder: Placeholder for status messages
    """
    if not result:
        status_placeholder.error("Agent did not return a result")
        return
    
    # Extract response content
    response_content = extract_response_content(result)
    
    # Add to conversation history if we have content
    if response_content:
        # Debug the assistant message
        print(f"Adding assistant message to conversation history: {response_content[:50]}...")
        
        # Add assistant message
        add_assistant_message(response_content)
        
        # Debug conversation state after adding
        _debug_conversation_history()
        
        # Auto-scroll to bottom
        auto_scroll_chat()
        
        # Show preview in status
        preview = response_content[:200] + "..." if len(response_content) > 200 else response_content
        preview = preview.replace('\n', ' ').replace('"', '&quot;')
        status_placeholder.success(preview)
        
        # Store for future context
        running_agents[chat_id] = result
    else:
        # No response found - create a fallback response
        fallback_message = "I'm sorry, I processed your request but couldn't generate a proper response. Please try again or rephrase your question."
        
        # Add the fallback message
        add_assistant_message(fallback_message)
        
        # Show warning
        status_placeholder.warning("Agent didn't provide a response. Using fallback.")
        print("Using fallback response due to extraction failure.")
        
        # Store for future context
        running_agents[chat_id] = result

def extract_response_content(result: Any) -> str:
    """
    Extract the response content from a result object.
    
    Args:
        result: The result object from running an agent
        
    Returns:
        The extracted response content
    """
    response_content = ""
    
    # Print entire result object structure for debugging
    print("\nDEBUG - Result object properties:")
    for attr in dir(result):
        if not attr.startswith('_'):
            value = getattr(result, attr)
            print(f"  {attr}: {type(value)} = {str(value)[:100]}...")
    
    # Try different properties to get response content
    if hasattr(result, 'output') and result.output:
        response_content = result.output
        print(f"Using result.output: {response_content[:50]}...")
    elif hasattr(result, 'final_output') and result.final_output:
        response_content = result.final_output
        print(f"Using result.final_output: {response_content[:50]}...")
    # Try to get from answer if available
    elif hasattr(result, 'answer') and result.answer:
        response_content = result.answer
        print(f"Using result.answer: {response_content[:50]}...")
    # Try to get from messages in all steps
    elif hasattr(result, 'steps') and result.steps:
        # We need to look for assistant messages in all steps for complete context
        # This ensures we don't miss responses that come after tool calls
        for step in result.steps:
            if hasattr(step, 'messages') and step.messages:
                for msg in step.messages:
                    if hasattr(msg, 'role') and msg.role == 'assistant' and hasattr(msg, 'content'):
                        # If we already have content, append this as continuation
                        if response_content:
                            # Only add if it's not a duplicate (sometimes models repeat themselves)
                            if msg.content not in response_content:
                                response_content += "\n\n" + msg.content
                                print(f"Appending additional assistant message: {msg.content[:50]}...")
                        else:
                            response_content = msg.content
                            print(f"Using assistant message from step: {response_content[:50]}...")
    
    # Debug tool calls
    _debug_tool_calls(result)
    
    # Fallback: Try to use string representation if nothing else
    if not response_content and hasattr(result, '__str__'):
        response_content = str(result)
        print(f"Using string representation: {response_content[:50]}...")
    
    # Debug steps
    if not response_content and hasattr(result, 'steps') and result.steps:
        # Try to extract from steps
        for step in result.steps:
            if hasattr(step, 'messages') and step.messages:
                for msg in step.messages:
                    if hasattr(msg, 'role') and msg.role == "assistant" and hasattr(msg, 'content'):
                        response_content = msg.content
                        print(f"Found in steps: {response_content[:50]}...")
                        break
    
    return response_content

def handle_agent_error(error: Exception, status_placeholder: Any):
    """
    Handle errors that occur when running an agent.
    
    Args:
        error: The exception that was raised
        status_placeholder: Placeholder for status messages
    """
    # Handle errors with more information
    status_placeholder.error(f"Error: {str(error)}")
    
    error_trace = traceback.format_exc()
    print(f"Error in chat interaction: {error_trace}")
    
    # Show detailed traceback in expandable area
    with st.expander("Error Details"):
        st.code(error_trace, language="python")

def _debug_agent_info(agent: Agent):
    """
    Log debug information about the agent.
    
    Args:
        agent: The agent to log information about
    """
    print(f"Agent model: {agent.model}")
    
    # Debug tools
    tools_count = len(agent.tools) if hasattr(agent, 'tools') and agent.tools else 0
    print(f"Agent standard tools: {tools_count}")
    
    # Debug MCP servers
    if hasattr(agent, 'mcp_servers') and agent.mcp_servers:
        print(f"Agent MCP servers: {len(agent.mcp_servers)}")
        for i, server in enumerate(agent.mcp_servers):
            print(f"  MCP Server {i+1}: {type(server).__name__} - {getattr(server, 'name', 'Unnamed')}")
            # Try to list tools from this server
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tools = loop.run_until_complete(server.list_tools())
                print(f"    Available tools: {[t.name for t in tools]}")
                loop.close()
            except Exception as e:
                print(f"    Error listing tools: {str(e)}")
    else:
        print("Agent has no MCP servers")

def _debug_conversation_history():
    """Log debug information about the conversation history."""
    print(f"Conversation history after adding message: {len(st.session_state.conversation_history)} messages")
    for i, msg in enumerate(st.session_state.conversation_history):
        print(f"  Message {i+1}: {msg.get('role')} - {msg.get('content', '')[:30]}...")

def _debug_tool_calls(result: Any):
    """
    Log debug information about tool calls in the result.
    
    Args:
        result: The result object from running an agent
    """
    if hasattr(result, 'steps') and result.steps:
        print("\nAgent execution steps:")
        for i, step in enumerate(result.steps):
            print(f"Step {i+1}:")
            if hasattr(step, 'messages') and step.messages:
                for msg in step.messages:
                    if hasattr(msg, 'role'):
                        print(f"  {msg.role}: {str(msg.content)[:100]}...")
            
            if hasattr(step, 'tool_calls') and step.tool_calls:
                print(f"  Tool calls in step {i+1}:")
                for tc in step.tool_calls:
                    print(f"    - {tc.name}: {tc.args}")
                    if hasattr(tc, 'response'):
                        print(f"      Result: {tc.response}")
                        
            if hasattr(step, 'mcp_tool_calls') and step.mcp_tool_calls:
                print(f"  MCP Tool calls in step {i+1}:")
                for tc in step.mcp_tool_calls:
                    print(f"    - {tc.name}: {tc.arguments}")
                    if hasattr(tc, 'result'):
                        print(f"      Result: {tc.result}")