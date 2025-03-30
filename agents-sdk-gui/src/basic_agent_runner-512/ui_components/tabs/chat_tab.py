import streamlit as st
import time
from typing import Dict, List, Any, Tuple, Optional, Callable

from agents import Agent, RunConfig, RunResult

from ui_components import display_chat, display_tool_calls
from agent_management import (
    add_user_message, add_assistant_message, clear_conversation,
    run_agent_with_history
)

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
    
    # Add debug information
    with st.expander("Debug Info", expanded=False):
        st.write("Agent Configuration:")
        st.write(f"- Name: {agent.name}")
        st.write(f"- Model: {agent.model}")
        st.write(f"- Tools: {len(agent.tools) if hasattr(agent, 'tools') and agent.tools else 0}")
        st.write("Run Configuration:")
        if run_config:
            st.write(f"- Tracing: {'Disabled' if run_config.tracing_disabled else 'Enabled'}")
            st.write(f"- Workflow: {run_config.workflow_name}")
            st.write(f"- Max Turns: {st.session_state.get('max_turns', 10)}")
        else:
            st.write("No run configuration available")
    
    # Display conversation history with a more chat-like appearance
    st.subheader("Conversation")
    
    # Create a container with a fixed height for the chat history to make it scrollable
    chat_container = st.container(height=400)
    
    # Display the chat using our utility function
    display_chat(chat_container, st.session_state.conversation_history)
    
    # Status area with more space
    status_placeholder = st.empty()
    
    # Tool calls area (collapsible)
    tool_calls_container = _render_tool_calls_area()
    
    # Input area and control buttons
    user_input, send_clicked = _render_input_area()
    
    # Action buttons row
    clear_clicked = _render_action_buttons()
    
    # Handle user interactions
    _handle_chat_interactions(
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

def _render_tool_calls_area():
    """
    Render the collapsible tool calls area
    
    Returns:
        The tool calls container
    """
    with st.expander("Tool Calls", expanded=False):
        tool_calls_container = st.container()
        
    return tool_calls_container
    
def _render_input_area() -> Tuple[str, bool]:
    """
    Render the message input area and send button
    
    Returns:
        Tuple containing (user_input, send_button_clicked)
    """
    # Create a more chat-like input area
    st.write("") # Add some space
    st.divider() # Add a divider between conversation and input area
    
    # User input with key for clearing
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    # Use a horizontal layout for input and buttons
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_area(
            "Type your message", 
            value=st.session_state.user_input, 
            placeholder="Ask a question...", 
            key="input_area", 
            height=80
        )
    
    with col2:
        st.write("") # Add space to align buttons
        send_button = st.button("Send", type="primary", key="send_button")
    
    return user_input, send_button
    
def _render_action_buttons() -> bool:
    """
    Render action buttons like Clear Chat
    
    Returns:
        Boolean indicating if Clear Chat was clicked
    """
    # Add some space
    st.write("")
    
    # Create columns for buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    
    # Clear chat button
    clear_button = col1.button("Clear Chat", key="clear_chat_button")
    
    # Add more action buttons in the future
    # e.g., Save Conversation, Copy to Clipboard
    
    return clear_button
    
def _handle_chat_interactions(
    agent: Agent,
    run_config: RunConfig,
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
    Handle user interactions with the chat interface
    
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
        # Clear conversation history and get new chat ID
        new_chat_id = clear_conversation()
        # Clear any stored results if needed
        if on_clear_chat is not None:
            on_clear_chat()
        # Force a rerun to update the UI
        st.rerun()
    
    # Handle sending a message
    if send_clicked and user_input:
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
        
        # Add user message to history first (for immediate display)
        add_user_message(user_input)
        
        # Clear the input field after submission
        st.session_state.user_input = ""
        
        # Create placeholder for tool calls
        with tool_calls_container:
            tool_calls_placeholder = st.empty()
        
        # Clear previous outputs
        tool_calls_placeholder.markdown("*No tool calls yet*")
        
        # Get the current chat ID
        chat_id = st.session_state.current_chat_id
        
        # Run the agent with history
        try:
            # Print debug info
            print(f"Running agent with input: {user_input[:50]}...")
            print(f"Agent model: {agent.model}")
            print(f"Agent tools: {len(agent.tools) if hasattr(agent, 'tools') and agent.tools else 0}")
            
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
            if result:
                # Extract response content
                response_content = ""
                
                # Try different properties to get response content
                if hasattr(result, 'output') and result.output:
                    response_content = result.output
                    print(f"Using result.output: {response_content[:50]}...")
                elif hasattr(result, 'final_output') and result.final_output:
                    response_content = result.final_output
                    print(f"Using result.final_output: {response_content[:50]}...")
                elif hasattr(result, 'steps') and result.steps:
                    # Try to extract from steps
                    for step in result.steps:
                        if hasattr(step, 'messages') and step.messages:
                            for msg in step.messages:
                                if hasattr(msg, 'role') and msg.role == "assistant" and hasattr(msg, 'content'):
                                    response_content = msg.content
                                    print(f"Found in steps: {response_content[:50]}...")
                                    break
                
                # Fallback to string representation
                if not response_content and result:
                    response_content = str(result)
                    print(f"Using string representation: {response_content[:50]}...")
                    
                # Add to conversation history if we have content
                if response_content:
                    # Add assistant message
                    add_assistant_message(response_content)
                    
                    # Auto-scroll to bottom with JavaScript
                    st.markdown("""
                    <script>
                        var chatContainer = document.querySelector('[data-testid="stVerticalBlock"]');
                        if (chatContainer) {
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                    </script>
                    """, unsafe_allow_html=True)
                    
                    # Show preview in status
                    preview = response_content[:200] + "..." if len(response_content) > 200 else response_content
                    preview = preview.replace('\n', ' ').replace('"', '&quot;')
                    status_placeholder.success(preview)
                    
                    # Store for future context
                    running_agents[chat_id] = result
                else:
                    # No response found
                    status_placeholder.warning("Agent didn't provide a response")
                    print("Result properties:", dir(result))
            else:
                # No result returned
                status_placeholder.error("Agent did not return a result")
            
            # Rerun to update UI
            st.rerun()
            
        except Exception as e:
            # Handle errors with more information
            status_placeholder.error(f"Error: {str(e)}")
            
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in chat interaction: {error_trace}")
            
            # Show detailed traceback in expandable area
            with st.expander("Error Details"):
                st.code(error_trace, language="python")