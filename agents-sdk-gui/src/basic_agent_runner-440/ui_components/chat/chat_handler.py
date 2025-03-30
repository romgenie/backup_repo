import streamlit as st
import time
import traceback
from typing import Dict, Any, Optional, Callable

from agents import Agent, RunConfig, RunResult

from agent_management import (
    add_user_message, add_assistant_message, 
    clear_conversation, run_agent_with_history
)

def handle_chat_interactions(
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
    # Get the max turns from run config or use default
    max_turns = 10
    if run_config is not None and hasattr(run_config, "max_turns"):
        max_turns = run_config.max_turns
    
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
    if send_clicked and user_input and not st.session_state.get("message_processing", False):
        # Set processing flag to prevent duplicate submissions
        st.session_state.message_processing = True
        
        # Tool calls area
        tool_calls_container = st.container()
        
        # Use Streamlit's native info message for processing indication
        input_preview = user_input[:50] + ("..." if len(user_input) > 50 else "")
        status_placeholder.info(f"Processing input: \"{input_preview}\"...")
        
        # Store input before clearing
        processed_input = user_input.strip()
        
        # Add user message to history
        add_user_message(processed_input)
        
        # Clear the input field after submission
        st.session_state.user_input = ""
        st.session_state.input_area = ""
        
        # Create placeholder for tool calls
        with tool_calls_container:
            tool_calls_placeholder = st.empty()
        
        # Clear previous outputs
        tool_calls_placeholder.markdown("*No tool calls yet*")
        
        # Get the current chat ID
        chat_id = st.session_state.current_chat_id
        
        # Run the agent with history
        try:
            # Run the agent using our utility function
            result = run_agent_with_history(
                agent=agent,
                user_input=processed_input,
                chat_container=chat_container,
                tool_calls_container=tool_calls_container,
                max_turns=max_turns,
                run_config=run_config,
                running_agents=running_agents,
                chat_id=chat_id
            )
            
            if result:
                # Add assistant response to conversation history
                response_content = ""
                if hasattr(result, 'output') and result.output:
                    response_content = result.output
                elif hasattr(result, 'final_output') and result.final_output:
                    response_content = result.final_output
                    
                if response_content:
                    # Add to conversation history
                    add_assistant_message(response_content)
                    
                    # Add HTML to auto-scroll to the bottom of the conversation
                    st.markdown("""
                    <script>
                        var chatContainer = document.querySelector('[data-testid="stVerticalBlock"]');
                        if (chatContainer) {
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                    </script>
                    """, unsafe_allow_html=True)
                    
                    # Show a preview of the response
                    preview = response_content[:200] + "..." if len(response_content) > 200 else response_content
                    # Clean preview for display in HTML
                    preview = preview.replace('\n', ' ').replace('"', '&quot;')
                    
                    # Use Streamlit's native success message with the actual response content
                    status_placeholder.success(preview)
                    
                    # Store the result for the next turn in the external dictionary
                    running_agents[chat_id] = result
                    
            # Reset processing flag before rerun
            st.session_state.message_processing = False
            
            # Force a rerun to update the UI with new messages
            st.rerun()
            
        except Exception as e:
            # Use Streamlit's native error message
            status_placeholder.error(f"Error: {str(e)}")
            
            # Reset processing flag to allow retrying
            st.session_state.message_processing = False
            
            # Add a collapsible section for the full traceback
            with st.expander("See detailed error traceback"):
                st.code(traceback.format_exc(), language="python")