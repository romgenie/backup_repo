import streamlit as st
import asyncio
from typing import Optional, Dict, List, Any, Callable
import time
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Import the Agents SDK
from agents import Agent, Runner, RunConfig, RunResult
from agents.model_settings import ModelSettings
from openai.types.responses import ResponseTextDeltaEvent

# Import our utility modules
from sys_utils import save_to_env_file, load_from_env_file
from async_utils import async_to_sync
from ui_components import display_chat, display_tool_calls
from agent_management import (
    init_conversation, add_user_message, add_assistant_message, 
    clear_conversation, run_agent_with_history
)

class AgentRunnerApp:
    def __init__(self):
        self.running_agents: Dict[str, RunResult] = {}
        
        # Load .env file if it exists
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path)
        
        # Get OpenAI API key from environment
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.env_file_path = dotenv_path if dotenv_path else str(Path.cwd() / ".env")
        
        # Initialize conversation state
        init_conversation()
        
    def run(self):
        st.title("Agent Runner")
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("Agent Configuration")
            
            # API Key configuration
            st.subheader("API Key")
            api_key = st.text_input("OpenAI API Key", value=self.api_key, type="password")
            
            # .env file buttons
            col1, col2 = st.columns(2)
            
            if col1.button("Save to .env"):
                if api_key:
                    saved_path = save_to_env_file(self.env_file_path, api_key)
                    st.success(f"API key saved to {saved_path}")
                else:
                    st.warning("Please enter an API key to save")
            
            if col2.button("Load from .env"):
                loaded_api_key, success = load_from_env_file(self.env_file_path)
                if success:
                    self.api_key = loaded_api_key
                    st.success("API key loaded from .env file")
                    # Force a rerun to update the text input with the new API key
                    st.rerun()
                else:
                    st.warning("No .env file found")
            
            # Agent creation section
            agent_name = st.text_input("Agent Name", "Assistant")
            agent_instructions = st.text_area("Instructions", "You are a helpful assistant.")
            
            # Model settings
            st.subheader("Model Settings")
            model_name = st.selectbox(
                "Model", 
                ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"]
            )
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.number_input("Max Tokens", 100, 4000, 1000, 100)
            
            # Run settings
            st.subheader("Run Settings")
            max_turns = st.number_input("Max Turns", 1, 20, 10, 1)
            enable_tracing = st.checkbox("Enable Tracing", True)
            
            # Create a button to save the agent configuration
            if st.button("Create Agent"):
                # Store API key for future use
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    self.api_key = api_key
                elif not self.api_key:
                    st.error("Please provide an OpenAI API key")
                    return
                
                st.session_state.agent = Agent(
                    name=agent_name,
                    instructions=agent_instructions,
                    model=model_name,
                    model_settings=ModelSettings(
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                )
                st.session_state.run_config = RunConfig(
                    tracing_disabled=not enable_tracing,
                    workflow_name=f"{agent_name} Workflow"
                )
                st.success(f"Agent '{agent_name}' created!")
        
        # Main content area
        st.header("Agent Runner")
        
        # Only show the run interface if an agent has been created
        if "agent" in st.session_state:
            # Display conversation history with a more chat-like appearance
            st.subheader("Conversation")
            
            # Create a container with a fixed height for the chat history to make it scrollable
            chat_container = st.container(height=400)
            
            # Display the chat using our utility function
            display_chat(chat_container, st.session_state.conversation_history)
            
            # Create a more chat-like input area
            st.write("") # Add some space
            st.divider() # Add a divider between conversation and input area
            
            # User input with key for clearing
            if "user_input" not in st.session_state:
                st.session_state.user_input = ""
            
            # Use a horizontal layout for input and buttons
            col1, col2 = st.columns([5, 1])
            
            with col1:
                user_input = st.text_area("Type your message", value=st.session_state.user_input, 
                                         placeholder="Ask a question...", key="input_area", 
                                         height=80)
            
            with col2:
                st.write("") # Add space to align buttons
                run_button = st.button("Send", type="primary")
                clear_button = st.button("Clear Chat")
            
            # Status area with more space
            status_placeholder = st.empty()
            
            # Tool calls area
            st.subheader("Tool Calls")
            tool_calls_container = st.container()
            
            # Handle clear history button
            if clear_button:
                # Clear conversation history and get new chat ID
                new_chat_id = clear_conversation()
                # Clear any stored results
                self.running_agents = {}
                # Force a rerun to update the UI
                st.rerun()
            
            if run_button and user_input:
                # Use Streamlit's native info message
                input_preview = user_input[:50] + ("..." if len(user_input) > 50 else "")
                status_placeholder.info(f"Processing input: \"{input_preview}\"...")
                
                # Add user message to history
                add_user_message(user_input)
                
                # Clear the input field after submission
                st.session_state.user_input = ""
                
                # Create placeholder for tool calls
                with tool_calls_container:
                    tool_calls_placeholder = st.empty()
                
                # Clear previous outputs
                tool_calls_placeholder.markdown("*No tool calls yet*")
                
                # Run the agent with history
                try:
                    # Get the agent from session state
                    agent = st.session_state.agent
                    run_config = st.session_state.run_config
                    chat_id = st.session_state.current_chat_id
                    
                    # Run the agent using our utility function
                    result = run_agent_with_history(
                        agent=agent,
                        user_input=user_input,
                        chat_container=chat_container,
                        tool_calls_container=tool_calls_container,
                        max_turns=max_turns,
                        run_config=run_config,
                        running_agents=self.running_agents,
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
                            
                            # Store the result for the next turn
                            self.running_agents[chat_id] = result
                    
                    # Force a rerun to update the UI with new messages
                    st.rerun()
                    
                except Exception as e:
                    # Use Streamlit's native error message
                    status_placeholder.error(f"Error: {str(e)}")
                    
                    import traceback
                    # Add a collapsible section for the full traceback
                    with st.expander("See detailed error traceback"):
                        st.code(traceback.format_exc(), language="python")
        else:
            st.info("Please create an agent using the sidebar configuration.")
    # Class contains only the run method now
    # All other functionality has been moved to separate modules

if __name__ == "__main__":
    app = AgentRunnerApp()
    app.run()