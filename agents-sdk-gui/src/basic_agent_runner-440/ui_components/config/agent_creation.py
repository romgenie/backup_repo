import streamlit as st
import os
import time
import traceback
from typing import Dict, Any, Callable

from agents import Agent, RunConfig, ModelSettings, Tool

def handle_agent_creation(
    agent_name: str,
    agent_instructions: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_turns: int,
    enable_tracing: bool,
    api_key: str,
    on_agent_create: Callable
):
    """
    Handle the agent creation process
    
    Args:
        agent_name: Name of the agent
        agent_instructions: System instructions for the agent
        model_name: The LLM model to use
        temperature: Temperature setting (0.0-1.0)
        max_tokens: Maximum tokens in response
        max_turns: Maximum conversation turns
        enable_tracing: Whether to enable tracing
        api_key: OpenAI API key
        on_agent_create: Callback for when agent is created
    
    Returns:
        Boolean indicating success
    """
    # Track agent creation state
    if "agent_creating" not in st.session_state:
        st.session_state.agent_creating = False
        
    # Create a button to save the agent configuration
    create_button = st.button("Create Agent", type="primary", disabled=st.session_state.agent_creating)
    
    # Handle button click or continue processing if already started
    if create_button or st.session_state.agent_creating:
        # Set creating flag to prevent multiple submissions
        st.session_state.agent_creating = True
        
        # Create a status indicator
        status = st.empty()
        
        # Check if API key is available
        if not api_key:
            status.error("Please provide an OpenAI API key in the Settings tab")
            st.session_state.agent_creating = False
            return False
        
        # Show processing status
        status.info("Creating agent... Please wait")
        
        try:
            # Set API key in environment
            os.environ["OPENAI_API_KEY"] = api_key
            
            # Get selected tools from session state if available
            tools = st.session_state.get("selected_tools", [])
            
            # Log tools being used (for debugging)
            st.write(f"Configuring agent with {len(tools)} tools")
            for i, tool in enumerate(tools):
                tool_name = getattr(tool, 'name', f'Tool {i+1}')
                st.write(f"- {tool_name}: {type(tool).__name__}")
            
            # Create the agent
            agent = Agent(
                name=agent_name,
                instructions=agent_instructions,
                model=model_name,
                tools=tools,
                model_settings=ModelSettings(
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
            
            # Create run config
            run_config = RunConfig(
                tracing_disabled=not enable_tracing,
                workflow_name=f"{agent_name} Workflow"
            )
            
            # Call the creation callback
            on_agent_create(agent, run_config)
            
            # Add a welcome message from the assistant to the conversation
            from agent_management import add_assistant_message, clear_conversation
            
            # Clear any previous conversation first
            clear_conversation()
            
            # Add welcome message
            welcome_message = f"Hello, I'm {agent_name}. How can I help you today?"
            add_assistant_message(welcome_message)
            
            # Show success message
            status.success(f"Agent '{agent_name}' created successfully! Go to the Chat tab to start the conversation.")
            
            # Reset the creating flag
            st.session_state.agent_creating = False
            
            # Add a small delay and rerun to ensure UI updates properly
            time.sleep(0.5)
            st.rerun()
            
            return True
            
        except Exception as e:
            # Show error message
            status.error(f"Error creating agent: {str(e)}")
            st.session_state.agent_creating = False
            
            # Show detailed error in expandable section
            with st.expander("Error Details"):
                st.code(traceback.format_exc(), language="python")
            
            return False
    
    return False