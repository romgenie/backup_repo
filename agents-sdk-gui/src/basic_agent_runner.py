import streamlit as st
import asyncio
from typing import Optional, Dict, List, Any, Tuple
import time
import uuid
import threading
import os
import json

# Import the Agents SDK
from agents import Agent, Runner, RunConfig
from agents.model_settings import ModelSettings
from openai.types.responses import ResponseTextDeltaEvent

# Skip Streamlit email collection on startup
os.environ['STREAMLIT_EMAIL_COLLECTION'] = 'false'

class AgentRunnerApp:
    def __init__(self):
        self.running_agents = {}
        
    def run(self):
        st.set_page_config(page_title="Agent Runner", layout="wide")
        st.title("Agent Runner")
        
        # Initialize session state for environment variables
        if "env_vars" not in st.session_state:
            st.session_state.env_vars = [("OPENAI_API_KEY", "")]
        
        # Initialize session state for saved agents
        if "saved_agents" not in st.session_state:
            st.session_state.saved_agents = {}
        
        # Tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Agent Runner", "Environment Variables", "Saved Agents"])
        
        with tab1:
            self._show_agent_runner()
            
        with tab2:
            self._show_environment_vars()
            
        with tab3:
            self._show_saved_agents()
    
    def _show_environment_vars(self):
        """Show environment variables management UI"""
        st.header("Environment Variables")
        st.markdown("Set environment variables needed by the Agents SDK (like API keys).")
        
        # Display current environment variables
        for i, (key, value) in enumerate(st.session_state.env_vars):
            col1, col2, col3 = st.columns([3, 6, 1])
            with col1:
                new_key = st.text_input(f"Key {i}", key, key=f"env_key_{i}")
            with col2:
                new_value = st.text_input(f"Value {i}", value, type="password", key=f"env_val_{i}")
            with col3:
                if st.button("🗑️", key=f"delete_env_{i}"):
                    st.session_state.env_vars.pop(i)
                    st.rerun()
            
            # Update the session state
            st.session_state.env_vars[i] = (new_key, new_value)
            
            # Apply to os.environ
            if new_key and new_key != key:
                os.environ[new_key] = new_value
            if new_key:
                os.environ[new_key] = new_value
        
        # Add new environment variable
        if st.button("Add Environment Variable"):
            st.session_state.env_vars.append(("", ""))
            st.rerun()
        
        # Apply all environment variables button
        if st.button("Apply All Environment Variables"):
            for key, value in st.session_state.env_vars:
                if key:
                    os.environ[key] = value
            st.success("Environment variables applied!")
    
    def _show_saved_agents(self):
        """Show saved agents management UI"""
        st.header("Saved Agents")
        st.markdown("Manage your saved agent configurations.")
        
        # Display saved agents
        if not st.session_state.saved_agents:
            st.info("No saved agents yet. Create and save an agent from the Agent Runner tab.")
        else:
            for agent_name, agent_config in st.session_state.saved_agents.items():
                with st.expander(f"Agent: {agent_name}"):
                    st.code(self._generate_agent_code(agent_config), language="python")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load", key=f"load_{agent_name}"):
                            st.session_state.agent = self._create_agent_from_config(agent_config)
                            st.success(f"Agent '{agent_name}' loaded!")
                            st.rerun()
                    with col2:
                        if st.button("Delete", key=f"delete_{agent_name}"):
                            del st.session_state.saved_agents[agent_name]
                            st.success(f"Agent '{agent_name}' deleted!")
                            st.rerun()
    
    def _generate_agent_code(self, agent_config):
        """Generate Python code for creating this agent"""
        code = f"""from agents import Agent
from agents.model_settings import ModelSettings

# Create the agent
agent = Agent(
    name="{agent_config['name']}",
    instructions=\"\"\"{agent_config['instructions']}\"\"\",
    model="{agent_config['model']}",
    model_settings=ModelSettings(
        temperature={agent_config['temperature']},
        max_tokens={agent_config['max_tokens']}
    )
)

# Run the agent
from agents import Runner
result = Runner.run_sync(agent, input="Your input here")
print(result.final_output)
"""
        return code
    
    def _create_agent_from_config(self, agent_config):
        """Create an Agent object from a saved configuration"""
        return Agent(
            name=agent_config['name'],
            instructions=agent_config['instructions'],
            model=agent_config['model'],
            model_settings=ModelSettings(
                temperature=agent_config['temperature'],
                max_tokens=agent_config['max_tokens']
            )
        )
            
    def _show_agent_runner(self):
        """Show the main agent runner interface"""
        # Sidebar for configuration
        with st.sidebar:
            st.header("Agent Configuration")
            
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
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create Agent"):
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
            
            with col2:
                if st.button("Save Agent"):
                    if "agent" not in st.session_state:
                        st.warning("Please create an agent first")
                    else:
                        agent_config = {
                            "name": agent_name,
                            "instructions": agent_instructions,
                            "model": model_name,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "enable_tracing": enable_tracing
                        }
                        st.session_state.saved_agents[agent_name] = agent_config
                        st.success(f"Agent '{agent_name}' saved!")
        
        # Main content area
        st.header("Agent Runner")
        
        # Only show the run interface if an agent has been created
        if "agent" in st.session_state:
            # User input
            user_input = st.text_area("User Input", placeholder="Enter your request here...")
            
            # Run button
            col1, col2 = st.columns([1, 5])
            run_button = col1.button("Run Agent")
            status_placeholder = col2.empty()
            
            # Results area
            st.subheader("Results")
            result_container = st.container()
            
            # Tool calls area
            st.subheader("Tool Calls")
            tool_calls_container = st.container()
            
            if run_button and user_input:
                run_id = str(uuid.uuid4())
                status_placeholder.info("Running agent...")
                
                # Create placeholders for streaming output
                with result_container:
                    output_placeholder = st.empty()
                
                with tool_calls_container:
                    tool_calls_placeholder = st.empty()
                
                # Clear previous outputs
                output_placeholder.markdown("**Output:** *Waiting for agent response...*")
                tool_calls_placeholder.markdown("*No tool calls yet*")
                
                # Run the agent with streaming
                try:
                    # Get the agent from session state
                    agent = st.session_state.agent
                    run_config = st.session_state.run_config
                    
                    # Since we can't directly use asyncio in Streamlit, we'll use threading
                    # Start the streaming processing in a separate thread
                    output_text = ""
                    tool_calls = []
                    
                    # Initialize thread-safe containers for streaming results
                    if "output_text" not in st.session_state:
                        st.session_state.output_text = ""
                    if "tool_calls" not in st.session_state:
                        st.session_state.tool_calls = []
                    
                    st.session_state.output_text = ""
                    st.session_state.tool_calls = []
                    
                    # Create a thread to process the streaming results
                    result = Runner.run_streamed(
                        agent, 
                        input=user_input, 
                        max_turns=max_turns,
                        run_config=run_config
                    )
                    
                    def process_stream():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        async def stream_events():
                            async for event in result.stream_events():
                                if hasattr(event, 'type') and event.type == "raw_response_event" and hasattr(event.data, 'delta'):
                                    st.session_state.output_text += event.data.delta
                                    output_placeholder.markdown(f"**Output:** {st.session_state.output_text}")
                                elif hasattr(event, 'type') and event.type == "tool_call_event":
                                    tool_call = {
                                        "name": event.data.name,
                                        "time": time.strftime("%H:%M:%S"),
                                        "args": getattr(event.data, 'args', {})
                                    }
                                    st.session_state.tool_calls.append(tool_call)
                                    tool_calls_md = "**Tool Calls:**\n" + "\n".join([
                                        f"- {t['name']} ({t['time']}): {t['args']}" 
                                        for t in st.session_state.tool_calls
                                    ])
                                    tool_calls_placeholder.markdown(tool_calls_md)
                        
                        try:
                            loop.run_until_complete(stream_events())
                        except Exception as e:
                            print(f"Error in stream_events: {e}")
                    
                    thread = threading.Thread(target=process_stream)
                    thread.start()
                    
                    # Wait for a short time to let the thread start
                    time.sleep(0.1)
                    
                    # Show a message while processing
                    status_placeholder.info("Processing response...")
                    
                    # Wait for the thread to complete (in a real app, you'd use st.rerun())
                    thread.join()
                    
                    # Update status when complete
                    status_placeholder.success("Completed!")
                    
                    # Show final result if available
                    if hasattr(result, 'final_output') and result.final_output:
                        output_placeholder.markdown(f"**Final Output:** {result.final_output}")
                    
                except Exception as e:
                    status_placeholder.error(f"Error: {str(e)}")
        else:
            st.info("Please create an agent using the sidebar configuration.")

if __name__ == "__main__":
    app = AgentRunnerApp()
    app.run()