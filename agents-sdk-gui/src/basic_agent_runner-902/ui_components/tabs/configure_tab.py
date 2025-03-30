import streamlit as st
import os
from typing import Dict, Any, Callable, Optional, List

from agents import Agent, RunConfig, ModelSettings, Tool

def render_configure_tab(
    api_key: str,
    env_file_path: str,
    on_agent_create: Callable[[Agent, RunConfig], None],
    on_api_key_save: Callable[[str], str],
    on_api_key_load: Callable[[str], tuple]
) -> None:
    """
    Render the configure tab of the application
    
    Args:
        api_key: Current API key
        env_file_path: Path to the .env file
        on_agent_create: Callback when agent is created
        on_api_key_save: Callback to save API key
        on_api_key_load: Callback to load API key
    """
    st.header("Configure Agent")
    
    # Create a form for agent creation
    with st.form(key="agent_creation_form"):
        # Agent name
        agent_name = st.text_input("Agent Name", "Assistant")
        
        # Instructions
        agent_instructions = st.text_area(
            "Instructions", 
            """You are a helpful assistant with access to MCP (Model Context Protocol) tools. 
These tools give you special capabilities such as:
- Time functions (checking current date/time with the getCurrentTime tool)
- Web automation with Puppeteer
- Memory storage
- File system operations
- Web content fetching

Very important: You MUST use the available tools whenever they can help answer a question.
When a user asks for information that requires a tool, you MUST proactively use it without being prompted.

Examples of when to use tools:
- If asked about the current time or date → use the getCurrentTime tool
- If asked about web content → use the fetch tool
- If asked to remember information → use the Memory tool
- If asked to browse a website → use Puppeteer tools
- If asked about files → use Filesystem tools

Step by step guide for using tools:
1. Identify when a user question can be answered better with a tool
2. Choose the appropriate tool based on the question
3. Call the tool with the correct parameters
4. Use the tool's response to answer the user's question

Important: When using a tool, format your response exactly like this:
<tool>TOOL_NAME</tool>
<args>{"parameter": "value"}</args>

For example, to get the current time:
<tool>getCurrentTime</tool>
<args>{}</args>

If you don't know which tools are available, you can list them using:
<tool>listTools</tool>
<args>{}</args>

Never pretend to use a tool or make up tool results. If needed, tell the user which tool you'd use but explain it's not available.

Please use the tools whenever possible - they are your main way to provide accurate information!""",
            height=300
        )
        
        # Model settings in a collapsible section
        st.subheader("Model Settings")
        
        # Model dropdown
        model_name = st.selectbox(
            "Model", 
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"],
            help="Select the AI model to use for this agent"
        )
        
        # Temperature and max tokens
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "Temperature", 
                0.0, 1.0, 0.7, 0.1,
                help="Higher values make output more random, lower values more deterministic"
            )
        with col2:
            max_tokens = st.number_input(
                "Max Tokens", 
                100, 4000, 1000, 100,
                help="Maximum number of tokens in the response"
            )
        
        # Run settings
        st.subheader("Run Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            max_turns = st.number_input(
                "Max Turns", 
                1, 20, 10, 1,
                help="Maximum number of interaction turns"
            )
        with col2:
            enable_tracing = st.checkbox(
                "Enable Tracing", 
                True,
                help="Record detailed traces of agent execution"
            )
        
        # Create a submit button
        submit_button = st.form_submit_button("Create Agent", type="primary")
    
    # Process form submission - this happens outside the form
    if submit_button:
        print("DEBUG: Form submitted, creating agent directly")
        
        # Check if API key is available
        if not api_key:
            st.error("Please provide an OpenAI API key in the Settings tab")
            return
            
        # Set API key in environment
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Add our custom tools
        from agent_management.agent_runner import get_weather, search_news
        from ui_components.tools.custom_tools import get_current_time_custom
        
        # Include our custom time tool as well as the weather and news tools
        tools = [get_current_time_custom, get_weather, search_news]
        
        # Check for MCP servers from session state (now using agent_mcp_servers)
        mcp_servers = st.session_state.get("agent_mcp_servers", [])
        
        # Debug MCP server info
        print(f"DEBUG: Creating agent with {len(mcp_servers)} MCP servers")
        for i, server in enumerate(mcp_servers):
            server_name = getattr(server, 'name', f"Server {i}")
            print(f"  - MCP server {i}: {server_name}")
        
        # Create the agent with form values
        agent = Agent(
            name=agent_name,
            instructions=agent_instructions,
            model=model_name,
            tools=tools,
            model_settings=ModelSettings(
                temperature=temperature,
                max_tokens=max_tokens
            ),
            mcp_servers=mcp_servers
        )
        
        # Create run config
        run_config = RunConfig(
            tracing_disabled=not enable_tracing,
            workflow_name=f"{agent_name} Workflow"
        )
        
        # Store max_turns in session state to use during agent execution
        st.session_state.max_turns = max_turns
        
        # Call the creation callback
        on_agent_create(agent, run_config)
        
        # Redundantly ensure the agent is in session state
        st.session_state.agent = agent
        st.session_state.run_config = run_config
        print(f"DEBUG: Directly set agent in session state: {'agent' in st.session_state}")
        
        # Show success message
        st.success(f"Agent '{agent_name}' created successfully! Redirecting to Chat tab...")
        
        # Set a special flag to indicate we should switch to the chat tab
        st.session_state.switch_to_chat_tab = True
        
        # Insert JavaScript to automatically click the "Chat" tab
        st.markdown("""
        <script>
            // Wait for the DOM to be fully loaded
            document.addEventListener("DOMContentLoaded", function() {
                // Function to click the Chat tab
                function clickChatTab() {
                    // Find all tab buttons
                    const tabs = document.querySelectorAll('button[role="tab"]');
                    // Click the first one (Chat tab)
                    if (tabs.length > 0) {
                        tabs[0].click();
                    }
                }
                
                // Try immediately and also with a delay to ensure the tabs are loaded
                clickChatTab();
                setTimeout(clickChatTab, 500);
            });
        </script>
        """, unsafe_allow_html=True)
        
        # Force a rerun to update the UI with the new agent
        st.rerun()
    
    # Display the current agent if it exists
    from services.session_manager import SessionManager
    
    if SessionManager.has_agent():
        st.divider()
        
        current_agent = SessionManager.get_agent()
        st.subheader("Current Agent Configuration")
        
        st.markdown(f"**Name:** {current_agent.name}")
        st.markdown(f"**Model:** {current_agent.model}")
        
        if hasattr(current_agent, 'model_settings') and current_agent.model_settings:
            st.markdown(f"**Temperature:** {current_agent.model_settings.temperature}")
            st.markdown(f"**Max Tokens:** {current_agent.model_settings.max_tokens}")
        
        st.markdown("**Instructions:**")
        st.markdown(f"```\n{current_agent.instructions}\n```")
        
        # Display tools if any
        if hasattr(current_agent, 'tools') and current_agent.tools:
            st.markdown("**Tools:**")
            for tool in current_agent.tools:
                tool_type = tool.__class__.__name__
                tool_name = getattr(tool, 'name', tool_type)
                st.markdown(f"- {tool_name} ({tool_type})")
                
        # Display MCP servers if any
        if hasattr(current_agent, 'mcp_servers') and current_agent.mcp_servers:
            st.markdown("**MCP Servers:**")
            for server in current_agent.mcp_servers:
                server_name = getattr(server, 'name', 'Unnamed Server')
                st.markdown(f"- {server_name}")
                
            # Add a note about managing MCP servers
            st.info("Use the MCP tab to manage MCP servers")