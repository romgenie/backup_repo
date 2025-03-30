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
    
    # Agent creation section
    agent_name = st.text_input("Agent Name", "Assistant")
    
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
        height=300,
        help="Provide detailed instructions for the agent's behavior"
    )
    
    # Model settings in a collapsible section
    with st.expander("Model Settings", expanded=True):
        model_name = st.selectbox(
            "Model", 
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"],
            help="Select the AI model to use for this agent"
        )
        
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
    
    # Run settings in a collapsible section
    with st.expander("Run Settings", expanded=True):
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
    
    # Create a button to save the agent configuration
    if st.button("Create Agent", type="primary"):
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
        print(f"DEBUG: Session state has 'agent_mcp_servers': {'agent_mcp_servers' in st.session_state}")
        print(f"DEBUG: Session state keys: {list(st.session_state.keys())}")
        
        for i, server in enumerate(mcp_servers):
            print(f"DEBUG: Server {i+1}: {type(server).__name__} - {getattr(server, 'name', 'Unnamed')}")
            
        # Also check the selected_mcp_servers in case mcp_servers weren't loaded
        print(f"DEBUG: selected_mcp_servers: {st.session_state.get('selected_mcp_servers', [])}")
        
        # If we have selected servers but no active servers, try to load them now
        if (not mcp_servers or len(mcp_servers) == 0) and len(st.session_state.get('selected_mcp_servers', [])) > 0:
            print("DEBUG: No MCP servers loaded but some are selected, attempting to load now")
            try:
                # Try to access the MCP manager
                if "_app" in st.session_state and hasattr(st.session_state._app, "mcp_manager"):
                    mcp_manager = st.session_state._app.mcp_manager
                    mcp_servers = mcp_manager.get_servers_for_agent(st.session_state.selected_mcp_servers)
                    print(f"DEBUG: Loaded {len(mcp_servers)} MCP servers dynamically")
            except Exception as e:
                print(f"DEBUG: Error loading MCP servers: {str(e)}")
        
        # Show a message if no MCP servers are selected
        if not mcp_servers:
            st.warning("No MCP servers selected. Please go to the MCP tab to configure and select servers.")
            st.info("Using MCP servers is recommended for providing your agent with useful capabilities.")
            mcp_servers = []
        
        # Create the agent
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
        
        # Create run config (max_turns is not a parameter for RunConfig)
        run_config = RunConfig(
            tracing_disabled=not enable_tracing,
            workflow_name=f"{agent_name} Workflow"
            # Note: tools_debug_enabled parameter doesn't exist in this RunConfig version
        )
        
        # Store max_turns in session state to use during agent execution
        st.session_state.max_turns = max_turns
        
        # Call the creation callback
        on_agent_create(agent, run_config)
        
        # Show success message
        st.success(f"Agent '{agent_name}' created successfully! Go to the Chat tab to start the conversation.")
        
    # If agent is already created, show its configuration
    if "agent" in st.session_state:
        st.divider()
        
        current_agent = st.session_state.agent
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