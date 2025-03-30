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
    st.markdown("""
    <div class="form-section">
        <h2>Configure Agent</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a form for agent creation
    with st.form(key="agent_creation_form"):
        st.markdown('<div class="form-group">', unsafe_allow_html=True)
        # Agent name
        agent_name = st.text_input("Agent Name", "Assistant")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-group">', unsafe_allow_html=True)
        # Instructions
        agent_instructions = st.text_area(
            "Instructions", 
            """You are a helpful assistant with access to MCP (Model Context Protocol) tools.
These tools provide you with powerful capabilities that let you perform actions and access information that would normally be outside your context.

## Core MCP Tool Categories

1. **Time Services**
   - Get the current date and time in any timezone
   - Example: `get_current_time` tool for accurate time information

2. **Web Services**
   - Fetch web content from URLs
   - Navigate websites with Puppeteer
   - Take screenshots and extract information
   - Example: Use `fetch` to retrieve web content

3. **Memory Services**
   - Store and retrieve information across conversations
   - Example: Use `saveMemory` to remember facts for later use

4. **File System Services**
   - Read, write, and list files
   - Create and delete directories
   - Example: Use `readFile` to access document content

5. **GitHub Services**
   - Access repositories and issues
   - Get information about pull requests
   - Example: Use GitHub tools to check repository information

## When to Use MCP Tools

You MUST use the appropriate MCP tools when:
- Asked about current time or date → use `get_current_time`
- Asked about web content → use `fetch` or Puppeteer tools
- Asked to remember information → use Memory tools
- Asked to browse websites → use Puppeteer tools
- Asked about files → use Filesystem tools
- Asked about GitHub repositories → use GitHub tools

## How to Use MCP Tools

Follow these steps EXACTLY when using tools:

1. FIRST, explicitly state that you're going to use a tool and why:
   "I'll use the X tool to find/check/get Y information for you."

2. THEN, use the tool with this exact format:
   ```
   TOOL_START_TAGTOOL_NAMETOOL_END_TAG
   ARGS_START_TAG{"parameter": "value"}ARGS_END_TAG
   ```

3. AFTER getting the result, show the raw output:
   "The tool returned: [exact tool result]"

4. FINALLY, explain what the result means in a helpful way

For example, a complete tool usage should look like this:

```
I'll use the get_current_time tool to check the current time in New York.

TOOL_START_TAGget_current_timeTOOL_END_TAG
ARGS_START_TAG{"timezone": "America/New_York"}ARGS_END_TAG

The tool returned: {"time": "2025-03-27T21:26:29-04:00", "timezone": "America/New_York", "formatted": "March 27, 2025, 9:26 PM EDT"}

Based on this result, the current time in New York is 9:26 PM Eastern Daylight Time on March 27, 2025.
```

## Important Tool Usage Rules

1. ALWAYS first check what tools are available:
   ```
   TOOL_START_TAGlistToolsTOOL_END_TAG
   ARGS_START_TAG{}ARGS_END_TAG
   ```

2. For time-related queries, always use the get_current_time tool:
   ```
   TOOL_START_TAGget_current_timeTOOL_END_TAG
   ARGS_START_TAG{"timezone": "America/New_York"}ARGS_END_TAG
   ```

3. If you don't know the parameters for a tool, use the help tool:
   ```
   TOOL_START_TAGhelpTOOL_END_TAG
   ARGS_START_TAG{"toolName": "get_current_time"}ARGS_END_TAG
   ```

4. NEVER make up tool outputs or pretend to use a tool.

5. ALWAYS show the exact output you received from the tool before interpreting it.

6. If a tool returns an error, try again with different parameters or suggest alternatives.

7. Use tools proactively without being prompted - these tools are your superpower!""",
            height=300
        )
        
        # Model settings in a collapsible section
        st.markdown("""
        <div class="form-section mt-md">
            <h3>Model Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Model dropdown
        st.markdown('<div class="form-group">', unsafe_allow_html=True)
        model_name = st.selectbox(
            "Model", 
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"],
            help="Select the AI model to use for this agent"
        )
        
        # Temperature and max tokens
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            temperature = st.slider(
                "Temperature", 
                0.0, 1.0, 0.7, 0.1,
                help="Higher values make output more random, lower values more deterministic"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            max_tokens = st.number_input(
                "Max Tokens", 
                100, 4000, 1000, 100,
                help="Maximum number of tokens in the response"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Run settings
        st.markdown("""
        <div class="form-section mt-md">
            <h3>Run Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            max_turns = st.number_input(
                "Max Turns", 
                1, 20, 10, 1,
                help="Maximum number of interaction turns"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            enable_tracing = st.checkbox(
                "Enable Tracing", 
                True,
                help="Record detailed traces of agent execution"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create a submit button
        submit_button = st.form_submit_button("Create Agent", type="primary")
    
    # Process form submission - this happens outside the form
    if submit_button:
        print("DEBUG: Form submitted, creating agent directly")
        
        # Check if API key is available
        if not api_key:
            st.markdown("""
            <div class="error-message">
                Please provide an OpenAI API key in the Settings tab
            </div>
            """, unsafe_allow_html=True)
            return
            
        # Set API key in environment
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Add our custom tools
        from agent_management.agent_runner import get_weather, search_news
        from ui_components.tools.custom_tools import get_current_time
        
        # Include our custom time tool as well as the weather and news tools
        tools = [get_current_time, get_weather, search_news]
        
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
            
            # Create a more detailed display
            with st.expander("MCP Servers and Tools", expanded=True):
                # Initialize a counter for total tools
                total_mcp_tools = 0
                
                for server in current_agent.mcp_servers:
                    server_name = getattr(server, 'name', 'Unnamed Server')
                    
                    # Try to get tools from this server
                    server_tools = []
                    server_status = "unknown"
                    error_message = None
                    
                    # Check if app and mcp_manager are available
                    app = None
                    if "_app" in st.session_state:
                        app = st.session_state._app
                    
                    try:
                        if app and hasattr(app, 'mcp_manager'):
                            # Find this server in the manager
                            found_server = False
                            for server_id, tools in app.mcp_manager.server_tools_cache.items():
                                # Try to match by name
                                server_config = app.mcp_manager.server_configs.get(server_id, {})
                                if server_config.get('name') == server_name:
                                    server_tools = tools
                                    found_server = True
                                    
                                    # Check server health status
                                    if server_id in app.mcp_manager.server_health:
                                        server_status = app.mcp_manager.server_health[server_id].get("status", "unknown")
                                        if server_status == "error" or server_status == "connection_error":
                                            error_message = app.mcp_manager.server_health[server_id].get("error", "Unknown error")
                                    break
                                    
                            if not found_server:
                                server_status = "not_found"
                                error_message = f"Server '{server_name}' not found in MCP manager cache"
                    except Exception as e:
                        server_status = "error"
                        error_message = str(e)
                    
                    # Display server with tool count and status
                    tool_count = len(server_tools)
                    total_mcp_tools += tool_count
                    
                    # Status indicator based on server health
                    if server_status == "healthy" or server_status == "connected":
                        status_indicator = "🟢"
                        status_text = ""
                    elif server_status == "error" or server_status == "connection_error" or server_status == "creation_failed":
                        status_indicator = "🔴"
                        status_text = f" - Error: {error_message}" if error_message else " - Error"
                    elif server_status == "not_found":
                        status_indicator = "🟠"
                        status_text = " - Not found in cache"
                    else:
                        status_indicator = "⚪"
                        status_text = ""
                        
                    st.markdown(f"### {status_indicator} {server_name} ({tool_count} tools){status_text}")
                    
                    # If we have tools, display them
                    if server_tools:
                        # Create a table-like display
                        for tool in server_tools:
                            tool_name = tool.get('name', 'Unnamed Tool')
                            tool_desc = tool.get('description', 'No description available')
                            
                            st.markdown(f"**{tool_name}**")
                            st.markdown(f"_{tool_desc}_")
                            
                            # Show parameters if any
                            params = tool.get('parameters', [])
                            if params:
                                st.markdown("**Parameters:**")
                                param_md = ""
                                for param in params:
                                    param_name = param.get('name', 'unnamed')
                                    param_type = param.get('type', 'unknown')
                                    param_required = " (Required)" if param.get('required', False) else ""
                                    param_md += f"- `{param_name}`: {param_type}{param_required}\n"
                                
                                st.markdown(param_md)
                            
                            st.markdown("---")
                    else:
                        st.info(f"No tools information available for {server_name}")
                
                # Summary information
                st.markdown(f"**Total MCP Tools Available: {total_mcp_tools}**")
                
                # Help text
                if total_mcp_tools > 0:
                    st.markdown("""
                    **How to use these tools:**
                    ```
                    TOOL_START_TAGTOOL_NAMETOOL_END_TAG
                    ARGS_START_TAG{"parameter": "value"}ARGS_END_TAG
                    ```
                    """)
            
            # Add a note about managing MCP servers
            st.info("Go to the MCP tab to manage MCP servers and the Tools tab to see all available tools")