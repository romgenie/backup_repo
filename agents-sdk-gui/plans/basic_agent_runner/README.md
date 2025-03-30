# Basic Agent Runner with MCP Support

A streamlined interface for interacting with AI agents using OpenAI's Agents SDK with Model Context Protocol (MCP) integration.

## Features

- **Tabbed Interface**: Navigate between Chat, Configure, MCP, History, and Settings tabs
- **Agent Configuration**: Set up agents with different models and parameters
- **MCP Server Integration**: Connect to MCP servers for enhanced tool capabilities
- **Chat History**: Maintain conversation context with scrollable history
- **Tool Call Visualization**: See when and how agents use MCP tools

## Implementation Details

### Architecture

The application is built with a modular architecture:

- **UI Components**: Organized by functional areas (tabs, tools, displays)
- **Agent Management**: Handles agent creation and execution
- **System Utilities**: Environment management, async handling
- **Async Support**: Seamlessly handles async operations in Streamlit's sync environment

### Directory Structure

```
plans/basic_agent_runner/
├── agent_management/      # Agent execution and conversation management
│   ├── mcp_manager.py     # MCP server lifecycle management
│   └── agent_runner.py    # Agent execution logic
├── async_utils/           # Async-to-sync conversion utilities
├── streamlit_interface.py # Main application entry point
├── sys_utils/             # System utilities (env handling)
└── ui_components/         # UI components by functional area
    ├── chat_display.py
    ├── tool_display.py    # Enhanced for MCP tool visualization
    ├── tabs/             # Tab-specific UI components
    │   ├── chat_tab.py
    │   ├── configure_tab.py
    │   ├── history_tab.py
    │   ├── mcp_tab.py     # MCP server management
    │   ├── settings_tab.py
    │   └── tools_tab.py
    └── tools/            # Tool-specific UI components
        ├── mcp_tools.py  # MCP-specific tool components
        └── tool_list.py
```

## Development Status

### Completed

- ✅ Basic tabbed interface structure
- ✅ Agent configuration interface
- ✅ Chat interface with message history
- ✅ Tool calls visualization
- ✅ MCP server management interface
- ✅ MCP server integration with agents
- ✅ MCP tool visualization
- ✅ Agent execution with conversation history
- ✅ Enhanced error handling and debugging

### In Progress

- 🔄 Improved agent response handling
- 🔄 MCP server connection testing
- 🔄 History persistence and management

### Planned

- 📝 Streaming responses for improved UX
- 📝 Enhanced MCP tool discovery
- 📝 MCP server lifecycle management
- 📝 Export/import conversation functionality
- 📝 File upload for context
- 📝 Additional model providers beyond OpenAI

## Usage

1. Set up MCP servers in the MCP tab
2. Test and select MCP servers to use
3. Configure your agent in the Configure tab with MCP server integration
4. Chat with your agent in the Chat tab
5. View past conversations in the History tab
6. Manage API keys and settings in the Settings tab

## MCP Server Setup

To use MCP servers:

1. Install Node.js and NPM for local stdio servers
2. Install MCP servers (e.g., `npm install -g @modelcontextprotocol/server-filesystem`)
3. Configure the server in the MCP tab
4. Test the connection to verify tool availability
5. Select the server(s) to use with your agent

## Debugging

The application includes debug panels to help troubleshoot issues:
- Agent configuration details (in Chat tab)
- MCP server connection testing in MCP tab
- MCP tool discovery and parameters
- Response extraction fallback paths
- Detailed error traceback views
- Enhanced MCP tool call debugging
- Custom tool implementation testing
- Event loop management diagnostics

## MCP Integration Details

This application integrates with the Model Context Protocol to provide:

- Support for both stdio (local) and SSE (remote) MCP servers
- Tool discovery from MCP servers
- Visualization of MCP tool calls with specialized formatting
- Server lifecycle management
- Tool caching options for performance optimization

### MCP Tool Best Practices

When working with MCP tools:

1. **Parameter Handling**: MCP tools may report 0 parameters but still expect specific inputs
2. **Custom Fallbacks**: Implement parameter-free custom tools as fallbacks
3. **Error Handling**: Enhance logging for MCP tool calls to debug issues
4. **Event Loop Management**: Use proper async-to-sync conversions with loop persistence
5. **Result Inspection**: Always check both 'result' and 'error' fields in MCP tool responses
6. **API Compatibility**: Avoid default parameters in custom tools to prevent OpenAI API validation errors