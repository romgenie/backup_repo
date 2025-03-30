# Basic Agent Runner

A streamlined interface for interacting with AI agents using OpenAI's Agents SDK.

## Features

- **Tabbed Interface**: Navigate between Chat, Configure, Tools, History, and Settings tabs
- **Agent Configuration**: Set up agents with different models and parameters
- **Tools Management**: Add pre-built and custom tools to enhance agent capabilities
- **Chat History**: Maintain conversation context with scrollable history
- **Tool Call Visualization**: See when and how agents use tools

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
├── async_utils/           # Async-to-sync conversion utilities
├── streamlit_interface.py # Main application entry point
├── sys_utils/             # System utilities (env handling)
└── ui_components/         # UI components by functional area
    ├── chat_display.py
    ├── tool_display.py
    ├── tabs/             # Tab-specific UI components
    │   ├── chat_tab.py
    │   ├── configure_tab.py
    │   ├── history_tab.py
    │   ├── settings_tab.py
    │   └── tools_tab.py
    └── tools/            # Tool-specific UI components
        ├── custom_tools.py
        ├── example_tools.py
        ├── predefined_tools.py
        └── tool_list.py
```

## Development Status

### Completed

- ✅ Basic tabbed interface structure
- ✅ Agent configuration interface
- ✅ Chat interface with message history
- ✅ Tool calls visualization
- ✅ Example tool implementation
- ✅ Predefined tools support (WebSearch, FileSearch)
- ✅ Custom tool creation interface
- ✅ Agent execution with conversation history
- ✅ Enhanced error handling and debugging

### In Progress

- 🔄 Improved agent response handling
- 🔄 Tool selection and application
- 🔄 History persistence and management

### Planned

- 📝 Streaming responses for improved UX
- 📝 Enhanced visualization for agent execution flow
- 📝 Export/import conversation functionality
- 📝 File upload for context
- 📝 Additional model providers beyond OpenAI

## Usage

1. Configure your agent in the Configure tab
2. Add tools in the Tools tab
3. Chat with your agent in the Chat tab
4. View past conversations in the History tab
5. Manage API keys and settings in the Settings tab

## Debugging

The application includes debug panels to help troubleshoot issues:
- Agent configuration details (in Chat tab)
- Response extraction fallback paths
- Detailed error traceback views