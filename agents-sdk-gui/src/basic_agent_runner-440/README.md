# Basic Agent Runner

A Streamlit-based graphical user interface for running LLM agents with the OpenAI Agents SDK.

## Features

- Configure and run agents with different models and settings
- Chat interface with conversation history
- Tool integration for more capable agents
- Environment management for API keys
- Session management for continuing conversations

## Getting Started

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:
   - Add it to a `.env` file in the project root (will be created if it doesn't exist)
   - Or input it in the Settings tab of the application

### Running the Application

```bash
streamlit run streamlit_interface.py
```

## Usage Guide

### Configuring an Agent

1. Go to the **Configure** tab
2. Enter a name and instructions for your agent
3. Select the model and adjust settings
4. Click "Create Agent"

### Adding Tools

1. Go to the **Tools** tab
2. Choose from example tools:
   - **Weather Tool**: Get weather information for a location
   - **News Search Tool**: Search for news articles on a topic
   - **Calendar Tool**: Add events to a calendar
3. Or add predefined tools like WebSearchTool or FileSearchTool
4. Click "Add Tool" and then "Apply Selected Tools to Agent"

### Chatting with the Agent

1. Go to the **Chat** tab
2. Type your message in the input area
3. View the agent's responses in the conversation
4. Examine tool calls in the collapsible "Tool Calls" section

### Managing Conversations

- Use "Clear Chat" to start a new conversation
- View your conversation history in the History tab (coming soon)

## Project Structure

- `streamlit_interface.py`: Main application entry point
- `agent_management/`: Agent creation and execution
  - `agent_runner.py`: Core agent runner functionality
  - `conversation.py`: Conversation state management
  - `tools.py`: Tool definitions and implementations
- `ui_components/`: UI components and display utilities
  - `chat/`: Chat interface components
  - `config/`: Configuration interface components
  - `tabs/`: Tab interface components
- `async_utils/`: Asynchronous helpers and wrappers
- `sys_utils/`: Environment variables and system utilities

## Implementation Progress

### Completed ✅
- ✅ Basic Streamlit interface for agent execution
- ✅ Agent configuration form with name, instructions, model, temperature, max tokens
- ✅ Environment variable management for API keys
- ✅ Loading API keys from environment variables
- ✅ Saving and loading .env files
- ✅ Fixed async execution issues in Streamlit environment
- ✅ Added conversation history with proper session state management
- ✅ Implemented Streamlit-themed chat interface for conversations
- ✅ Added proper code block rendering in agent responses
- ✅ Improved error handling and recovery
- ✅ Added response preview in status messages
- ✅ Refactored codebase into modular components
- ✅ Added tool integration with function calling
- ✅ Created comprehensive modular component structure

### Next Steps
1. Add real-time progress reporting with token-by-token display
2. Add agent saving and loading functionality
3. Add support for additional model providers
4. Integrate visualization tools for agent execution flows
5. Add conversation history export/import functionality
6. Implement comprehensive testing for all modules

## License

This project is licensed under the MIT License - see the LICENSE file for details.