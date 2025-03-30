# Basic Agent Runner Module Plan

## Overview
The Basic Agent Runner is one of the core components for the GUI implementation. It provides the functionality for creating and executing agents, handling their results, and managing the overall execution flow.

## Source Component
- **Original Path**: `/src/agents/run.py`
- **Key Functionality**: Provides core runner functions for synchronous and asynchronous agent execution

## Implementation Plan

### 1. Backend Components
- Create a Runner service class that wraps the SDK's Runner functionality
- Implement both synchronous and asynchronous execution methods
- Add event emitting capabilities for progress updates

### 2. GUI Components
- Agent execution control panel
  - Start/stop buttons
  - Input field for agent instructions
  - Execution status indicator
- Results display panel
  - Output text area
  - Tool usage visualization
  - Execution history
- Environment Variables Management
  - Input fields for setting API keys and other environment variables
  - Apply/reset capabilities
- Agent Saving and Loading
  - Save agent configuration to file
  - Generate executable Python code
  - Load saved agents

### 3. Integration Points
- Connect with visualization module for displaying agent relationships
- Integrate with SSE for real-time progress updates
- Hook into streaming APIs for incremental result display

## Streamlit Implementation

### Key Features
1. **Tabbed Interface**
   - Agent Runner tab
   - Environment Variables tab
   - Saved Agents tab

2. **Environment Variables Management**
   - Add/remove environment variables
   - Securely store API keys
   - Apply variables to the current session

3. **Agent Configuration**
   - Name and instructions setup
   - Model selection and parameter tuning
   - Run settings configuration

4. **Agent Execution**
   - Real-time streaming of responses
   - Tool call tracking
   - Status indicators

5. **Agent Persistence**
   - Save agent configurations
   - Generate executable Python code
   - Load saved agents for reuse

## API Design

```python
class AgentRunnerApp:
    """Streamlit application for running agents"""
    
    def run(self):
        """Main entry point for the Streamlit app"""
    
    def _show_agent_runner(self):
        """Display the agent runner interface"""
    
    def _show_environment_vars(self):
        """Display environment variables management UI"""
    
    def _show_saved_agents(self):
        """Display saved agents management UI"""
    
    def _generate_agent_code(self, agent_config):
        """Generate Python code for creating an agent"""
```

## UI Mockup

```
+--------------------------------------------------+
| Agent Runner | Environment Vars | Saved Agents    |
+--------------------------------------------------+
| [Sidebar]                                        |
| Agent Name: [______________________]             |
| Instructions: [                     ]            |
|               [                     ]            |
|                                                  |
| Model: [gpt-4o_______________▼]                  |
| Temperature: [0.0-------|------1.0]  0.7         |
| Max Tokens: [__1000__]                           |
|                                                  |
| [Create Agent] [Save Agent]                      |
+--------------------------------------------------+
| User Input:                                      |
| [                                              ] |
| [                                              ] |
|                                                  |
| [Run Agent]     Status: Completed                |
|                                                  |
| Results:                                         |
| +----------------------------------------------+ |
| | Output: This is the agent's response...      | |
| |                                              | |
| +----------------------------------------------+ |
|                                                  |
| Tool Calls:                                      |
| +----------------------------------------------+ |
| | - get_weather (12:05:23): {"city": "Tokyo"}  | |
| +----------------------------------------------+ |
+--------------------------------------------------+
```

## Implementation Challenges and Progress

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
  - ✅ Environment utilities (sys_utils)
  - ✅ Async utilities (async_utils)
  - ✅ UI components (ui_components)
  - ✅ Agent management (agent_management)

### In Progress 🔄
- Implementing tabbed interface for better organization
- Handling long-running executions
- Displaying streaming results efficiently
- Managing connection state for real-time updates

## Implementation Challenges
- Integrating the streaming API for real-time token generation
- Supporting multiple model providers
- Managing complex conversation state across multiple reruns in Streamlit
- Balancing modularity with maintainability
- Designing intuitive UI for complex agent configurations

## Next Steps
1. ✅ Create basic service layer for Runner functionality
2. ✅ Implement environment variable management for API keys
3. ✅ Add conversation history with proper session state management
4. ✅ Refactor codebase into modular components
5. Implement tabbed interface with agent execution, environment vars, and saved agents
   - Chat tab (conversation interface)
   - Configuration tab (agent setup)
   - Tools tab (custom tool configuration)
   - History tab (conversation management)
   - Settings tab (application settings)
6. Add real-time progress reporting with token-by-token display
7. Add tool configuration interface for custom tools
8. Add agent saving and loading functionality
9. Add support for additional model providers
10. Integrate visualization tools for agent execution flows
11. Add conversation history export/import functionality
12. Implement comprehensive testing for all modules