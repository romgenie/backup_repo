"""
Agent Runner Dashboard

A dashboard UI for the Agent Runner application that displays
the model configuration, active agents, and other statistics.
"""

import streamlit as st
import os
import time
from datetime import datetime
from pathlib import Path

# Import components
from model_selector import model_selector_card, render_model_selector
from model_management import get_default_model, create_agent_with_model
from setup_wizard import SetupWizard
from responsive_components import (
    responsive_container, 
    enhanced_chat_container,
    responsive_status_bar,
    detect_screen_size
)

# Import agent tools
from agents import Agent, RunConfig, RunResult, function_tool

# Define custom tools
@function_tool
def get_time(timezone: str) -> str:
    """
    Get the current time in a specific timezone.
    
    Args:
        timezone: The timezone to get the time for (e.g., "UTC", "US/Pacific", "Europe/London")
                 Use "local" for the local system time.
    
    Returns:
        A string with the current time information
    """
    import datetime
    import pytz
    from zoneinfo import ZoneInfo
    
    # Check for "local" timezone request
    if not timezone or timezone.lower() == "local":
        now = datetime.datetime.now()
        local_tz = datetime.datetime.now().astimezone().tzinfo
        return f"The current local time is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')} ({local_tz})"
    
    # Try to parse the timezone
    try:
        # First try zoneinfo
        try:
            if timezone in ZoneInfo.available_timezones():
                tz = ZoneInfo(timezone)
            else:
                # Fall back to pytz
                tz = pytz.timezone(timezone)
        except (AttributeError, ImportError):
            # If ZoneInfo.available_timezones() is not available, go straight to pytz
            tz = pytz.timezone(timezone)
        
        # Get the current time in the requested timezone
        now = datetime.datetime.now(tz)
        return f"The current time in {timezone} is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}"
        
    except Exception as e:
        # Handle common timezone abbreviations
        abbrev_map = {
            "EST": "US/Eastern", "CST": "US/Central", 
            "MST": "US/Mountain", "PST": "US/Pacific",
            "GMT": "GMT", "UTC": "UTC"
        }
        if timezone and timezone.upper() in abbrev_map:
            try:
                tz = pytz.timezone(abbrev_map[timezone.upper()])
                now = datetime.datetime.now(tz)
                return f"The current time in {timezone.upper()} ({abbrev_map[timezone.upper()]}) is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}"
            except:
                pass
        
        return f"Sorry, I couldn't recognize the timezone '{timezone}'. Please try a valid timezone like 'UTC', 'US/Pacific', or 'Europe/London'."

@function_tool
def get_weather(location: str, units: str) -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and country (e.g., "London,UK")
        units: The units to use (metric or imperial)
        
    Returns:
        A description of the current weather
    """
    # This is a mock implementation - in a real application, you would call a weather API
    
    # Default to metric if units value is not valid
    if not units or (units != "metric" and units != "imperial"):
        units = "metric"
        
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temperature = 20 if units == "metric" else 68
    
    # Handle empty location
    if not location:
        location = "Unknown location"
        
    condition = weather_conditions[hash(location + str(datetime.date.today())) % len(weather_conditions)]
    
    return f"The weather in {location} is currently {condition} with a temperature of {temperature}°{'C' if units == 'metric' else 'F'}"

@function_tool
def search_news(query: str, max_results: int) -> str:
    """
    Search for news articles matching a query.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (1-10)
        
    Returns:
        A list of news headlines and snippets
    """
    # This is a mock implementation - in a real application, you would call a news API
    mock_results = [
        {"title": f"Latest developments in {query}", "snippet": f"New research shows promising results related to {query}..."},
        {"title": f"Analysis: The impact of {query} on industry", "snippet": f"Experts weigh in on how {query} is changing the landscape..."},
        {"title": f"Interview with {query} specialist", "snippet": f"We spoke with leading researchers about their work on {query}..."},
        {"title": f"{query} breakthrough announced", "snippet": f"A major discovery in the field of {query} was announced today..."},
    ]
    
    # Ensure max_results is an integer
    try:
        max_results_int = int(max_results)
    except (ValueError, TypeError):
        max_results_int = 3  # Fallback if conversion fails
        
    results = mock_results[:min(max_results_int, len(mock_results))]
    formatted_results = "\n\n".join([f"**{r['title']}**\n{r['snippet']}" for r in results])
    
    return f"Found {len(results)} results for '{query}':\n\n{formatted_results}"

def render_dashboard():
    """Render the main dashboard UI."""
    st.title("Agent Runner Dashboard")
    
    # Dashboard overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Model information card
        model_selector_card()
    
    with col2:
        # Agent statistics card
        agent_count = len(st.session_state.get("agents", {}))
        active_agent = st.session_state.get("agent_name", "")
        
        st.markdown("""
        <div class="info-card">
            <div class="info-card-header">
                <h3>Agent Status</h3>
            </div>
            <div class="info-card-body">
                <div class="agent-stats">
                    <div class="stat-item">
                        <span class="stat-value">{}</span>
                        <span class="stat-label">Available Agents</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">{}</span>
                        <span class="stat-label">Active Agent</span>
                    </div>
                </div>
            </div>
            <div class="info-card-footer">
                <button class="card-action-btn" onclick="Streamlit.setComponentValue('show_agent_wizard', true)">
                    Create New Agent
                </button>
            </div>
        </div>
        <style>
        .info-card {
            background-color: rgba(65, 105, 225, 0.05);
            border: 1px solid rgba(65, 105, 225, 0.15);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            height: calc(100% - 32px);
            display: flex;
            flex-direction: column;
            transition: all 0.3s ease;
        }
        .info-card:hover {
            border-color: rgba(65, 105, 225, 0.3);
            box-shadow: 0 4px 12px rgba(65, 105, 225, 0.1);
        }
        .agent-stats {
            display: flex;
            justify-content: space-around;
            margin: 16px 0;
        }
        .stat-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: #4169E1;
            margin-bottom: 4px;
        }
        .stat-label {
            font-size: 14px;
            color: #626E7F;
        }
        .card-action-btn {
            background-color: rgba(65, 105, 225, 0.1);
            color: #4169E1;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s ease;
        }
        .card-action-btn:hover {
            background-color: rgba(65, 105, 225, 0.2);
        }
        .info-card-footer {
            margin-top: auto;
            text-align: center;
        }
        </style>
        """.format(agent_count, active_agent or "None"), unsafe_allow_html=True)
    
    with col3:
        # Tools card
        tools = [get_time, get_weather, search_news]
        
        st.markdown("""
        <div class="info-card tools-card">
            <div class="info-card-header">
                <h3>Available Tools</h3>
            </div>
            <div class="info-card-body">
                <ul class="tools-list">
                    {}
                </ul>
            </div>
            <div class="info-card-footer">
                <button class="card-action-btn" onclick="Streamlit.setComponentValue('show_tools_config', true)">
                    Configure Tools
                </button>
            </div>
        </div>
        <style>
        .tools-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .tools-list li {
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 8px;
            background-color: rgba(65, 105, 225, 0.05);
            display: flex;
            align-items: center;
        }
        .tools-list li:before {
            content: "🧰";
            margin-right: 8px;
        }
        .tools-list li:nth-child(1):before {
            content: "🕒";
        }
        .tools-list li:nth-child(2):before {
            content: "🌤️";
        }
        .tools-list li:nth-child(3):before {
            content: "🔍";
        }
        </style>
        """.format('\n'.join([f"<li>{tool.__name__}</li>" for tool in tools])), unsafe_allow_html=True)
    
    # Quick actions
    st.subheader("Quick Actions")
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Create Agent", type="primary"):
            st.session_state.show_wizard = True
    
    with col2:
        if st.button("Configure Models"):
            st.session_state.show_model_management = True
    
    with col3:
        if st.button("Launch Chat"):
            st.session_state.show_chat = True
            
    with col4:
        if st.button("View Tools"):
            st.session_state.show_tools_config = True
    
    # Recent activity
    st.subheader("Recent Activity")
    
    # Chat interface if active
    if st.session_state.get("show_chat", False):
        st.markdown("### Chat Interface")
        
        # Check if we have an agent
        agent = st.session_state.get("agent")
        
        if not agent:
            st.warning("No agent is active. Please create an agent first.")
            
            if st.button("Create Agent Now"):
                st.session_state.show_wizard = True
        else:
            # Display agent info
            st.markdown(f"Chatting with: **{st.session_state.agent_name}**")
            
            # Chat container
            chat_container = enhanced_chat_container()
            
            # Initialize conversation if needed
            if "conversation" not in st.session_state:
                st.session_state.conversation = [
                    {"role": "assistant", "content": """Hello! I'm your AI assistant. I can help with:
• Weather updates and forecasts
• Time and scheduling information
• Searching for information on topics

What would you like assistance with today?"""},
                ]
            
            # Display conversation
            with chat_container:
                for message in st.session_state.conversation:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="user-message">
                            <div class="user-message-content">{message['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="assistant-message">
                            <div class="assistant-message-content">{message['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Check and clear input field if flag is set
            if st.session_state.get("should_clear_input", False):
                st.session_state.should_clear_input = False
                st.session_state.user_input = ""
                
            # Tool calls container
            tool_calls_container = st.container()
            
            # Status placeholder
            status_placeholder = st.empty()
            
            # Input area
            user_input = st.text_input("Type your message", key="user_input")
            col1, col2 = st.columns([4, 1])
            
            with col2:
                if st.button("Send", type="primary"):
                    if user_input:
                        # Add user message to conversation display
                        st.session_state.conversation.append({"role": "user", "content": user_input})
                        
                        # Get agent from session state
                        agent = st.session_state.get("agent")
                        run_config = st.session_state.get("run_config")
                        
                        if not agent:
                            status_placeholder.error("Agent is not configured properly")
                        else:
                            # Show thinking spinner
                            with st.spinner("Agent is thinking..."):
                                try:
                                    # Set up current chat ID if not present
                                    if "current_chat_id" not in st.session_state:
                                        st.session_state.current_chat_id = f"chat_{int(time.time())}"
                                    
                                    # Create a placeholder for tool calls
                                    with tool_calls_container:
                                        tool_calls_placeholder = st.empty()
                                        tool_calls_placeholder.markdown("*Processing tool calls...*")
                                    
                                    # Get the chat ID and running agents dictionary
                                    chat_id = st.session_state.current_chat_id
                                    if "running_agents" not in st.session_state:
                                        st.session_state.running_agents = {}
                                    running_agents = st.session_state.running_agents
                                    
                                    # Prepare agent input with history if available
                                    if chat_id in running_agents:
                                        previous_result = running_agents[chat_id]
                                        # Create conversation context
                                        input_list = previous_result.to_input_list()
                                        agent_input = input_list + [{"role": "user", "content": user_input}]
                                    else:
                                        # First message in conversation
                                        agent_input = user_input
                                    
                                    # Run the agent
                                    result = None
                                    try:
                                        # Run the agent with our own implementation
                                        from async_utils.async_to_sync import async_to_sync
                                        
                                        # Define sync wrapper for running the agent
                                        @async_to_sync
                                        async def run_agent(agent, user_input, max_turns, run_config):
                                            """Run the agent and return the result."""
                                            from agents import Runner
                                            try:
                                                # The first parameter to Runner.run should be the agent object directly, not a named parameter
                                                return await Runner.run(
                                                    agent,  # Pass agent as first positional argument
                                                    input=user_input,
                                                    max_turns=max_turns,
                                                    run_config=run_config
                                                )
                                            except Exception as e:
                                                print(f"Error in Runner.run: {str(e)}")
                                                raise
                                        
                                        # Run the agent
                                        result = run_agent(
                                            agent=agent,
                                            user_input=agent_input,
                                            max_turns=10,  # Default max turns
                                            run_config=run_config
                                        )
                                        
                                        # Process the result to extract the response
                                        response_content = ""
                                        
                                        # Try to get from output or final_output
                                        if hasattr(result, 'output') and result.output:
                                            response_content = result.output
                                        elif hasattr(result, 'final_output') and result.final_output:
                                            response_content = result.final_output
                                        # Try to extract from steps if available
                                        elif hasattr(result, 'steps') and result.steps:
                                            for step in result.steps:
                                                if hasattr(step, 'messages') and step.messages:
                                                    for msg in step.messages:
                                                        if hasattr(msg, 'role') and msg.role == 'assistant' and hasattr(msg, 'content'):
                                                            # If we already have content, append this as continuation
                                                            if response_content:
                                                                if msg.content not in response_content:
                                                                    response_content += "\n\n" + msg.content
                                                            else:
                                                                response_content = msg.content
                                        
                                        # Store result for future context
                                        running_agents[chat_id] = result
                                        
                                        # Show tool calls if available
                                        if hasattr(result, 'steps') and result.steps:
                                            tool_calls_all = []
                                            for step in result.steps:
                                                if hasattr(step, 'tool_calls') and step.tool_calls:
                                                    for tc in step.tool_calls:
                                                        tool_calls_all.append({
                                                            "name": tc.name,
                                                            "args": getattr(tc, 'args', {}),
                                                            "response": getattr(tc, 'response', 'No response')
                                                        })
                                            
                                            if tool_calls_all:
                                                tool_calls_markdown = "### Tool Calls\n"
                                                for i, tc in enumerate(tool_calls_all):
                                                    tool_calls_markdown += f"**{tc['name']}**\n"
                                                    tool_calls_markdown += f"Arguments: `{tc['args']}`\n"
                                                    tool_calls_markdown += f"Result: {tc['response']}\n\n"
                                                
                                                tool_calls_placeholder.markdown(tool_calls_markdown)
                                            else:
                                                tool_calls_placeholder.markdown("*No tool calls made*")
                                        else:
                                            tool_calls_placeholder.markdown("*No tool calls made*")
                                        
                                        # Add the response to conversation
                                        if response_content:
                                            st.session_state.conversation.append({"role": "assistant", "content": response_content})
                                        else:
                                            # Fallback message if no response content could be extracted
                                            fallback = "I processed your request but couldn't generate a proper response. Please try again."
                                            st.session_state.conversation.append({"role": "assistant", "content": fallback})
                                            
                                    except Exception as e:
                                        import traceback
                                        error_msg = f"Error running agent: {str(e)}"
                                        st.session_state.conversation.append({"role": "assistant", "content": f"Error: {error_msg}"})
                                        status_placeholder.error(error_msg)
                                        with st.expander("Error Details"):
                                            st.code(traceback.format_exc())
                                
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        
                        # Set a flag to clear input on next run instead of directly modifying the widget
                        st.session_state.should_clear_input = True
                        st.rerun()
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Clear Chat"):
                    # Reset conversation but keep agent
                    st.session_state.conversation = [
                        {"role": "assistant", "content": """Hello! I'm your AI assistant. I can help with:
• Weather updates and forecasts
• Time and scheduling information  
• Searching for information on topics

What would you like assistance with today?"""},
                    ]
                    
                    # Create a new chat ID to reset conversation history
                    st.session_state.current_chat_id = f"chat_{int(time.time())}"
                    
                    # Clear the tool calls display
                    with tool_calls_container:
                        st.empty()
                    
                    st.rerun()
            with col2:
                if st.button("Configure Agent"):
                    st.session_state.show_wizard = True
                    st.rerun()
            with col3:
                if st.button("Close Chat"):
                    st.session_state.show_chat = False
                    st.rerun()
    
    # Show the model management if requested
    if st.session_state.get("show_model_management", False):
        with st.expander("Model Management", expanded=True):
            from model_management import render_model_management
            render_model_management()
            
            if st.button("Close Management"):
                st.session_state.show_model_management = False
                st.rerun()
    
    # Show the setup wizard if requested
    if st.session_state.get("show_wizard", False):
        with st.expander("Agent Setup", expanded=True):
            def api_key_save(key, save_to_file, file_path):
                """Save API key to environment and optionally to file."""
                import os
                
                # Always set environment variable
                os.environ["OPENAI_API_KEY"] = key
                
                # Save to file if requested
                if save_to_file:
                    try:
                        from pathlib import Path
                        env_path = Path(file_path)
                        
                        # Create directory if it doesn't exist
                        env_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Save the key
                        with open(env_path, "w") as f:
                            f.write(f"OPENAI_API_KEY={key}\n")
                            
                        st.toast(f"API key saved to environment and {file_path}")
                    except Exception as e:
                        st.error(f"Error saving to {file_path}: {str(e)}")
                        st.toast(f"API key saved to environment only (file error)")
                else:
                    st.toast(f"API key saved to environment only")
                
                return key
            
            def api_key_load(file_path):
                """Load API key from file."""
                try:
                    from pathlib import Path
                    env_path = Path(file_path)
                    
                    if env_path.exists():
                        # Read the file
                        with open(env_path, "r") as f:
                            env_content = f.read()
                            
                        # Look for the API key
                        import re
                        api_key_match = re.search(r'OPENAI_API_KEY=([^\n]+)', env_content)
                        
                        if api_key_match:
                            api_key = api_key_match.group(1).strip()
                            return api_key, True
                            
                    return "", False
                except Exception as e:
                    st.error(f"Error loading from {file_path}: {str(e)}")
                    return "", False
            
            def agent_create(config, run_config):
                """Create an agent using the configuration."""
                if not config:
                    return False
                
                try:
                    # Get model info from the model management
                    provider, model_id, params = get_default_model()
                    
                    # Create a real agent with the configured tools and model
                    agent = Agent(
                        model=model_id,
                        tools=[get_weather, get_time, search_news],
                        instructions="""You are a helpful AI assistant that can provide information about:
                        - Current time in different timezones
                        - Weather information for locations
                        - Search for news and information
                        
                        Always be polite, concise, and provide accurate information using your tools.
                        When you don't know something, admit it and suggest using one of your tools if appropriate.
                        """,
                        name=config["name"]
                    )
                    
                    # Store the agent in session state
                    st.session_state.agent = agent
                    st.session_state.agent_name = config["name"]
                    st.session_state.run_config = RunConfig(workflow_name=config["name"])
                    
                    # Store in agents dictionary
                    if "agents" not in st.session_state:
                        st.session_state.agents = {}
                    
                    st.session_state.agents[config["name"]] = {
                        "agent": agent,
                        "model": model_id,
                        "provider": provider,
                        "created_at": datetime.now().isoformat(),
                    }
                    
                    # Initialize running_agents dictionary if not present
                    if "running_agents" not in st.session_state:
                        st.session_state.running_agents = {}
                    
                    st.balloons()
                    return True
                except Exception as e:
                    st.error(f"Error creating agent: {str(e)}")
                    return False
            
            # Create and render the setup wizard
            wizard = SetupWizard(
                on_api_key_save=api_key_save,
                on_api_key_load=api_key_load,
                on_agent_create=agent_create
            )
            wizard.render()
            
            # If wizard is completed, close it and show chat
            if wizard.is_completed():
                st.session_state.show_wizard = False
                st.session_state.show_chat = True
                st.rerun()
            
            if st.button("Close Setup"):
                st.session_state.show_wizard = False
                st.rerun()

if __name__ == "__main__":
    # Page configuration
    st.set_page_config(
        page_title="Agent Runner Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    # Load CSS styles
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(current_dir, "enhanced_styles.css")) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        st.warning("Enhanced styles CSS file not found.")
    
    # Initialize state
    if "show_model_management" not in st.session_state:
        st.session_state.show_model_management = False
    if "show_model_mgmt" not in st.session_state:
        st.session_state.show_model_mgmt = False
    if "show_wizard" not in st.session_state:
        st.session_state.show_wizard = False
    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False
    if "show_tools_config" not in st.session_state:
        st.session_state.show_tools_config = False
    
    # Render the dashboard
    render_dashboard()