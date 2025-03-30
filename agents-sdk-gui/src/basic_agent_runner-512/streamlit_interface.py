import streamlit as st
import asyncio
from typing import Optional, Dict, List, Any, Callable
import time
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Import the Agents SDK
from agents import Agent, Runner, RunConfig, RunResult
from agents.model_settings import ModelSettings
from openai.types.responses import ResponseTextDeltaEvent

# Import our utility modules
from sys_utils import save_to_env_file, load_from_env_file
from async_utils import async_to_sync
from ui_components import display_chat, display_tool_calls
from ui_components.tabs import (
    render_chat_tab, render_configure_tab, render_tools_tab,
    render_history_tab, render_settings_tab
)
from agent_management import (
    init_conversation, add_user_message, add_assistant_message, 
    clear_conversation, run_agent_with_history
)

class AgentRunnerApp:
    def __init__(self):
        self.running_agents: Dict[str, RunResult] = {}
        
        # Load .env file if it exists
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path)
        
        # Get OpenAI API key from environment
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.env_file_path = dotenv_path if dotenv_path else str(Path.cwd() / ".env")
        
        # Initialize conversation state
        init_conversation()
        
    def run(self):
        st.title("Agent Runner")
        
        # Create tabs to organize the UI
        chat_tab, config_tab, tools_tab, history_tab, settings_tab = st.tabs([
            "Chat", "Configure", "Tools", "History", "Settings"
        ])
        
        # Render content in each tab directly
        with chat_tab:
            # Render chat tab
            agent = st.session_state.get("agent", None)
            run_config = st.session_state.get("run_config", None)
            render_chat_tab(
                agent=agent,
                run_config=run_config,
                running_agents=self.running_agents,
                on_clear_chat=lambda: setattr(self, "running_agents", {})
            )
        
        with config_tab:
            # Render the configuration tab
            render_configure_tab(
                api_key=self.api_key,
                env_file_path=self.env_file_path,
                on_agent_create=self._on_agent_create,
                on_api_key_save=self._on_api_key_save,
                on_api_key_load=self._on_api_key_load
            )
        
        with tools_tab:
            # Render the tools tab
            render_tools_tab()
        
        with history_tab:
            # Render the history tab
            render_history_tab()
        
        with settings_tab:
            # Render the settings tab
            render_settings_tab(
                api_key=self.api_key,
                env_file_path=self.env_file_path,
                on_api_key_save=self._on_api_key_save,
                on_api_key_load=self._on_api_key_load
            )
    
    def _on_agent_create(self, agent, run_config):
        """Callback when an agent is created"""
        st.session_state.agent = agent
        st.session_state.run_config = run_config
    
    def _on_api_key_save(self, api_key):
        """Callback to save API key"""
        # Store API key for future use
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            self.api_key = api_key
        
        # Save to env file
        return save_to_env_file(self.env_file_path, api_key)
    
    def _on_api_key_load(self, env_file_path):
        """Callback to load API key"""
        loaded_api_key, success = load_from_env_file(env_file_path)
        if success:
            self.api_key = loaded_api_key
        
        return loaded_api_key, success
    # Class contains only the run method now
    # All other functionality has been moved to separate modules

if __name__ == "__main__":
    app = AgentRunnerApp()
    app.run()