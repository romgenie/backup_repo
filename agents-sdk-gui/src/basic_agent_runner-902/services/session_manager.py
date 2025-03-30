"""
Session manager module for the Agent Runner application.
Centralizes all session state operations and provides a consistent interface
for managing session state across the application.
"""

import streamlit as st
import uuid
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable

# Type for session state values
T = TypeVar('T')

class SessionManager:
    """
    Centralized session state manager for streamlit applications.
    Provides methods for getting, setting, and checking session state values.
    """
    
    @staticmethod
    def init() -> None:
        """Initialize required session state variables with default values."""
        # Initialize conversation history
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
            print("Initializing empty conversation history")
        
        # Force the conversation_history to be a list if it isn't already
        if not isinstance(st.session_state.conversation_history, list):
            print(f"Fixing conversation history type: {type(st.session_state.conversation_history)}")
            st.session_state.conversation_history = []
        
        # Initialize a unique chat ID for the current session
        if "current_chat_id" not in st.session_state:
            st.session_state.current_chat_id = str(uuid.uuid4())
        
        # Initialize MCP related state
        if "mcp_servers" not in st.session_state:
            st.session_state.mcp_servers = {}
            
        if "selected_mcp_servers" not in st.session_state:
            st.session_state.selected_mcp_servers = []
            
        # Initialize user input field
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

    @staticmethod
    def get(key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get a value from session state with a default fallback.
        
        Args:
            key: The session state key
            default: Default value if key doesn't exist
            
        Returns:
            The value from session state or the default
        """
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Set a value in session state.
        
        Args:
            key: The session state key
            value: The value to set
        """
        st.session_state[key] = value

    @staticmethod
    def has(key: str) -> bool:
        """
        Check if a key exists in session state.
        
        Args:
            key: The session state key
            
        Returns:
            True if the key exists, False otherwise
        """
        return key in st.session_state
    
    @staticmethod
    def delete(key: str) -> None:
        """
        Delete a key from session state if it exists.
        
        Args:
            key: The session state key to delete
        """
        if key in st.session_state:
            del st.session_state[key]
    
    @staticmethod
    def clear_all() -> None:
        """
        Clear all session state (dangerous - use with caution).
        """
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    
    @staticmethod
    def debug() -> Dict[str, Any]:
        """
        Return a dictionary of all session state keys and their types.
        Useful for debugging.
        
        Returns:
            Dictionary of session state keys and value types
        """
        return {key: type(value).__name__ for key, value in st.session_state.items()}
    
    @staticmethod
    def print_debug() -> None:
        """Print debug information about the current session state."""
        print("\nSESSION STATE DEBUG INFO:")
        print(f"Number of keys: {len(st.session_state)}")
        
        for key, value in st.session_state.items():
            value_type = type(value).__name__
            value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            print(f"  {key}: ({value_type}) {value_preview}")
    
    # Conversation-specific methods
    @staticmethod
    def add_user_message(content: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            content: The message content
        """
        if not SessionManager.has("conversation_history"):
            SessionManager.init()
            
        st.session_state.conversation_history.append({
            "role": "user",
            "content": content
        })
        print(f"Added user message: {content[:50]}...")

    @staticmethod
    def add_assistant_message(content: str) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            content: The message content
        """
        if not SessionManager.has("conversation_history"):
            SessionManager.init()
            
        # Debug
        print(f"Adding assistant message: {content[:100]}...")
        
        # Make sure we have a valid message
        if not content or not isinstance(content, str):
            print(f"WARNING: Invalid assistant message content type: {type(content)}")
            content = str(content) if content else "I'm sorry, I couldn't generate a proper response."
        
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": content
        })
        print(f"Added assistant message. History now has {len(st.session_state.conversation_history)} messages")

    @staticmethod
    def clear_conversation() -> str:
        """
        Clear the conversation history and generate a new chat ID.
        
        Returns:
            The new chat ID
        """
        print("Clearing conversation history")
        # Reset conversation history
        st.session_state.conversation_history = []
        # Generate a new chat ID
        new_chat_id = str(uuid.uuid4())
        st.session_state.current_chat_id = new_chat_id
        return new_chat_id

    @staticmethod
    def get_conversation_history() -> List[Dict[str, Any]]:
        """
        Get the current conversation history.
        
        Returns:
            The conversation history as a list of message dictionaries
        """
        if not SessionManager.has("conversation_history"):
            SessionManager.init()
        return st.session_state.conversation_history

    @staticmethod
    def get_current_chat_id() -> str:
        """
        Get the current chat ID.
        
        Returns:
            The current chat ID as a string
        """
        if not SessionManager.has("current_chat_id"):
            SessionManager.init()
        return st.session_state.current_chat_id
    
    # Agent-related methods
    @staticmethod
    def has_agent() -> bool:
        """
        Check if an agent is available in the session.
        
        Returns:
            True if an agent is available, False otherwise
        """
        has_agent = "agent" in st.session_state and st.session_state.agent is not None
        print(f"DEBUG: has_agent check: {has_agent}, 'agent' in keys: {'agent' in st.session_state}")
        return has_agent
    
    @staticmethod
    def set_agent(agent: Any, run_config: Any = None) -> None:
        """
        Set the agent and optionally run config in session state.
        
        Args:
            agent: The agent object
            run_config: Optional run configuration
        """
        # Make sure we're directly updating session state
        st.session_state["agent"] = agent
        if run_config is not None:
            st.session_state["run_config"] = run_config
            
        # Print debug information
        print(f"DEBUG: Set agent in SessionManager: {agent.name if agent else 'None'}")
        print(f"DEBUG: Checking session state after setting: {'agent' in st.session_state}") 
        print(f"DEBUG: Agent in session state: {st.session_state.get('agent') is not None}")
    
    @staticmethod
    def get_agent() -> Optional[Any]:
        """
        Get the current agent from session state.
        
        Returns:
            The agent or None if not available
        """
        return st.session_state.get("agent")
    
    @staticmethod
    def get_run_config() -> Optional[Any]:
        """
        Get the current run config from session state.
        
        Returns:
            The run config or None if not available
        """
        return st.session_state.get("run_config")
    
    # Flag operations for managing Streamlit reruns
    @staticmethod
    def set_flag(flag_name: str, value: bool = True) -> None:
        """
        Set a boolean flag in session state.
        
        Args:
            flag_name: Name of the flag
            value: Value to set (default: True)
        """
        st.session_state[flag_name] = value
    
    @staticmethod
    def clear_flag(flag_name: str) -> None:
        """
        Clear a flag by setting it to False.
        
        Args:
            flag_name: Name of the flag
        """
        st.session_state[flag_name] = False
    
    @staticmethod
    def check_flag(flag_name: str) -> bool:
        """
        Check if a flag is set and true.
        
        Args:
            flag_name: Name of the flag
            
        Returns:
            True if the flag exists and is True, False otherwise
        """
        return st.session_state.get(flag_name, False)
    
    @staticmethod
    def execute_once(flag_name: str, function: Callable, *args, **kwargs) -> Any:
        """
        Execute a function once and set a flag to prevent re-execution.
        
        Args:
            flag_name: Flag to check and set
            function: Function to execute
            args: Arguments for the function
            kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
        """
        result = None
        if not SessionManager.check_flag(flag_name):
            result = function(*args, **kwargs)
            SessionManager.set_flag(flag_name)
        return result
    
    # MCP server methods
    @staticmethod
    def add_mcp_server(server_id: str, server_config: Dict[str, Any]) -> None:
        """
        Add an MCP server configuration to session state.
        
        Args:
            server_id: The server ID
            server_config: Server configuration dictionary
        """
        if "mcp_servers" not in st.session_state:
            st.session_state.mcp_servers = {}
        st.session_state.mcp_servers[server_id] = server_config
    
    @staticmethod
    def remove_mcp_server(server_id: str) -> None:
        """
        Remove an MCP server configuration from session state.
        
        Args:
            server_id: The server ID to remove
        """
        if "mcp_servers" in st.session_state and server_id in st.session_state.mcp_servers:
            del st.session_state.mcp_servers[server_id]
        
        # Also remove from selected servers if present
        if "selected_mcp_servers" in st.session_state and server_id in st.session_state.selected_mcp_servers:
            st.session_state.selected_mcp_servers.remove(server_id)
    
    @staticmethod
    def get_mcp_servers() -> Dict[str, Dict[str, Any]]:
        """
        Get all MCP server configurations.
        
        Returns:
            Dictionary of server IDs to server configs
        """
        return st.session_state.get("mcp_servers", {})
    
    @staticmethod
    def select_mcp_server(server_id: str) -> None:
        """
        Add a server ID to the selected servers list.
        
        Args:
            server_id: The server ID to select
        """
        if "selected_mcp_servers" not in st.session_state:
            st.session_state.selected_mcp_servers = []
            
        if server_id not in st.session_state.selected_mcp_servers:
            st.session_state.selected_mcp_servers.append(server_id)
    
    @staticmethod
    def deselect_mcp_server(server_id: str) -> None:
        """
        Remove a server ID from the selected servers list.
        
        Args:
            server_id: The server ID to deselect
        """
        if "selected_mcp_servers" in st.session_state and server_id in st.session_state.selected_mcp_servers:
            st.session_state.selected_mcp_servers.remove(server_id)
    
    @staticmethod
    def get_selected_mcp_servers() -> List[str]:
        """
        Get list of selected server IDs.
        
        Returns:
            List of selected server IDs
        """
        return st.session_state.get("selected_mcp_servers", [])
    
    @staticmethod
    def set_selected_mcp_servers(server_ids: List[str]) -> None:
        """
        Set the list of selected server IDs.
        
        Args:
            server_ids: List of server IDs to select
        """
        st.session_state.selected_mcp_servers = server_ids