from typing import List, Dict, Any, Optional
from services.session_manager import SessionManager

def init_conversation():
    """
    Initialize the conversation history in the session state
    """
    SessionManager.init()

def add_user_message(content: str):
    """
    Add a user message to the conversation history
    
    Args:
        content: The message content
    """
    SessionManager.add_user_message(content)

def add_assistant_message(content: str):
    """
    Add an assistant message to the conversation history
    
    Args:
        content: The message content
    """
    SessionManager.add_assistant_message(content)

def clear_conversation():
    """
    Clear the conversation history and generate a new chat ID
    
    Returns:
        str: The new chat ID
    """
    return SessionManager.clear_conversation()

def get_conversation_history() -> List[Dict[str, Any]]:
    """
    Get the current conversation history
    
    Returns:
        The conversation history as a list of message dictionaries
    """
    return SessionManager.get_conversation_history()

def get_current_chat_id() -> str:
    """
    Get the current chat ID
    
    Returns:
        The current chat ID as a string
    """
    return SessionManager.get_current_chat_id()