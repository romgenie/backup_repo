import streamlit as st
import uuid
from typing import List, Dict, Any, Optional

def init_conversation():
    """
    Initialize the conversation history in the session state
    """
    # Initialize session state for conversation history with default empty list
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

def add_user_message(content: str):
    """
    Add a user message to the conversation history
    
    Args:
        content: The message content
    """
    st.session_state.conversation_history.append({
        "role": "user",
        "content": content
    })

def add_assistant_message(content: str):
    """
    Add an assistant message to the conversation history
    
    Args:
        content: The message content
    """
    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": content
    })

def clear_conversation():
    """
    Clear the conversation history and generate a new chat ID
    
    Returns:
        str: The new chat ID
    """
    print("Clearing conversation history")
    # Reset conversation history
    st.session_state.conversation_history = []
    # Generate a new chat ID
    st.session_state.current_chat_id = str(uuid.uuid4())
    return st.session_state.current_chat_id

def get_conversation_history() -> List[Dict[str, Any]]:
    """
    Get the current conversation history
    
    Returns:
        The conversation history as a list of message dictionaries
    """
    return st.session_state.conversation_history

def get_current_chat_id() -> str:
    """
    Get the current chat ID
    
    Returns:
        The current chat ID as a string
    """
    return st.session_state.current_chat_id