import streamlit as st
from typing import List, Dict, Any

def render_message_content(content: str):
    """
    Renders message content with proper handling for code blocks
    
    Args:
        content: The message content to render
    """
    # Handle code blocks if present
    if "```" in content:
        # Split content by code blocks
        parts = content.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text
                if part.strip():
                    st.write(part.strip())
            else:  # Code block
                # Check if language is specified
                code_lines = part.split("\n", 1)
                if len(code_lines) > 1:
                    lang, code = code_lines[0].strip(), code_lines[1]
                    st.code(code, language=lang if lang else None)
                else:
                    st.code(part)
    else:
        # Regular text without code blocks
        st.write(content)

def display_chat(container, conversation_history: List[Dict[str, Any]]) -> int:
    """
    Displays a chat conversation in a Streamlit container
    
    Args:
        container: The Streamlit container to display in
        conversation_history: List of message dictionaries with role and content
        
    Returns:
        int: The number of messages displayed
    """
    # Counter to track if any messages were displayed
    message_count = 0
    
    with container:
        # Check if we have any messages to display
        if len(conversation_history) == 0:
            st.info("No messages yet. Send a message to start the conversation.")
        else:
            # Display each message in the conversation history
            for message in conversation_history:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "user":
                    # Use Streamlit's native chat_message with "user" avatar
                    with st.chat_message("user", avatar="👤"):
                        st.write(content)
                    # Increment message counter
                    message_count += 1
                        
                elif role == "assistant":
                    # Use Streamlit's native chat_message with "assistant" avatar
                    with st.chat_message("assistant", avatar="🤖"):
                        render_message_content(content)
                    # Increment message counter
                    message_count += 1
            
            # Show info message if no messages were displayed despite having entries in the history
            if message_count == 0 and len(conversation_history) > 0:
                st.warning("There are messages in history but none could be displayed. This may be a format issue.")
    
    return message_count