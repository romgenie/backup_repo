"""
Components for rendering the conversation history in the chat interface.
"""

import streamlit as st
from typing import List, Dict, Any

def display_chat_history(container, conversation_history: List[Dict[str, Any]]):
    """
    Display the conversation history in a chat-like interface.
    
    Args:
        container: The Streamlit container to render in
        conversation_history: List of conversation message dictionaries
    """
    from ui_components import display_chat
    
    # Use the existing display_chat utility for rendering
    display_chat(container, conversation_history)
    
def format_message_content(content: str) -> str:
    """
    Format message content for display.
    
    Args:
        content: The message content to format
        
    Returns:
        Formatted message content
    """
    # Handle code blocks
    if "```" in content:
        # Simple formatting for code blocks
        parts = content.split("```")
        formatted_parts = []
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text
                formatted_parts.append(part)
            else:  # Code block
                # Check if language is specified
                if part.strip() and "\n" in part:
                    lang_line, code = part.split("\n", 1)
                    lang = lang_line.strip()
                    formatted_parts.append(f"```{lang}\n{code}\n```")
                else:
                    # No language specified or single line
                    formatted_parts.append(f"```\n{part}\n```")
        
        return "".join(formatted_parts)
    
    return content

def render_message(message: Dict[str, Any], container) -> None:
    """
    Render a single message in the conversation.
    
    Args:
        message: The message to render
        container: The Streamlit container to render in
    """
    role = message.get("role", "unknown")
    content = message.get("content", "")
    
    # Format content
    formatted_content = format_message_content(content)
    
    # Render according to role
    with container:
        # Use Streamlit's chat message component
        with st.chat_message(role):
            st.markdown(formatted_content)

def auto_scroll_chat():
    """
    Add JavaScript to auto-scroll the chat container to the bottom.
    """
    st.markdown("""
    <script>
        var chatContainer = document.querySelector('[data-testid="stVerticalBlock"]');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
    """, unsafe_allow_html=True)