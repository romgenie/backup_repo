"""
User input components for the chat interface.
"""

import streamlit as st
from typing import Tuple

def render_input_area() -> Tuple[str, bool]:
    """
    Render the message input area and send button.
    
    Returns:
        Tuple containing (user_input, send_button_clicked)
    """
    # Create a more chat-like input area
    st.markdown('<div class="mt-md mb-md"></div>', unsafe_allow_html=True)
    st.divider()  # Add a divider between conversation and input area
    
    # User input with key for clearing
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    # Container for our styled input area
    st.markdown('<div class="form-group p-sm">', unsafe_allow_html=True)
    
    # Use a horizontal layout for input and buttons
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_area(
            "Type your message", 
            value=st.session_state.user_input, 
            placeholder="Ask a question like 'what time is it?' to see tool usage...", 
            key="input_area", 
            height=80
        )
    
    with col2:
        st.markdown('<div class="mt-md"></div>', unsafe_allow_html=True)  # Add space to align buttons
        send_button = st.button("Send", type="primary", key="send_button")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return user_input, send_button

def validate_user_input(user_input: str) -> Tuple[bool, str]:
    """
    Validate user input for potential issues.
    
    Args:
        user_input: The user input to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for empty input
    if len(user_input.strip()) == 0:
        return False, "Please enter a valid message."
    
    # Check for excessively long inputs
    if len(user_input) > 10000:  # Set a reasonable limit
        return False, "Your message is too long. Please keep messages under 10,000 characters."
    
    # Check for potential prompt injection patterns
    suspicious_patterns = ["<s>", "</s>", "system:", "assistant:", "TOOL_START_TAG", "you are now", "ignore previous instructions"]
    for pattern in suspicious_patterns:
        if pattern.lower() in user_input.lower():
            print(f"WARNING: Potentially suspicious input pattern detected: {pattern}")
            # Continue processing but log the warning
    
    return True, ""

def clear_input_field():
    """Clear the input field."""
    st.session_state.user_input = ""