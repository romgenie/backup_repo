import streamlit as st
from typing import Tuple, Dict, Any

def _on_input_change():
    """Callback for input changes"""
    # Check for enter key (without shift key for newlines)
    current_input = st.session_state.input_area
    if current_input.endswith('\n') and not st.session_state.get('shift_pressed', False):
        # Remove the trailing newline
        st.session_state.input_area = current_input.rstrip('\n')
        # Flag for submission
        st.session_state.submit_message = True

def render_input_area() -> Tuple[str, bool]:
    """
    Render the message input area and send button
    
    Returns:
        Tuple containing (user_input, send_button_clicked)
    """
    # Create a more chat-like input area
    st.write("") # Add some space
    st.divider() # Add a divider between conversation and input area
    
    # Track message submission
    if "submit_message" not in st.session_state:
        st.session_state.submit_message = False
    
    # Track processing state
    if "message_processing" not in st.session_state:
        st.session_state.message_processing = False
        
    # User input storage
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
        
    # Determine if button should be disabled
    disable_button = st.session_state.get("message_processing", False)
    
    # Use a form to better handle submission
    with st.form(key="chat_form", clear_on_submit=False):
        # Use a horizontal layout for input and buttons
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_area(
                "Type your message", 
                value=st.session_state.user_input, 
                placeholder="Ask a question... (Press Enter to send)", 
                key="input_area", 
                height=80,
                on_change=_on_input_change
            )
        
        with col2:
            # Fix empty label warning by using a non-visible label
            st.markdown("##") # Better spacing that doesn't trigger warning
            submit_button = st.form_submit_button(
                "Send", 
                type="primary", 
                disabled=disable_button
            )
    
    # Check if message should be submitted
    send_clicked = submit_button or st.session_state.submit_message
    
    # Reset submission flag
    if st.session_state.submit_message:
        st.session_state.submit_message = False
    
    return user_input, send_clicked