"""
Session control components for the chat interface.
"""

import streamlit as st

def render_action_buttons() -> bool:
    """
    Render action buttons like Clear Chat.
    
    Returns:
        Boolean indicating if Clear Chat was clicked
    """
    # Add some space with styled container
    st.markdown('<div class="mt-sm mb-sm"></div>', unsafe_allow_html=True)
    
    # Container for the action buttons
    st.markdown('<div class="flex p-sm">', unsafe_allow_html=True)
    
    # Create columns for buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    
    # Clear chat button
    with col1:
        clear_button = st.button("Clear Chat", key="clear_chat_button")
    
    # Add more action buttons in the future
    # e.g., Save Conversation, Copy to Clipboard
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return clear_button

def handle_clear_conversation(on_clear_chat=None):
    """
    Handle clearing the conversation history.
    
    Args:
        on_clear_chat: Optional callback when chat is cleared
    """
    from agent_management import clear_conversation
    
    # Clear conversation history and get new chat ID
    new_chat_id = clear_conversation()
    
    # Call any additional callback
    if on_clear_chat is not None:
        on_clear_chat()
    
    # Return the new chat ID
    return new_chat_id

def render_session_info(chat_id: str):
    """
    Render information about the current session.
    
    Args:
        chat_id: The current chat ID
    """
    with st.expander("Session Information", expanded=False):
        st.write(f"Chat ID: {chat_id}")
        
        # Display session age
        if "session_start_time" in st.session_state:
            import time
            from datetime import timedelta
            
            current_time = time.time()
            session_age = current_time - st.session_state.session_start_time
            human_readable_age = str(timedelta(seconds=int(session_age)))
            
            st.write(f"Session Age: {human_readable_age}")
        
        # Show message count
        if "conversation_history" in st.session_state:
            message_count = len(st.session_state.conversation_history)
            st.write(f"Messages: {message_count}")

def render_chat_settings():
    """Render chat-specific settings."""
    with st.expander("Chat Settings", expanded=False):
        # Max turns setting
        max_turns = st.slider(
            "Maximum Conversation Turns", 
            min_value=1, 
            max_value=20, 
            value=st.session_state.get("max_turns", 10),
            help="Maximum number of back-and-forth exchanges in a single conversation"
        )
        
        # Update session state if changed
        if "max_turns" not in st.session_state or st.session_state.max_turns != max_turns:
            st.session_state.max_turns = max_turns