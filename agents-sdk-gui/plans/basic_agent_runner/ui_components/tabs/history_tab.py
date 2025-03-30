import streamlit as st

def render_history_tab() -> None:
    """
    Render the history tab of the application
    """
    st.markdown("""
    <div class="form-section">
        <h2>Conversation History</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Placeholder for now - will be implemented in future
    st.markdown("""
    <div class="info-message">
        This tab will contain conversation history management. Coming soon!
    </div>
    """, unsafe_allow_html=True)