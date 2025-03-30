import streamlit as st
from typing import Tuple

def render_run_settings() -> Tuple[int, bool]:
    """
    Render the run settings section
    
    Returns:
        Tuple containing (max_turns, enable_tracing)
    """
    with st.expander("Run Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            max_turns = st.number_input(
                "Max Turns", 
                1, 20, 10, 1,
                help="Maximum number of interaction turns"
            )
            
        with col2:
            enable_tracing = st.checkbox(
                "Enable Tracing", 
                True,
                help="Record detailed traces of agent execution"
            )
    
    return max_turns, enable_tracing