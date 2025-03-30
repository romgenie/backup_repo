import streamlit as st
from typing import Tuple

def render_model_settings() -> Tuple[str, float, int]:
    """
    Render the model settings section
    
    Returns:
        Tuple containing (model_name, temperature, max_tokens)
    """
    with st.expander("Model Settings", expanded=True):
        model_name = st.selectbox(
            "Model", 
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"],
            help="Select the AI model to use for this agent"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature", 
                0.0, 1.0, 0.7, 0.1,
                help="Higher values make output more random, lower values more deterministic"
            )
            
        with col2:
            max_tokens = st.number_input(
                "Max Tokens", 
                100, 4000, 1000, 100,
                help="Maximum number of tokens in the response"
            )
    
    return model_name, temperature, max_tokens