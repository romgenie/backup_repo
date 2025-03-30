"""
Main entry point for running the enhanced SDK features showcase.
"""
import streamlit as st
from .showcase import showcase_ui

# Configure the page
st.set_page_config(
    page_title="OpenAI Agents SDK - Enhanced Features",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Run the showcase
showcase_ui()