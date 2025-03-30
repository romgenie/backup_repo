"""
Minimal Integration Example for Model Management System

This example shows how to integrate the model management system
with the main Agent Runner application with minimal changes.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import existing app components (mocked for this example)
class MockApp:
    """Mock application class for demonstration purposes."""
    def __init__(self):
        self.agent = None
        self.tools = []
        self.instructions = ""
    
    def create_agent(self, name, instructions, tools, **kwargs):
        """Mock agent creation function."""
        self.agent = {
            "name": name,
            "instructions": instructions,
            "tools": tools,
            **kwargs
        }
        return self.agent

# Import model management components
from ui_improvements.integration import (
    integrate_model_management_css,
    integrate_with_agent_creation,
    show_model_info
)

from ui_improvements.model_selector import render_compact_model_selector

def main():
    """Main application."""
    st.set_page_config(
        page_title="Model Management Integration Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    # Apply CSS
    integrate_model_management_css()
    
    # Create mock app
    app = MockApp()
    
    # Enhance the agent creation function
    enhanced_create_agent = integrate_with_agent_creation(app.create_agent)
    
    # Setup UI
    st.title("Model Management Integration Demo")
    st.write("This demo shows how to integrate the model management system with minimal changes.")
    
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        st.write("Agent Runner Demo")
        
        # Show model info in sidebar
        show_model_info(st.sidebar)
    
    # Main content
    st.header("Create Agent")
    
    # Create agent form
    with st.form("create_agent_form"):
        agent_name = st.text_input("Agent Name", "My Assistant")
        agent_instructions = st.text_area(
            "Instructions", 
            "You are a helpful assistant that can answer questions about various topics."
        )
        
        # Tool selection
        available_tools = ["Search", "Weather", "Calculator", "Database"]
        selected_tools = st.multiselect("Tools", available_tools)
        
        # Add the model selection component
        st.subheader("Model Selection")
        provider, model, settings = render_compact_model_selector(show_settings=True)
        
        # Submit button
        submit = st.form_submit_button("Create Agent")
    
    # Handle form submission
    if submit:
        with st.spinner("Creating agent..."):
            # Create the agent using the enhanced function
            agent = enhanced_create_agent(
                name=agent_name,
                instructions=agent_instructions,
                tools=selected_tools,
                # The function will automatically add model-related parameters
            )
        
        st.success(f"Agent '{agent_name}' created successfully!")
        
        # Show the created agent
        st.subheader("Agent Configuration")
        st.json(agent)

# Bonus: Show how to integrate with the configure tab
def mock_configure_tab():
    """Mock configure tab for demonstration purposes."""
    tabs = st.tabs(["Agent", "Tools", "Settings"])
    
    with tabs[0]:
        st.header("Agent Configuration")
        st.write("Configure agent parameters here.")
    
    with tabs[1]:
        st.header("Tools Configuration")
        st.write("Configure tools parameters here.")
    
    with tabs[2]:
        st.header("Settings Configuration")
        st.write("Configure application settings here.")
    
    return tabs

def show_tab_integration():
    """Show how to integrate with tabs."""
    st.header("Tab Integration Example")
    
    # Add a model tab to the existing tabs
    from ui_improvements.integration import add_to_configure_tab
    
    # Enhance the tab renderer
    enhanced_tab_renderer = add_to_configure_tab(mock_configure_tab)
    
    # Render the enhanced tabs
    tabs = enhanced_tab_renderer()
    
    st.write("Note: A new 'Model Selection' tab was added to the existing tabs.")

if __name__ == "__main__":
    # Main demo
    main()
    
    st.divider()
    
    # Show tab integration example
    show_tab_integration()