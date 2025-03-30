"""
Demo Application for UI Improvements

This demo showcases all the UI enhancements including the model management system.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import UI components
from ui_improvements.integration import (
    integrate_model_management_css,
    show_model_info,
    get_model_info_for_agent,
    render_model_tab
)

from ui_improvements.model_selector import (
    render_compact_model_selector,
    render_visual_model_selector
)

def main():
    """Main demo application."""
    st.set_page_config(
        page_title="Agent Runner UI Improvements Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    integrate_model_management_css()
    
    # Sidebar
    st.sidebar.title("Agent Runner")
    st.sidebar.markdown("## Demo Navigation")
    
    # Mock user login
    user_name = st.sidebar.text_input("Username", value="demo_user")
    
    # Show model info in sidebar
    show_model_info(st.sidebar)
    
    # Tabs selection
    demo_section = st.sidebar.radio(
        "Select Demo Section",
        options=[
            "Introduction",
            "Model Management",
            "Model Selection",
            "Agent Configuration",
            "Integration Features"
        ]
    )
    
    # Header
    st.title("Agent Runner UI Improvements Demo")
    st.markdown(f"Welcome, {user_name}! This demo showcases the new model management system.")
    
    # Introduction Section
    if demo_section == "Introduction":
        st.header("Introduction")
        st.markdown("""
        ## UI Improvements Overview
        
        This demo showcases various UI improvements for the Agent Runner application,
        with a focus on the new Model Management System.
        
        ### Key Features:
        
        1. **Model Management**: Configure and manage LLM providers and models
        2. **Provider Selection**: Visual cards for selecting AI providers
        3. **Model Selection**: Easy selection of models with detailed information
        4. **Integration**: Integration points with existing app components
        5. **Responsive Design**: Improved layout for all screen sizes
        
        Use the sidebar navigation to explore different parts of the demo.
        """)
        
        # Show a teaser of the model management UI
        st.subheader("Current Model")
        model_info = get_model_info_for_agent()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Provider", model_info["provider_display"])
        with col2:
            st.metric("Model", model_info["model_display"])
            
        st.info(f"**Context Size**: {model_info['context_tokens']:,} tokens\n\n{model_info['description']}")
        
        st.markdown("**Try it out:** Navigate to the Model Management section to configure models.")
    
    # Model Management Section
    elif demo_section == "Model Management":
        render_model_tab()
    
    # Model Selection Section
    elif demo_section == "Model Selection":
        st.header("Model Selection Components")
        st.markdown("""
        These components can be integrated into various parts of the application
        to provide model selection functionality.
        """)
        
        tab1, tab2 = st.tabs(["Compact Selector", "Visual Selector"])
        
        with tab1:
            st.subheader("Compact Model Selector")
            st.markdown("""
            This lightweight selector is designed to be embedded within other UI components,
            such as the agent configuration panel.
            """)
            
            # Define a callback
            def on_model_change(provider, model, settings):
                st.success(f"Model changed to: {provider} - {model}")
            
            # Render the compact selector
            provider, model, settings = render_compact_model_selector(
                on_model_change=on_model_change,
                show_settings=True
            )
            
            # Show the current selection
            st.divider()
            st.markdown("### Selected Model")
            st.json({
                "provider": provider,
                "model": model,
                "settings": settings.to_dict() if settings else {}
            })
            
            st.markdown("""
            **Integration usage:**
            ```python
            from ui_improvements.model_selector import render_compact_model_selector
            
            # In your UI code:
            provider, model, settings = render_compact_model_selector(
                on_model_change=my_callback_function,
                show_settings=True
            )
            
            # Use the selected model for agent creation
            agent = create_agent(
                model=model,
                model_settings=settings,
                # other agent parameters...
            )
            ```
            """)
        
        with tab2:
            st.subheader("Visual Model Selector")
            st.markdown("""
            This more visual selector uses cards to display providers and models,
            making it ideal for dedicated model selection pages.
            """)
            
            # Render the visual selector
            provider, model, settings = render_visual_model_selector()
            
            # Show the current selection
            st.divider()
            st.markdown("### Selected Model")
            st.json({
                "provider": provider,
                "model": model,
                "settings": settings.to_dict() if settings else {}
            })
            
            st.markdown("""
            **Integration usage:**
            ```python
            from ui_improvements.model_selector import render_visual_model_selector
            
            # In your UI code:
            provider, model, settings = render_visual_model_selector()
            
            # Use the selected model
            st.write(f"Selected {provider} model: {model}")
            ```
            """)
    
    # Agent Configuration Section
    elif demo_section == "Agent Configuration":
        st.header("Agent Configuration with Model Selection")
        
        # Simulate agent configuration tab
        st.subheader("Create a new agent")
        
        # Mock agent configuration form
        with st.form(key="agent_config_form"):
            agent_name = st.text_input("Agent Name", value="My Agent")
            agent_instructions = st.text_area("Instructions", value="You are a helpful assistant.")
            
            # Model selection using the compact selector
            st.markdown("### Select Model")
            provider, model, settings = render_compact_model_selector(show_settings=True)
            
            # Tool selection (mock)
            st.markdown("### Select Tools")
            tool_options = ["Web Search", "Calculator", "File Browser", "Database Query"]
            selected_tools = st.multiselect("Tools", options=tool_options)
            
            # Submit button
            submit = st.form_submit_button("Create Agent")
            
            if submit:
                # Show loading state
                with st.spinner("Creating agent..."):
                    # Simulate API call
                    time.sleep(1.5)
                
                # Show success message with model info
                st.success(f"Agent '{agent_name}' created successfully with {provider} model: {model}")
                
                # Show the agent configuration
                st.markdown("### Agent Configuration")
                st.json({
                    "name": agent_name,
                    "instructions": agent_instructions,
                    "provider": provider,
                    "model": model,
                    "model_settings": settings.to_dict() if settings else {},
                    "tools": selected_tools
                })
    
    # Integration Features Section 
    elif demo_section == "Integration Features":
        st.header("Integration Features")
        st.markdown("""
        This section demonstrates how the model management system can be integrated
        with the existing Agent Runner application.
        """)
        
        integration_tabs = st.tabs([
            "Agent Creation Integration",
            "Configuration Tab Integration",
            "Session Info Integration",
            "API Integration"
        ])
        
        with integration_tabs[0]:
            st.subheader("Agent Creation Integration")
            st.markdown("""
            The model management system can be integrated with the agent creation workflow
            to automatically apply the selected model and its settings.
            
            ```python
            from ui_improvements.integration import integrate_with_agent_creation
            
            # Original agent creation function
            def create_agent(name, instructions, tools):
                # Create an agent...
                return agent
            
            # Enhanced function with model selection
            enhanced_create_agent = integrate_with_agent_creation(create_agent)
            
            # Use the enhanced function
            agent = enhanced_create_agent(name="My Agent", instructions="...", tools=[...])
            ```
            
            This automatically enhances the agent creation function to include model selection,
            and applies the selected model when creating the agent.
            """)
            
            st.info("This integration requires no changes to the core agent creation logic.")
        
        with integration_tabs[1]:
            st.subheader("Configuration Tab Integration")
            st.markdown("""
            The model management system can be integrated with the existing configuration tab
            to add a new model selection section.
            
            ```python
            from ui_improvements.integration import add_to_configure_tab
            
            # Original tab rendering function
            def render_configure_tab():
                # Render the configuration tab...
                return [tab1, tab2, tab3]
            
            # Enhanced function with model tab
            enhanced_render_tab = add_to_configure_tab(render_configure_tab)
            
            # Use the enhanced function
            tabs = enhanced_render_tab()
            ```
            
            This adds a new Model Selection tab to the existing configuration tabs.
            """)
            
            # Simulate the result with a mock UI
            st.write("Example Result:")
            mock_tabs = st.tabs(["Agent", "Tools", "API Keys", "Model Selection"])
            
            with mock_tabs[3]:
                st.subheader("AI Model Selection")
                st.write("Select the AI model to use for your agents")
                
                # Show a simplified version of the selector
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.selectbox("Provider", ["OpenAI", "Anthropic", "Cohere"])
                with col2:
                    st.selectbox("Model", ["GPT-4", "GPT-3.5 Turbo", "Claude 3 Opus"])
        
        with integration_tabs[2]:
            st.subheader("Session Info Integration")
            st.markdown("""
            The model management system can be integrated with the session information display
            to show the current model being used.
            
            ```python
            from ui_improvements.integration import show_model_info
            
            # In your UI code
            with st.sidebar:
                show_model_info()
                
            # Or in a specific container
            with st.expander("Session Information"):
                show_model_info(st)
            ```
            
            This displays the current model information in the specified container.
            """)
            
            # Show example
            st.write("Example Result:")
            with st.expander("Session Information", expanded=True):
                show_model_info(st)
        
        with integration_tabs[3]:
            st.subheader("API Integration")
            st.markdown("""
            The model management system can be integrated with the agent runner API
            to automatically apply the selected model when creating agents.
            
            ```python
            from ui_improvements.integration import (
                integrate_with_agent_runner,
                apply_model_to_agent_config
            )
            
            # Set up integration
            integrate_with_agent_runner("/path/to/agent_runner.py")
            
            # Apply model to an agent config
            agent_config = {
                "name": "My Agent",
                "instructions": "...",
                "tools": [...]
            }
            
            # This adds model and model_settings to the config
            updated_config = apply_model_to_agent_config(agent_config)
            ```
            
            This automatically applies the selected model to agent configurations.
            """)
            
            # Show example configuration
            st.write("Example Result:")
            
            sample_config = {
                "name": "My Agent",
                "instructions": "You are a helpful assistant.",
                "tools": ["Web Search", "Calculator"]
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original Config:")
                st.json(sample_config)
            
            with col2:
                st.write("Updated Config:")
                # Get the current model info
                provider, model, settings = render_compact_model_selector(show_settings=False)
                
                # Apply to the sample config
                sample_config["model"] = model
                sample_config["model_settings"] = settings.to_dict() if settings else {}
                
                st.json(sample_config)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🤖 Agent Runner UI Improvements Demo | "
        "Created with [Streamlit](https://streamlit.io) | "
        "Version 1.0.0"
    )

if __name__ == "__main__":
    main()