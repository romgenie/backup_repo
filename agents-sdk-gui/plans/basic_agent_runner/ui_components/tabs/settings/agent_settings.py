"""
Agent settings components for settings tab.
"""

import streamlit as st
from typing import Optional, Callable, Dict, Any

def render_api_key_settings(
    api_key: str,
    env_file_path: str,
    on_api_key_save: Callable[[str], str],
    on_api_key_load: Callable[[str], tuple]
) -> None:
    """
    Render API key configuration settings
    
    Args:
        api_key: Current API key
        env_file_path: Path to the .env file
        on_api_key_save: Callback to save API key
        on_api_key_load: Callback to load API key
    """
    st.markdown("""
    <div class="form-section">
        <h3>API Key Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="form-group">', unsafe_allow_html=True)
    api_key_input = st.text_input("OpenAI API Key", value=api_key, type="password")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    if col1.button("Save to .env"):
        if api_key_input:
            saved_path = on_api_key_save(api_key_input)
            st.markdown(f"""
            <div class="success-message">
                API key saved to {saved_path}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="error-message">
                API key cannot be empty
            </div>
            """, unsafe_allow_html=True)
    
    # Add a load from .env file button
    if col2.button("Load from .env"):
        try:
            loaded_key, loaded_path = on_api_key_load(env_file_path)
            if loaded_key:
                st.markdown(f"""
                <div class="success-message">
                    API key loaded from {loaded_path}
                </div>
                """, unsafe_allow_html=True)
                # Force a rerun to update the input fields
                st.session_state.api_key = loaded_key
                st.rerun()
            else:
                st.markdown("""
                <div class="warning-message">
                    No API key found in .env file
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class="error-message">
                Error loading API key: {str(e)}
            </div>
            """, unsafe_allow_html=True)
    
    # Add an explain button to show information about API keys
    if col3.button("How to get a key"):
        st.markdown("""
        <div class="info-message">
            <h4>Getting an OpenAI API Key</h4>
            
            1. Go to <a href="https://platform.openai.com/account/api-keys" target="_blank">OpenAI API Keys</a>
            2. Sign in or create an account
            3. Click "Create new secret key"
            4. Copy the generated key and paste it here
            5. Click "Save to .env" to save it for future use
            
            <strong>Note:</strong> API usage is billed to your OpenAI account. Check their pricing for details.
        </div>
        """, unsafe_allow_html=True)
    
    # Show current status
    st.markdown(f"""
    <div class="neutral-message">
        Current .env path: <code>{env_file_path}</code><br>
        API key status: <strong>{'Configured' if api_key else 'Not set'}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Add model selection settings
    render_model_settings()

def render_model_settings() -> None:
    """Render model selection settings"""
    # Model settings
    st.markdown("""
    <div class="form-section mt-md">
        <h3>Model Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize state for model settings if not present
    if "model_settings" not in st.session_state:
        st.session_state.model_settings = {
            "default_model": "gpt-4",
            "enable_all_models": False,
            "favorite_models": ["gpt-4", "gpt-3.5-turbo"]
        }
    
    # Default model dropdown
    st.markdown('<div class="form-group">', unsafe_allow_html=True)
    available_models = ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    
    # Add Claude models if option is enabled
    if st.session_state.model_settings.get("enable_all_models", False):
        available_models += ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
    
    default_model = st.selectbox(
        "Default Model",
        options=available_models,
        index=available_models.index(st.session_state.model_settings["default_model"])
        if st.session_state.model_settings["default_model"] in available_models
        else 0
    )
    st.session_state.model_settings["default_model"] = default_model
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Options for model settings
    show_all_models = st.checkbox(
        "Show all model options (including Claude models)", 
        value=st.session_state.model_settings.get("enable_all_models", False),
        key="show_all_models"
    )
    st.session_state.model_settings["enable_all_models"] = show_all_models
    
    # Information about the selected model
    model_info = {
        "gpt-4-turbo": "Latest GPT-4 model with improved capabilities and context window",
        "gpt-4": "Powerful model for complex tasks and reasoning",
        "gpt-3.5-turbo": "Fast and cost-effective for most tasks",
        "claude-3-opus": "Claude's most powerful model for complex tasks",
        "claude-3-sonnet": "Claude's balanced model for most use cases",
        "claude-3-haiku": "Claude's fastest and most efficient model"
    }
    
    if default_model in model_info:
        st.markdown(f"""
        <div class="info-message">
            <strong>{default_model}</strong>: {model_info[default_model]}
        </div>
        """, unsafe_allow_html=True)
    
    # Experimental features section
    with st.expander("Experimental Features", expanded=False):
        # Initialize experimental settings if not present
        if "experimental_settings" not in st.session_state:
            st.session_state.experimental_settings = {
                "use_streaming": True,
                "use_parallel_tool_calls": False,
                "structured_output": False
            }
            
        # Streaming toggle
        st.session_state.experimental_settings["use_streaming"] = st.checkbox(
            "Enable streaming responses",
            value=st.session_state.experimental_settings.get("use_streaming", True),
            help="Display agent responses as they are generated"
        )
        
        # Parallel tool calls toggle
        st.session_state.experimental_settings["use_parallel_tool_calls"] = st.checkbox(
            "Enable parallel tool calls",
            value=st.session_state.experimental_settings.get("use_parallel_tool_calls", False),
            help="Allow the agent to call multiple tools in parallel (experimental)"
        )
        
        # Structured output toggle
        st.session_state.experimental_settings["structured_output"] = st.checkbox(
            "Enable structured output",
            value=st.session_state.experimental_settings.get("structured_output", False),
            help="Try to enforce structured output format from the agent (experimental)"
        )