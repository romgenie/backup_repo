import streamlit as st
import json

def basic_config_module():
    st.header("Basic Configuration")
    
    # Load existing values from session state if available
    if 'agent_name' not in st.session_state:
        st.session_state.agent_name = "My Agent"
    if 'instructions' not in st.session_state:
        st.session_state.instructions = "Help the user with their questions"
    if 'model' not in st.session_state:
        st.session_state.model = "o3-mini"
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 1.0
    if 'frequency_penalty' not in st.session_state:
        st.session_state.frequency_penalty = 0.0
    if 'presence_penalty' not in st.session_state:
        st.session_state.presence_penalty = 0.0
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 4096
    if 'max_turns' not in st.session_state:
        st.session_state.max_turns = 10
    if 'tool_choice' not in st.session_state:
        st.session_state.tool_choice = "auto"
    if 'tool_use_behavior' not in st.session_state:
        st.session_state.tool_use_behavior = "run_llm_again"
    if 'use_structured_output' not in st.session_state:
        st.session_state.use_structured_output = False
    if 'output_schema' not in st.session_state:
        st.session_state.output_schema = '{}'
    
    # Create form to batch the inputs
    with st.form(key="basic_config_form"):
        # Basic Agent Configuration
        st.subheader("Basic Agent Settings")
        agent_name = st.text_input(
            "Agent Name",
            value=st.session_state.agent_name
        )
        
        instructions = st.text_area(
            "Instructions",
            value=st.session_state.instructions,
            height=100
        )
        
        # Get available models from the models module
        available_models = list(st.session_state.models.keys()) if 'models' in st.session_state else [
            # Default models if models module not initialized yet
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
        ]
        
        # Get default model if available
        default_model = st.session_state.default_model if 'default_model' in st.session_state else "gpt-4o-mini"
        
        # Determine index for current model
        if st.session_state.model in available_models:
            current_index = available_models.index(st.session_state.model)
        else:
            current_index = available_models.index(default_model) if default_model in available_models else 0
            
        model = st.selectbox(
            "Model",
            options=available_models,
            index=current_index
        )
        
        # Show model capabilities if available
        if 'models' in st.session_state and model in st.session_state.models:
            model_info = st.session_state.models[model]
            st.caption(f"**Provider**: {model_info['provider']} | **Capabilities**: {', '.join(model_info['capabilities'])}")
            st.caption(f"**Description**: {model_info['description']}")
        
        # Advanced Model Settings
        st.subheader("Advanced Model Settings")
        with st.expander("Model Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    help="Controls randomness: lower values are more deterministic, higher values more creative"
                )
                
                top_p = st.slider(
                    "Top P",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.top_p,
                    step=0.01,
                    help="Controls diversity by limiting to top percentage of token probabilities"
                )
            
            with col2:
                frequency_penalty = st.slider(
                    "Frequency Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=st.session_state.frequency_penalty,
                    step=0.1,
                    help="Reduces repetition by penalizing tokens that have already appeared in the text"
                )
                
                presence_penalty = st.slider(
                    "Presence Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=st.session_state.presence_penalty,
                    step=0.1,
                    help="Encourages the model to use different tokens by penalizing tokens that have appeared before"
                )
            
            col3, col4 = st.columns(2)
            
            with col3:
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=1,
                    max_value=16384,
                    value=st.session_state.max_tokens,
                    help="Maximum number of tokens to generate"
                )
            
            with col4:
                max_turns = st.number_input(
                    "Max Turns",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.max_turns,
                    help="Maximum number of agent turns before stopping"
                )
        
        # Tool Usage Configuration
        st.subheader("Tool Usage Configuration")
        with st.expander("Tool Settings"):
            tool_choice = st.selectbox(
                "Tool Choice",
                options=["auto", "required", "none"],
                index=["auto", "required", "none"].index(st.session_state.tool_choice),
                help="Controls how the agent uses tools: auto (when needed), required (must use), none (never use)"
            )
            
            tool_use_behavior = st.selectbox(
                "Tool Use Behavior",
                options=["run_llm_again", "stop_on_first_tool"],
                index=0 if st.session_state.tool_use_behavior == "run_llm_again" or st.session_state.tool_use_behavior == "" else 1,
                help="Controls agent behavior after tool use: 'run_llm_again' (default, send results back to LLM), 'stop_on_first_tool' (stop after tool use)"
            )
        
        # Structured Output Configuration
        st.subheader("Output Configuration")
        with st.expander("Structured Output"):
            use_structured_output = st.checkbox(
                "Use Structured Output",
                value=st.session_state.use_structured_output,
                help="Enables the agent to return typed, structured responses"
            )
            
            if use_structured_output:
                output_schema = st.text_area(
                    "JSON Schema",
                    value=st.session_state.output_schema,
                    height=150,
                    help="JSON schema that defines the structure of the agent's response"
                )
                
                # Validate the JSON schema
                if output_schema:
                    try:
                        json.loads(output_schema)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON schema. Please check the format.")
            else:
                output_schema = '{}'
        
        # Submit button
        submitted = st.form_submit_button("Save Configuration")
        
        if submitted:
            # Save basic settings
            st.session_state.agent_name = agent_name
            st.session_state.instructions = instructions
            st.session_state.model = model
            
            # Save model parameters
            st.session_state.temperature = temperature
            st.session_state.top_p = top_p
            st.session_state.frequency_penalty = frequency_penalty
            st.session_state.presence_penalty = presence_penalty
            st.session_state.max_tokens = max_tokens
            st.session_state.max_turns = max_turns
            
            # Save tool settings
            st.session_state.tool_choice = tool_choice
            st.session_state.tool_use_behavior = tool_use_behavior
            
            # Save output settings
            st.session_state.use_structured_output = use_structured_output
            st.session_state.output_schema = output_schema
            
            st.success("Configuration saved!")
    
    # Return the current values
    return {
        "agent_name": st.session_state.agent_name,
        "instructions": st.session_state.instructions,
        "model": st.session_state.model,
        "model_settings": {
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p,
            "frequency_penalty": st.session_state.frequency_penalty,
            "presence_penalty": st.session_state.presence_penalty,
            "max_tokens": st.session_state.max_tokens,
            "max_turns": st.session_state.max_turns
        },
        "tool_settings": {
            "tool_choice": st.session_state.tool_choice,
            "tool_use_behavior": st.session_state.tool_use_behavior
        },
        "output_settings": {
            "use_structured_output": st.session_state.use_structured_output,
            "output_schema": st.session_state.output_schema if st.session_state.use_structured_output else None
        }
    }

# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Agent Basic Configuration", layout="wide")
    basic_config = basic_config_module()
    
    # Show the current configuration
    with st.expander("Current Configuration"):
        st.json(basic_config)