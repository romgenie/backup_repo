import streamlit as st
import json

def model_settings_module():
    st.header("Advanced Model Settings")
    
    # Initialize session state
    if 'response_format' not in st.session_state:
        st.session_state.response_format = "auto"
    
    if 'custom_response_format' not in st.session_state:
        st.session_state.custom_response_format = "{}"
    
    if 'max_retries' not in st.session_state:
        st.session_state.max_retries = 0
    
    if 'user_identifier' not in st.session_state:
        st.session_state.user_identifier = ""
    
    if 'seed' not in st.session_state:
        st.session_state.seed = None
    
    if 'timeout' not in st.session_state:
        st.session_state.timeout = 300.0  # 5 minutes in seconds
    
    if 'parallel_tool_calls' not in st.session_state:
        st.session_state.parallel_tool_calls = True
        
    if 'truncation' not in st.session_state:
        st.session_state.truncation = False
        
    if 'store' not in st.session_state:
        st.session_state.store = None
    
    # Create form
    with st.form(key="model_settings_form"):
        # Response Format Settings
        st.subheader("Response Format")
        st.markdown("""
        Configure how the model formats its responses. This can improve performance when you 
        need specific formats like JSON.
        """)
        
        response_format = st.radio(
            "Response Format Type",
            options=["auto", "json", "custom"],
            index=["auto", "json", "custom"].index(st.session_state.response_format),
            help="Format preference for model responses"
        )
        
        if response_format == "custom":
            custom_response_format = st.text_area(
                "Custom Response Format (JSON)",
                value=st.session_state.custom_response_format,
                height=100,
                help="Custom JSON schema for the response format"
            )
            
            # Validate JSON
            if custom_response_format:
                try:
                    json.loads(custom_response_format)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please check the syntax.")
        else:
            custom_response_format = "{}"
        
        # Reliability Settings
        st.subheader("Reliability Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_retries = st.number_input(
                "Max Retries",
                min_value=0,
                max_value=10,
                value=st.session_state.max_retries,
                help="Number of times to retry failed API calls"
            )
            
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=10.0,
                max_value=600.0,
                value=st.session_state.timeout,
                step=10.0,
                help="Maximum time to wait for a model response"
            )
        
        with col2:
            # Use None for default random seed behavior
            use_seed = st.checkbox(
                "Use Deterministic Seed",
                value=st.session_state.seed is not None,
                help="Enable for reproducible model outputs"
            )
            
            if use_seed:
                seed = st.number_input(
                    "Seed Value",
                    min_value=0,
                    max_value=2147483647,  # Max 32-bit integer
                    value=st.session_state.seed if st.session_state.seed is not None else 42,
                    help="Seed value for deterministic outputs"
                )
            else:
                seed = None
            
            parallel_tool_calls = st.checkbox(
                "Enable Parallel Tool Calls",
                value=st.session_state.parallel_tool_calls,
                help="Allow multiple tool calls to execute in parallel"
            )
            
            truncation = st.checkbox(
                "Enable Message Truncation",
                value=st.session_state.truncation,
                help="Truncate message history if needed to fit within context window"
            )
            
            store_options = ["None", "OpenAI", "Custom"]
            store_index = 0 if st.session_state.store is None else (1 if st.session_state.store == "openai" else 2)
            store = st.selectbox(
                "Message Store",
                options=store_options,
                index=store_index,
                help="Storage for message history (None, OpenAI, or Custom store)"
            )
        
        # User Identification
        st.subheader("User Identification")
        
        user_identifier = st.text_input(
            "User Identifier",
            value=st.session_state.user_identifier,
            placeholder="user-123",
            help="Optional user ID for tracking model interactions"
        )
        
        # Submit button
        submitted = st.form_submit_button("Save Model Settings")
        
        if submitted:
            # Save Response Format
            st.session_state.response_format = response_format
            if response_format == "custom":
                st.session_state.custom_response_format = custom_response_format
            
            # Save Reliability Settings
            st.session_state.max_retries = max_retries
            st.session_state.timeout = timeout
            st.session_state.seed = seed
            st.session_state.parallel_tool_calls = parallel_tool_calls
            st.session_state.truncation = truncation
            
            # Convert store selection to appropriate value
            if store == "None":
                st.session_state.store = None
            elif store == "OpenAI":
                st.session_state.store = "openai"
            else:
                st.session_state.store = "custom"  # For demonstration; would need actual custom store implementation
            
            # Save User Identification
            st.session_state.user_identifier = user_identifier
            
            st.success("Model settings saved!")
    
    # Show generated code
    st.subheader("Model Settings Code")
    
    code_lines = ["# Advanced Model Settings"]
    
    # Response format settings
    if st.session_state.response_format != "auto":
        code_lines.append("\n# Configure response format")
        
        if st.session_state.response_format == "json":
            code_lines.append("response_format = {\"type\": \"json_object\"}")
        elif st.session_state.response_format == "custom":
            try:
                custom_format = json.loads(st.session_state.custom_response_format)
                code_lines.append(f"response_format = {json.dumps(custom_format, indent=2)}")
            except json.JSONDecodeError:
                code_lines.append("# Invalid JSON format specified")
                code_lines.append("response_format = None")
    else:
        code_lines.append("# Using default auto response format")
        code_lines.append("response_format = None")
    
    # Reliability settings
    reliability_options = []
    
    if st.session_state.max_retries > 0:
        reliability_options.append(f"max_retries={st.session_state.max_retries}")
    
    if st.session_state.timeout != 300.0:  # If not the default
        reliability_options.append(f"timeout={st.session_state.timeout}")
    
    if st.session_state.seed is not None:
        reliability_options.append(f"seed={st.session_state.seed}")
    
    if not st.session_state.parallel_tool_calls:  # Only if disabled (default is True)
        reliability_options.append("parallel_tool_calls=False")
        
    if st.session_state.truncation:  # Only if enabled (default is False)
        reliability_options.append("truncation=True")
        
    if st.session_state.store is not None:
        if st.session_state.store == "openai":
            reliability_options.append('store="openai"')
        else:
            reliability_options.append('# Custom store implementation needed')
            reliability_options.append('# store=CustomStore()')
    
    if reliability_options:
        code_lines.append("\n# Configure reliability settings")
        code_lines.append("model_settings = {")
        for option in reliability_options:
            code_lines.append(f"    {option},")
        code_lines.append("}")
    
    # User identification
    if st.session_state.user_identifier:
        code_lines.append("\n# Set user identifier")
        code_lines.append(f"user_id = \"{st.session_state.user_identifier}\"")
    
    # Example implementation
    code_lines.append("\n# Example: Apply settings when creating an agent")
    code_lines.append("from agents import Agent, Runner")
    code_lines.append("")
    code_lines.append("# Create agent with these settings")
    
    agent_params = []
    agent_params.append("name=\"My Agent\"")
    agent_params.append("instructions=\"Help the user with their questions\"")
    
    if st.session_state.response_format != "auto":
        agent_params.append("response_format=response_format")
    
    if reliability_options:
        agent_params.append("**model_settings")
    
    if st.session_state.user_identifier:
        agent_params.append("user=user_id")
    
    agent_creation = "agent = Agent(\n"
    for param in agent_params:
        agent_creation += f"    {param},\n"
    agent_creation += ")"
    
    code_lines.append(agent_creation)
    
    full_code = "\n".join(code_lines)
    
    st.code(full_code, language="python")
    
    # Return the current configuration
    return {
        "response_format": {
            "type": st.session_state.response_format,
            "custom": json.loads(st.session_state.custom_response_format) if st.session_state.response_format == "custom" else None
        },
        "reliability": {
            "max_retries": st.session_state.max_retries,
            "timeout": st.session_state.timeout,
            "seed": st.session_state.seed,
            "parallel_tool_calls": st.session_state.parallel_tool_calls,
            "truncation": st.session_state.truncation,
            "store": st.session_state.store
        },
        "user_identifier": st.session_state.user_identifier or None
    }

# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Advanced Model Settings", layout="wide")
    model_settings = model_settings_module()
    
    # Show the current configuration
    with st.expander("Current Configuration"):
        st.json(model_settings)