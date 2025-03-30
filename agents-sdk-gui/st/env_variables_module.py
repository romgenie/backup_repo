import streamlit as st
import pandas as pd
import json
import os

def env_variables_module():
    st.header("Environment Variables")
    
    # Initialize session state for environment variables
    if 'env_vars' not in st.session_state:
        st.session_state.env_vars = [
            {"key": "OPENAI_API_KEY", "value": "sk-XXXX", "type": "password"},
            {"key": "OPENAI_ORG_ID", "value": "org-123456", "type": "text"},
            {"key": "OPENAI_AGENTS_DONT_LOG_MODEL_DATA", "value": "0", "type": "text"},
            {"key": "OPENAI_AGENTS_DONT_LOG_TOOL_DATA", "value": "0", "type": "text"}
        ]
    
    if 'use_custom_client' not in st.session_state:
        st.session_state.use_custom_client = False
    
    if 'base_url' not in st.session_state:
        st.session_state.base_url = ""
    
    if 'use_mcp' not in st.session_state:
        st.session_state.use_mcp = False
    
    if 'mcp_server_type' not in st.session_state:
        st.session_state.mcp_server_type = "stdio"
    
    if 'mcp_server_url' not in st.session_state:
        st.session_state.mcp_server_url = ""
    
    if 'mcp_server_command' not in st.session_state:
        st.session_state.mcp_server_command = ""
    
    if 'mcp_enable_caching' not in st.session_state:
        st.session_state.mcp_enable_caching = True
    
    if 'env_presets' not in st.session_state:
        st.session_state.env_presets = {
            "Default": {
                "env_vars": [
                    {"key": "OPENAI_API_KEY", "value": "sk-XXXX", "type": "password"},
                    {"key": "OPENAI_ORG_ID", "value": "org-123456", "type": "text"},
                    {"key": "OPENAI_AGENTS_DONT_LOG_MODEL_DATA", "value": "0", "type": "text"},
                    {"key": "OPENAI_AGENTS_DONT_LOG_TOOL_DATA", "value": "0", "type": "text"}
                ],
                "use_custom_client": False,
                "base_url": ""
            }
        }
    
    if 'current_preset' not in st.session_state:
        st.session_state.current_preset = "Default"
    
    # Function to convert env vars to a dictionary
    def env_vars_to_dict():
        return {var["key"]: var["value"] for var in st.session_state.env_vars}
    
    # Environment Presets Management
    st.subheader("Environment Presets")
    
    # Load and Save Presets Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        preset_options = list(st.session_state.env_presets.keys())
        selected_preset = st.selectbox(
            "Select Preset",
            options=preset_options,
            index=preset_options.index(st.session_state.current_preset)
        )
    
    with col2:
        if st.button("Load Selected Preset"):
            if selected_preset in st.session_state.env_presets:
                preset_data = st.session_state.env_presets[selected_preset]
                st.session_state.env_vars = preset_data["env_vars"].copy()
                st.session_state.use_custom_client = preset_data["use_custom_client"]
                st.session_state.base_url = preset_data["base_url"]
                st.session_state.current_preset = selected_preset
                st.success(f"Loaded preset: {selected_preset}")
                st.rerun()
    
    with col3:
        new_preset_name = st.text_input("New Preset Name", placeholder="My Preset")
        if st.button("Save Current as Preset"):
            if new_preset_name:
                st.session_state.env_presets[new_preset_name] = {
                    "env_vars": st.session_state.env_vars.copy(),
                    "use_custom_client": st.session_state.use_custom_client,
                    "base_url": st.session_state.base_url
                }
                st.session_state.current_preset = new_preset_name
                st.success(f"Saved current configuration as preset: {new_preset_name}")
                st.rerun()
            else:
                st.error("Please enter a name for the preset")
    
    # Display current environment variables in a table
    st.subheader("Current Environment Variables")
    
    if st.session_state.env_vars:
        df = pd.DataFrame(st.session_state.env_vars)
        
        # Display values as masked for password types
        displayed_df = df.copy()
        for i, row in displayed_df.iterrows():
            if row['type'] == 'password':
                displayed_df.at[i, 'value'] = "••••••••••••"
        
        # Display the table
        st.dataframe(
            displayed_df[['key', 'value', 'type']], 
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No environment variables defined")
    
    # Form to add new variable
    with st.form("add_env_var_form"):
        st.subheader("Add New Variable")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            new_key = st.text_input("Key", placeholder="VARIABLE_NAME")
        
        with col2:
            var_type = st.selectbox("Type", options=["text", "password"])
            new_value = st.text_input(
                "Value", 
                type="password" if var_type == "password" else "default",
                placeholder="value"
            )
        
        with col3:
            st.write(" ")  # Spacer
            st.write(" ")  # Spacer
            add_button = st.form_submit_button("Add Variable")
        
        if add_button and new_key and new_value:
            # Add to session state
            st.session_state.env_vars.append({
                "key": new_key,
                "value": new_value,
                "type": var_type
            })
            st.success(f"Added environment variable: {new_key}")
            st.rerun()
    
    # Delete variable button - outside form to allow individual deletion
    if st.session_state.env_vars:
        st.subheader("Remove Variables")
        var_to_delete = st.selectbox(
            "Select variable to remove",
            options=[var["key"] for var in st.session_state.env_vars]
        )
        
        if st.button("Remove Selected Variable"):
            st.session_state.env_vars = [
                var for var in st.session_state.env_vars if var["key"] != var_to_delete
            ]
            st.success(f"Removed environment variable: {var_to_delete}")
            st.rerun()
    
    # Custom client configuration
    st.subheader("Client Configuration")
    
    use_custom_client = st.checkbox(
        "Use Custom OpenAI Client",
        value=st.session_state.use_custom_client
    )
    
    if use_custom_client:
        base_url = st.text_input(
            "Base URL (optional)",
            value=st.session_state.base_url,
            placeholder="https://your-custom-endpoint.com"
        )
    else:
        base_url = ""
    
    # Update client related session state
    st.session_state.use_custom_client = use_custom_client
    st.session_state.base_url = base_url
    
    # Keep MCP variables in the session state but not visible in UI
    mcp_server_type = st.session_state.mcp_server_type
    mcp_server_url = st.session_state.mcp_server_url
    mcp_server_command = st.session_state.mcp_server_command
    mcp_enable_caching = st.session_state.mcp_enable_caching
    use_mcp = False  # Disable MCP by default now that it's removed from UI
    
    # Update session state for MCP (hidden now)
    st.session_state.use_mcp = use_mcp
    
    # Preview the .env file
    st.subheader(".env File Preview")
    
    env_content = "\n".join([f"{var['key']}={var['value']}" for var in st.session_state.env_vars])
    
    if use_custom_client and base_url:
        env_content += f"\nOPENAI_BASE_URL={base_url}"
    
    st.code(env_content)
    
    # Export/Import Presets
    st.subheader("Export/Import Presets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export All Presets"):
            presets_json = json.dumps(st.session_state.env_presets, indent=2)
            st.download_button(
                label="Download Presets JSON",
                data=presets_json,
                file_name="env_presets.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader("Import Presets from JSON", type=["json"])
        if uploaded_file is not None:
            try:
                presets_data = json.loads(uploaded_file.getvalue())
                st.session_state.env_presets.update(presets_data)
                st.success(f"Imported {len(presets_data)} presets successfully")
                st.rerun()
            except Exception as e:
                st.error(f"Error importing presets: {str(e)}")
    
    # Return current configuration
    return {
        "env_vars": env_vars_to_dict(),
        "use_custom_client": use_custom_client,
        "base_url": base_url if use_custom_client else None,
        "current_preset": st.session_state.current_preset
    }

# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Environment Variables Configuration", layout="wide")
    env_config = env_variables_module()
    
    # Show the current configuration
    with st.expander("Current Configuration"):
        st.json(env_config)