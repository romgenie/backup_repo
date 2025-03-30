import streamlit as st
import json

def tracing_config_module():
    st.header("Tracing Configuration")
    
    # Initialize session state
    if 'tracing_disabled' not in st.session_state:
        st.session_state.tracing_disabled = False
    
    if 'tracing_api_key' not in st.session_state:
        st.session_state.tracing_api_key = ""
    
    if 'trace_metadata' not in st.session_state:
        st.session_state.trace_metadata = [
            {"key": "environment", "value": "development"}
        ]
    
    # Create form
    with st.form(key="tracing_config_form"):
        tracing_disabled = st.checkbox(
            "Disable Tracing",
            value=st.session_state.tracing_disabled
        )
        
        # Only show API key and metadata if tracing is enabled
        if not tracing_disabled:
            tracing_api_key = st.text_input(
                "Tracing API Key (optional)",
                value=st.session_state.tracing_api_key,
                type="password",
                placeholder="sk-..."
            )
            st.caption("Used specifically for tracing. If not provided, will use the default API key.")
            
            # Trace Metadata Configuration
            st.subheader("Trace Metadata")
            st.markdown("""
            Add custom metadata to your traces to improve organization and filtering.
            Metadata is added to all spans within a trace and can be used to filter traces in the UI.
            """)
            
            # Display current metadata in a table
            if st.session_state.trace_metadata:
                st.dataframe(
                    st.session_state.trace_metadata,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No trace metadata defined")
            
            # Metadata management using session state and form submission
            st.markdown("### Add New Metadata")
            
            # Add metadata key-value pairs within the form
            col1, col2 = st.columns(2)
            
            with col1:
                new_metadata_key = st.text_input("Key", key="new_meta_key", placeholder="environment")
            
            with col2:
                new_metadata_value = st.text_input("Value", key="new_meta_value", placeholder="production")
            
            # Will handle actual addition in the form submit handler
            
            # Show delete options within the form
            if st.session_state.trace_metadata:
                st.markdown("### Remove Metadata")
                metadata_to_delete = st.selectbox(
                    "Select metadata to remove",
                    options=[f"{meta['key']}: {meta['value']}" for meta in st.session_state.trace_metadata],
                    key="meta_to_delete"
                )
            
            # Define placeholder variables for form submit handler
            new_key = new_metadata_key if 'new_metadata_key' in locals() else ""
            new_value = new_metadata_value if 'new_metadata_value' in locals() else ""
            to_delete = metadata_to_delete if 'metadata_to_delete' in locals() else None
        else:
            tracing_api_key = ""
        
        # Add checkbox for metadata operations
        add_metadata = st.checkbox("Add new metadata entry", key="add_meta_checkbox", value=False)
        remove_metadata = st.checkbox("Remove selected metadata entry", key="remove_meta_checkbox", value=False)
        
        # Submit button
        submitted = st.form_submit_button("Save Tracing Configuration")
        
        if submitted:
            # Always update the basic settings
            st.session_state.tracing_disabled = tracing_disabled
            st.session_state.tracing_api_key = tracing_api_key
            
            # Handle metadata operations if tracing is enabled
            if not tracing_disabled:
                # Add new metadata if requested
                if add_metadata and new_key and new_value:
                    # Check if key already exists (update it)
                    key_exists = False
                    for i, meta in enumerate(st.session_state.trace_metadata):
                        if meta["key"] == new_key:
                            st.session_state.trace_metadata[i]["value"] = new_value
                            key_exists = True
                            break
                    
                    # Add new key if it doesn't exist
                    if not key_exists:
                        st.session_state.trace_metadata.append({
                            "key": new_key,
                            "value": new_value
                        })
                
                # Remove metadata if requested
                if remove_metadata and to_delete:
                    key_to_delete = to_delete.split(":")[0].strip()
                    st.session_state.trace_metadata = [
                        meta for meta in st.session_state.trace_metadata 
                        if meta["key"] != key_to_delete
                    ]
            
            st.success("Tracing configuration saved!")
    
    # Show generated code
    st.subheader("Tracing Configuration Code")
    
    imports = []
    code_lines = ["# Tracing Configuration"]
    
    if st.session_state.tracing_disabled:
        imports.append("set_tracing_disabled")
        code_lines.append("set_tracing_disabled(True)")
    else:
        if st.session_state.tracing_api_key:
            imports.append("set_tracing_export_api_key")
            code_lines.append(f'set_tracing_export_api_key("{st.session_state.tracing_api_key}")')
        
        if st.session_state.trace_metadata:
            imports.append("set_trace_attributes")
            
            # Format the metadata dictionary as Python code
            metadata_dict = {meta["key"]: meta["value"] for meta in st.session_state.trace_metadata}
            metadata_str = json.dumps(metadata_dict, indent=4).replace('"', "'")
            
            code_lines.append(f'set_trace_attributes({metadata_str})')
    
    imports_str = f"from agents import {', '.join(imports)}" if imports else ""
    code_str = "\n".join(code_lines) if len(code_lines) > 1 else "# No tracing configuration set"
    
    full_code = f"{imports_str}\n\n{code_str}" if imports_str else code_str
    
    st.code(full_code, language="python")
    
    # Return the current configuration
    return {
        "tracing_disabled": st.session_state.tracing_disabled,
        "tracing_api_key": st.session_state.tracing_api_key if not st.session_state.tracing_disabled else None,
        "trace_metadata": {meta["key"]: meta["value"] for meta in st.session_state.trace_metadata} if not st.session_state.tracing_disabled else None
    }

# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Tracing Configuration", layout="wide")
    tracing_config = tracing_config_module()
    
    # Show the current configuration
    with st.expander("Current Configuration"):
        st.json(tracing_config)