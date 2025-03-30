import streamlit as st
import json
import pandas as pd
from typing import Dict, List, Set, Optional
import datetime

def models_module():
    st.header("Models Configuration")
    
    # Initialize session state for models
    if 'models' not in st.session_state:
        st.session_state.models = {
            # OpenAI models (currently available)
            "gpt-4o": {
                "provider": "OpenAI",
                "capabilities": ["text", "reasoning", "tools", "json_response", "images", "vision"],
                "max_tokens": 4096,
                "created": "2024-05-13",
                "owned_by": "openai",
                "description": "GPT-4o is a large multimodal model that can solve tasks involving both vision and language"
            },
            "gpt-4o-mini": {
                "provider": "OpenAI",
                "capabilities": ["text", "reasoning", "tools", "json_response", "images", "vision"],
                "max_tokens": 4096,
                "created": "2024-05-13",
                "owned_by": "openai",
                "description": "A smaller, faster version of GPT-4o"
            },
            "gpt-4-turbo": {
                "provider": "OpenAI",
                "capabilities": ["text", "reasoning", "tools", "json_response"],
                "max_tokens": 4096,
                "created": "2023-12-13", 
                "owned_by": "openai",
                "description": "GPT-4 Turbo with improved instruction following"
            },
            "gpt-4": {
                "provider": "OpenAI",
                "capabilities": ["text", "reasoning", "tools", "json_response"],
                "max_tokens": 8192,
                "created": "2023-06-13",
                "owned_by": "openai",
                "description": "Most capable GPT-4 model"
            },
            "gpt-3.5-turbo": {
                "provider": "OpenAI",
                "capabilities": ["text", "tools", "json_response"],
                "max_tokens": 4096,
                "created": "2023-04-28",
                "owned_by": "openai",
                "description": "Fast and cost-effective model for most tasks"
            },
            # Anthropic models
            "claude-3-opus-20240229": {
                "provider": "Anthropic",
                "capabilities": ["text", "reasoning", "tools", "json_response", "images", "vision"],
                "max_tokens": 4096,
                "created": "2024-02-29",
                "owned_by": "anthropic",
                "description": "Most capable Claude model for complex tasks"
            },
            "claude-3-sonnet-20240229": {
                "provider": "Anthropic",
                "capabilities": ["text", "reasoning", "tools", "json_response", "images", "vision"],
                "max_tokens": 4096,
                "created": "2024-02-29",
                "owned_by": "anthropic",
                "description": "Balanced Claude model for most use cases"
            },
            "claude-3-haiku-20240307": {
                "provider": "Anthropic",
                "capabilities": ["text", "reasoning", "tools", "json_response", "images", "vision"],
                "max_tokens": 4096,
                "created": "2024-03-07",
                "owned_by": "anthropic",
                "description": "Fast, efficient Claude model for simpler tasks"
            }
        }
    
    if 'model_providers' not in st.session_state:
        st.session_state.model_providers = ["OpenAI", "Anthropic", "Custom"]
    
    if 'model_capabilities' not in st.session_state:
        st.session_state.model_capabilities = [
            "text", "reasoning", "tools", "json_response", 
            "images", "vision", "audio", "code", "function_calling"
        ]
        
    if 'default_model' not in st.session_state:
        st.session_state.default_model = "gpt-4o-mini"
    
    # Display all models table
    st.subheader("Available Models")
    
    # Convert models to DataFrame for display
    models_data = []
    for model_name, model_info in st.session_state.models.items():
        capabilities_str = ", ".join(model_info["capabilities"])
        models_data.append({
            "Model": model_name,
            "Provider": model_info["provider"],
            "Owner": model_info["owned_by"],
            "Created": model_info["created"],
            "Capabilities": capabilities_str,
            "Max Tokens": model_info["max_tokens"],
            "Description": model_info["description"]
        })
    
    if models_data:
        models_df = pd.DataFrame(models_data)
        st.dataframe(models_df, use_container_width=True)
    else:
        st.info("No models configured yet")
    
    # Default model selection
    st.subheader("Default Model")
    default_model = st.selectbox(
        "Select default model for agents",
        options=list(st.session_state.models.keys()),
        index=list(st.session_state.models.keys()).index(st.session_state.default_model) 
            if st.session_state.default_model in st.session_state.models else 0
    )
    
    if st.button("Set as Default", key="set_default_model_button"):
        st.session_state.default_model = default_model
        st.success(f"Default model set to: {default_model}")
    
    # Add tabs for different operations
    add_tab, edit_tab, remove_tab, provider_tab, capability_tab = st.tabs([
        "Add Model", "Edit Model", "Remove Model", "Manage Providers", "Manage Capabilities"
    ])
    
    # Add New Model Tab
    with add_tab:
        st.subheader("Add New Model")
        
        with st.form("add_model_form"):
            new_model_name = st.text_input("Model Name", placeholder="model-name")
            
            cols = st.columns(2)
            with cols[0]:
                new_provider = st.selectbox(
                    "Provider", 
                    options=st.session_state.model_providers
                )
                
                new_owned_by = st.text_input(
                    "Owner", 
                    placeholder="openai, anthropic, or organization-id",
                    value=new_provider.lower() if new_provider in ["OpenAI", "Anthropic"] else ""
                )
            
            with cols[1]:
                new_max_tokens = st.number_input(
                    "Max Tokens", 
                    min_value=1, 
                    max_value=128000,
                    value=4096
                )
                
                new_created = st.date_input(
                    "Created Date",
                    value=datetime.datetime.now().date()
                )
            
            new_description = st.text_area(
                "Description", 
                placeholder="Brief description of the model's capabilities and use case"
            )
            
            st.write("Model Capabilities")
            capabilities_cols = st.columns(3)
            selected_capabilities = []
            
            for i, capability in enumerate(st.session_state.model_capabilities):
                col_idx = i % 3
                with capabilities_cols[col_idx]:
                    if st.checkbox(capability, key=f"add_cap_{capability}"):
                        selected_capabilities.append(capability)
            
            submitted = st.form_submit_button("Add Model")
            
            if submitted:
                if not new_model_name:
                    st.error("Model name is required")
                elif new_model_name in st.session_state.models:
                    st.error(f"Model '{new_model_name}' already exists")
                elif not selected_capabilities:
                    st.error("Please select at least one capability")
                elif not new_owned_by:
                    st.error("Owner is required")
                else:
                    # Add the new model
                    st.session_state.models[new_model_name] = {
                        "provider": new_provider,
                        "capabilities": selected_capabilities,
                        "max_tokens": new_max_tokens,
                        "created": new_created.strftime("%Y-%m-%d"),
                        "owned_by": new_owned_by,
                        "description": new_description
                    }
                    st.success(f"Added model: {new_model_name}")
                    st.rerun()
    
    # Edit Model Tab
    with edit_tab:
        st.subheader("Edit Existing Model")
        
        model_to_edit = st.selectbox(
            "Select Model to Edit",
            options=list(st.session_state.models.keys())
        )
        
        if model_to_edit:
            model_info = st.session_state.models[model_to_edit]
            
            with st.form("edit_model_form"):
                cols = st.columns(2)
                
                with cols[0]:
                    edit_provider = st.selectbox(
                        "Provider", 
                        options=st.session_state.model_providers,
                        index=st.session_state.model_providers.index(model_info["provider"])
                    )
                    
                    edit_owned_by = st.text_input(
                        "Owner", 
                        value=model_info["owned_by"]
                    )
                
                with cols[1]:
                    edit_max_tokens = st.number_input(
                        "Max Tokens", 
                        min_value=1, 
                        max_value=128000,
                        value=model_info["max_tokens"]
                    )
                    
                    edit_created = st.date_input(
                        "Created Date",
                        value=datetime.datetime.strptime(model_info["created"], "%Y-%m-%d").date()
                    )
                
                edit_description = st.text_area(
                    "Description", 
                    value=model_info["description"]
                )
                
                st.write("Model Capabilities")
                edit_capabilities_cols = st.columns(3)
                edit_selected_capabilities = []
                
                for i, capability in enumerate(st.session_state.model_capabilities):
                    col_idx = i % 3
                    with edit_capabilities_cols[col_idx]:
                        if st.checkbox(
                            capability, 
                            value=capability in model_info["capabilities"],
                            key=f"edit_cap_{capability}"
                        ):
                            edit_selected_capabilities.append(capability)
                
                update_button = st.form_submit_button("Update Model")
                
                if update_button:
                    if not edit_selected_capabilities:
                        st.error("Please select at least one capability")
                    elif not edit_owned_by:
                        st.error("Owner is required")
                    else:
                        # Update the model
                        st.session_state.models[model_to_edit] = {
                            "provider": edit_provider,
                            "capabilities": edit_selected_capabilities,
                            "max_tokens": edit_max_tokens,
                            "created": edit_created.strftime("%Y-%m-%d"),
                            "owned_by": edit_owned_by,
                            "description": edit_description
                        }
                        st.success(f"Updated model: {model_to_edit}")
                        st.rerun()
    
    # Remove Model Tab
    with remove_tab:
        st.subheader("Remove Model")
        
        model_to_remove = st.selectbox(
            "Select Model to Remove",
            options=list(st.session_state.models.keys()),
            key="remove_model_select"
        )
        
        if model_to_remove == st.session_state.default_model:
            st.warning(f"Warning: {model_to_remove} is currently set as the default model. Removing it will reset the default.")
        
        if st.button("Remove Selected Model", key="remove_model_button"):
            if model_to_remove:
                # Remove the model
                if model_to_remove in st.session_state.models:
                    del st.session_state.models[model_to_remove]
                    
                    # Reset default model if needed
                    if model_to_remove == st.session_state.default_model:
                        if st.session_state.models:
                            st.session_state.default_model = list(st.session_state.models.keys())[0]
                        else:
                            st.session_state.default_model = ""
                    
                    st.success(f"Removed model: {model_to_remove}")
                    st.rerun()
                else:
                    st.error(f"Model '{model_to_remove}' not found")
    
    # Manage Providers Tab
    with provider_tab:
        st.subheader("Manage Model Providers")
        
        # Display current providers
        st.write("Current Providers:")
        st.write(", ".join(st.session_state.model_providers))
        
        # Add new provider
        with st.form("add_provider_form"):
            new_provider = st.text_input("New Provider Name", placeholder="Provider Name")
            add_provider_button = st.form_submit_button("Add Provider")
            
            if add_provider_button:
                if not new_provider:
                    st.error("Provider name is required")
                elif new_provider in st.session_state.model_providers:
                    st.error(f"Provider '{new_provider}' already exists")
                else:
                    st.session_state.model_providers.append(new_provider)
                    st.success(f"Added provider: {new_provider}")
                    st.rerun()
        
        # Remove provider
        provider_to_remove = st.selectbox(
            "Select Provider to Remove",
            options=st.session_state.model_providers
        )
        
        if st.button("Remove Selected Provider", key="remove_provider_button"):
            if provider_to_remove:
                # Check if provider is in use
                in_use = False
                for model_info in st.session_state.models.values():
                    if model_info["provider"] == provider_to_remove:
                        in_use = True
                        break
                
                if in_use:
                    st.error(f"Cannot remove provider '{provider_to_remove}' as it is used by one or more models")
                else:
                    st.session_state.model_providers.remove(provider_to_remove)
                    st.success(f"Removed provider: {provider_to_remove}")
                    st.rerun()
    
    # Manage Capabilities Tab
    with capability_tab:
        st.subheader("Manage Model Capabilities")
        
        # Display current capabilities
        st.write("Current Capabilities:")
        st.write(", ".join(st.session_state.model_capabilities))
        
        # Add new capability
        with st.form("add_capability_form"):
            new_capability = st.text_input("New Capability Name", placeholder="Capability Name")
            add_capability_button = st.form_submit_button("Add Capability")
            
            if add_capability_button:
                if not new_capability:
                    st.error("Capability name is required")
                elif new_capability in st.session_state.model_capabilities:
                    st.error(f"Capability '{new_capability}' already exists")
                else:
                    st.session_state.model_capabilities.append(new_capability)
                    st.success(f"Added capability: {new_capability}")
                    st.rerun()
        
        # Remove capability
        capability_to_remove = st.selectbox(
            "Select Capability to Remove",
            options=st.session_state.model_capabilities
        )
        
        if st.button("Remove Selected Capability", key="remove_capability_button"):
            if capability_to_remove:
                # Check if capability is in use
                in_use = False
                for model_info in st.session_state.models.values():
                    if capability_to_remove in model_info["capabilities"]:
                        in_use = True
                        break
                
                if in_use:
                    st.error(f"Cannot remove capability '{capability_to_remove}' as it is used by one or more models")
                else:
                    st.session_state.model_capabilities.remove(capability_to_remove)
                    st.success(f"Removed capability: {capability_to_remove}")
                    st.rerun()
    
    # Generate configuration object
    models_config = {
        "models": st.session_state.models,
        "providers": st.session_state.model_providers,
        "capabilities": st.session_state.model_capabilities,
        "default_model": st.session_state.default_model
    }
    
    # Show generated code
    st.subheader("Models Configuration Code")
    
    code_lines = ["# Models Configuration", ""]
    
    # Models definitions
    code_lines.append("# Define available models")
    code_lines.append("AVAILABLE_MODELS = {")
    
    for model_name, model_info in st.session_state.models.items():
        code_lines.append(f"    \"{model_name}\": {{")
        code_lines.append(f"        \"provider\": \"{model_info['provider']}\",")
        code_lines.append(f"        \"owned_by\": \"{model_info['owned_by']}\",")
        code_lines.append(f"        \"created\": \"{model_info['created']}\",")
        capabilities_str = ", ".join([f"\"{cap}\"" for cap in model_info["capabilities"]])
        code_lines.append(f"        \"capabilities\": [{capabilities_str}],")
        code_lines.append(f"        \"max_tokens\": {model_info['max_tokens']},")
        code_lines.append(f"        \"description\": \"{model_info['description']}\"")
        code_lines.append("    },")
    
    code_lines.append("}")
    code_lines.append("")
    
    # Default model
    code_lines.append(f"# Default model")
    code_lines.append(f"DEFAULT_MODEL = \"{st.session_state.default_model}\"")
    code_lines.append("")
    
    # Helper functions
    code_lines.append("# Helper functions for model capabilities")
    code_lines.append("def model_has_capability(model_name: str, capability: str) -> bool:")
    code_lines.append("    \"\"\"Check if a model has a specific capability\"\"\"")
    code_lines.append("    if model_name not in AVAILABLE_MODELS:")
    code_lines.append("        return False")
    code_lines.append("    return capability in AVAILABLE_MODELS[model_name][\"capabilities\"]")
    code_lines.append("")
    
    code_lines.append("def get_models_with_capability(capability: str) -> list[str]:")
    code_lines.append("    \"\"\"Get all models that have a specific capability\"\"\"")
    code_lines.append("    return [model for model, info in AVAILABLE_MODELS.items() ")
    code_lines.append("            if capability in info[\"capabilities\"]]")
    code_lines.append("")
    
    # Example usage
    code_lines.append("# Example usage with Agent:")
    code_lines.append("from agents import Agent")
    code_lines.append("")
    code_lines.append("def create_agent(model_name=None, require_reasoning=False):")
    code_lines.append("    # Use default model if none specified")
    code_lines.append("    if model_name is None:")
    code_lines.append("        model_name = DEFAULT_MODEL")
    code_lines.append("    ")
    code_lines.append("    # If reasoning is required, ensure model has reasoning capability")
    code_lines.append("    if require_reasoning and not model_has_capability(model_name, \"reasoning\"):")
    code_lines.append("        # Fall back to first model with reasoning capability")
    code_lines.append("        reasoning_models = get_models_with_capability(\"reasoning\")")
    code_lines.append("        if reasoning_models:")
    code_lines.append("            model_name = reasoning_models[0]")
    code_lines.append("        else:")
    code_lines.append("            raise ValueError(\"No models with reasoning capability available\")")
    code_lines.append("    ")
    code_lines.append("    return Agent(")
    code_lines.append("        name=\"My Agent\",")
    code_lines.append("        instructions=\"Help the user with their task\",")
    code_lines.append("        model=model_name")
    code_lines.append("    )")
    
    st.code("\n".join(code_lines), language="python")
    
    # Export as JSON option
    if st.button("Export Configuration as JSON", key="models_export_config_button"):
        json_str = json.dumps(models_config, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="models_config.json",
            mime="application/json",
            key="models_download_button"
        )
    
    return models_config

# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Models Configuration", layout="wide")
    config = models_module()
    
    with st.expander("Current Configuration"):
        st.json(config)