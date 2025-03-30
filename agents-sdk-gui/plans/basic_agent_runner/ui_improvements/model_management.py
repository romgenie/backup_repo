"""
Model Management Module for Agent Runner

This module provides functionality to manage different language models,
including configuration, selection, and validation of API keys.
"""

import os
import json
import enum
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any, Literal, Union, AsyncIterator
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

# Define ModelTracing enum
class ModelTracing(enum.Enum):
    """
    Tracing configuration for model calls.
    
    This enum controls the level of tracing information that should be collected
    during model execution.
    """
    DISABLED = 0  # Tracing is disabled entirely
    ENABLED = 1   # Tracing is enabled, and all data is included
    ENABLED_WITHOUT_DATA = 2  # Tracing is enabled, but inputs/outputs are not included

# Type aliases for clarity
TResponseInputItem = Dict[str, Any]
TResponseStreamEvent = Dict[str, Any]
ModelResponse = Dict[str, Any]
Tool = Dict[str, Any]
Handoff = Dict[str, Any]
AgentOutputSchema = Dict[str, Any]

# Model interface
class Model(ABC):
    """
    The base interface for calling an LLM.
    
    This abstract class defines the interface for model implementations.
    Different model providers (OpenAI, Anthropic, etc.) will implement
    concrete versions of this interface.
    """
    
    @abstractmethod
    async def get_response(
        self,
        system_instructions: Optional[str],
        input: Union[str, List[TResponseInputItem]],
        model_settings: 'ModelSettings',
        tools: List[Tool],
        output_schema: Optional[AgentOutputSchema],
        handoffs: List[Handoff],
        tracing: ModelTracing,
    ) -> ModelResponse:
        """
        Get a response from the model.
        
        Args:
            system_instructions: The system instructions to use
            input: The input items to the model, in OpenAI Responses format
            model_settings: The model settings to use
            tools: The tools available to the model
            output_schema: The output schema to use
            handoffs: The handoffs available to the model
            tracing: Tracing configuration
            
        Returns:
            The full model response
        """
        pass
    
    @abstractmethod
    async def stream_response(
        self,
        system_instructions: Optional[str],
        input: Union[str, List[TResponseInputItem]],
        model_settings: 'ModelSettings',
        tools: List[Tool],
        output_schema: Optional[AgentOutputSchema],
        handoffs: List[Handoff],
        tracing: ModelTracing,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """
        Stream a response from the model.
        
        Args:
            system_instructions: The system instructions to use
            input: The input items to the model, in OpenAI Responses format
            model_settings: The model settings to use
            tools: The tools available to the model
            output_schema: The output schema to use
            handoffs: The handoffs available to the model
            tracing: Tracing configuration
            
        Returns:
            An iterator of response stream events, in OpenAI Responses format
        """
        pass

# ModelProvider interface
class ModelProvider(ABC):
    """
    The base interface for a model provider.
    
    Model provider is responsible for looking up Models by name.
    """
    
    @abstractmethod
    def get_model(self, model_name: Optional[str]) -> Model:
        """
        Get a model by name.
        
        Args:
            model_name: The name of the model to get
            
        Returns:
            The model
        """
        pass

@dataclass
class ModelSettings:
    """
    Settings to use when calling an LLM.
    
    This class holds optional model configuration parameters (e.g. temperature, top_p, 
    penalties, truncation, etc.).
    
    Not all models/providers support all of these parameters, so please check the API 
    documentation for the specific model and provider you are using.
    """
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    tool_choice: Optional[Union[Literal["auto", "required", "none"], str]] = None
    parallel_tool_calls: Optional[bool] = None
    truncation: Optional[Literal['auto', 'disabled']] = None
    max_tokens: Optional[int] = None
    store: Optional[bool] = None
    tracing: Optional[ModelTracing] = None
    
    def resolve(self, override: Optional['ModelSettings'] = None) -> 'ModelSettings':
        """
        Produce a new ModelSettings by overlaying any non-None values from the override on top of this instance.
        
        Args:
            override: Another ModelSettings object to overlay on this one
            
        Returns:
            A new ModelSettings instance with overridden values
        """
        if override is None:
            return self
            
        # Start with current settings
        result = ModelSettings(**asdict(self))
        
        # Update with non-None values from override
        for field_name, field_value in asdict(override).items():
            if field_value is not None:
                setattr(result, field_name, field_value)
                
        return result
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelSettings':
        """Create ModelSettings from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in asdict(cls())})

# Define model providers and their respective models
SUPPORTED_PROVIDERS = {
    "OpenAI": {
        "models": [
            {"id": "gpt-4", "name": "GPT-4", "tokens": 8192, "description": "Most capable GPT-4 model"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "tokens": 128000, "description": "Improved GPT-4 with larger context window"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "tokens": 16384, "description": "Fast and efficient model"},
        ],
        "api_key_env": "OPENAI_API_KEY",
        "api_base_env": "OPENAI_API_BASE",
        "provider_type": "official",
        "client_type": "openai",
    },
    "Anthropic": {
        "models": [
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "tokens": 200000, "description": "Most capable Claude model"},
            {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet", "tokens": 180000, "description": "Balanced performance and efficiency"},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "tokens": 160000, "description": "Fastest Claude model"},
        ],
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_base_env": "ANTHROPIC_API_BASE",
        "provider_type": "official",
        "client_type": "anthropic",
    },
    "Cohere": {
        "models": [
            {"id": "command-r", "name": "Command R", "tokens": 128000, "description": "Cohere's recommended model"},
            {"id": "command-r-plus", "name": "Command R+", "tokens": 128000, "description": "Advanced reasoning capabilities"},
        ],
        "api_key_env": "COHERE_API_KEY",
        "api_base_env": None,
        "provider_type": "official",
        "client_type": "cohere",
    },
    # Custom OpenAI-compatible providers
    "Azure OpenAI": {
        "models": [
            {"id": "azure-gpt-4", "name": "Azure GPT-4", "tokens": 8192, "description": "GPT-4 on Azure"},
            {"id": "azure-gpt-4-32k", "name": "Azure GPT-4 32K", "tokens": 32768, "description": "GPT-4 with larger context on Azure"},
            {"id": "azure-gpt-35-turbo", "name": "Azure GPT-3.5 Turbo", "tokens": 16384, "description": "GPT-3.5 on Azure"},
        ],
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "api_base_env": "AZURE_OPENAI_ENDPOINT",
        "provider_type": "custom",
        "client_type": "openai_compatible",
        "requires_deployment": True,
    },
    "Custom Provider": {
        "models": [
            {"id": "custom-model", "name": "Custom Model", "tokens": 16384, "description": "Your custom model"},
        ],
        "api_key_env": "CUSTOM_API_KEY",
        "api_base_env": "CUSTOM_API_BASE",
        "provider_type": "custom",
        "client_type": "openai_compatible",
        "is_editable": True,
    },
}

# Default configuration file
CONFIG_DIR = Path(os.path.expanduser("~/.agent_runner"))
CONFIG_FILE = CONFIG_DIR / "model_config.json"

def initialize_model_config():
    """Initialize model configuration if it doesn't exist."""
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CONFIG_FILE.exists():
        # Default model settings for each provider
        default_model_settings = {
            "OpenAI": {
                "gpt-4": ModelSettings(temperature=0.7, top_p=1.0, max_tokens=4000).to_dict(),
                "gpt-4-turbo": ModelSettings(temperature=0.7, top_p=1.0, max_tokens=4000).to_dict(),
                "gpt-3.5-turbo": ModelSettings(temperature=0.8, top_p=1.0, max_tokens=2000).to_dict(),
            },
            "Anthropic": {
                "claude-3-opus-20240229": ModelSettings(temperature=0.7, top_p=0.9, max_tokens=4000).to_dict(),
                "claude-3-sonnet-20240229": ModelSettings(temperature=0.7, top_p=0.9, max_tokens=4000).to_dict(),
                "claude-3-haiku-20240307": ModelSettings(temperature=0.7, top_p=0.9, max_tokens=2000).to_dict(),
            },
            "Cohere": {
                "command-r": ModelSettings(temperature=0.8, top_p=0.95, max_tokens=2000).to_dict(),
                "command-r-plus": ModelSettings(temperature=0.7, top_p=0.95, max_tokens=2000).to_dict(),
            },
            "Azure OpenAI": {
                "azure-gpt-4": ModelSettings(temperature=0.7, top_p=1.0, max_tokens=4000).to_dict(),
                "azure-gpt-4-32k": ModelSettings(temperature=0.7, top_p=1.0, max_tokens=8000).to_dict(),
                "azure-gpt-35-turbo": ModelSettings(temperature=0.8, top_p=1.0, max_tokens=2000).to_dict(),
            },
            "Custom Provider": {
                "custom-model": ModelSettings(temperature=0.7, top_p=1.0, max_tokens=4000).to_dict(),
            }
        }
        
        # Default provider-specific settings
        default_provider_settings = {
            "Azure OpenAI": {
                "api_version": "2023-12-01-preview",
                "deployments": {
                    "azure-gpt-4": "gpt-4",
                    "azure-gpt-4-32k": "gpt-4-32k",
                    "azure-gpt-35-turbo": "gpt-35-turbo"
                }
            },
            "Custom Provider": {
                "api_base": "https://api.example.com/v1"
            }
        }
        
        default_config = {
            "default_provider": "OpenAI",
            "default_models": {
                "OpenAI": "gpt-3.5-turbo",
                "Anthropic": "claude-3-haiku-20240307",
                "Cohere": "command-r",
                "Azure OpenAI": "azure-gpt-35-turbo",
                "Custom Provider": "custom-model"
            },
            "api_keys": {},
            "model_settings": default_model_settings,
            "provider_settings": default_provider_settings,
            "integration_method": "agent_model",  # 'global_client', 'model_provider', or 'agent_model'
            "last_updated": datetime.now().isoformat(),
        }
        
        with open(CONFIG_FILE, "w") as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    # Configuration exists, load it
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            
        # Upgrade old format if needed
        if "custom_parameters" in config and "model_settings" not in config:
            # Convert old custom_parameters to new model_settings format
            model_settings = {}
            
            for key, params in config.get("custom_parameters", {}).items():
                if "." in key:  # Format was "provider.model"
                    provider, model = key.split(".", 1)
                    if provider not in model_settings:
                        model_settings[provider] = {}
                    model_settings[provider][model] = params
            
            # Add the new field and remove the old one
            config["model_settings"] = model_settings
            config.pop("custom_parameters", None)
            
            # Save the upgraded config
            save_model_config(config)
            
        return config
    except Exception as e:
        st.error(f"Error loading model configuration: {e}")
        return initialize_model_config()  # Reset to default if corrupt

def save_model_config(config: Dict):
    """Save model configuration to file."""
    try:
        config["last_updated"] = datetime.now().isoformat()
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving model configuration: {e}")
        return False

def get_current_model_info(provider: str, model_id: str) -> Dict:
    """Get information about a specific model."""
    provider_info = SUPPORTED_PROVIDERS.get(provider, {})
    models = provider_info.get("models", [])
    
    for model in models:
        if model["id"] == model_id:
            return model
    
    return {"id": model_id, "name": model_id, "tokens": 0, "description": "Unknown model"}

def get_model_settings(config: Dict, provider: str, model_id: str) -> ModelSettings:
    """Get model settings for a specific provider and model."""
    # Check if we have settings in the config
    if "model_settings" in config and provider in config["model_settings"] and model_id in config["model_settings"][provider]:
        settings_dict = config["model_settings"][provider][model_id]
        return ModelSettings.from_dict(settings_dict)
    
    # Default settings
    default_settings = {
        "OpenAI": ModelSettings(temperature=0.7, top_p=1.0, max_tokens=2000),
        "Anthropic": ModelSettings(temperature=0.7, top_p=0.9, max_tokens=2000),
        "Cohere": ModelSettings(temperature=0.8, top_p=0.95, max_tokens=2000),
    }
    
    return default_settings.get(provider, ModelSettings())

def validate_api_key(provider: str, api_key: str) -> Tuple[bool, str]:
    """Validate an API key for a provider."""
    if not api_key:
        return False, "API key is empty"
    
    # For demo purposes, we'll just check key format and basic validation
    if provider == "OpenAI" and not api_key.startswith("sk-"):
        return False, "OpenAI API keys should start with 'sk-'"
    elif provider == "Anthropic" and not api_key.startswith(("sk-ant-", "sk-")):
        return False, "Anthropic API keys typically start with 'sk-ant-' or 'sk-'"
    elif provider == "Cohere" and len(api_key) < 20:
        return False, "Cohere API keys should be at least 20 characters"
    
    # In a real implementation, you would make a test API call here
    # For this demo, we'll just return success
    return True, "API key looks valid"

def get_api_key_from_env(provider: str) -> str:
    """Get API key from environment variables."""
    env_var = SUPPORTED_PROVIDERS.get(provider, {}).get("api_key_env")
    if not env_var:
        return ""
    
    return os.environ.get(env_var, "")

def set_api_key_in_env(provider: str, api_key: str) -> bool:
    """Set API key in environment variables."""
    env_var = SUPPORTED_PROVIDERS.get(provider, {}).get("api_key_env")
    if not env_var:
        return False
    
    try:
        os.environ[env_var] = api_key
        return True
    except Exception:
        return False

def render_model_management():
    """Render the model management UI."""
    st.title("Model Management")
    st.markdown("Configure and manage the language models used by your agents.")
    
    # Initialize or load configuration
    if "model_config" not in st.session_state:
        st.session_state.model_config = initialize_model_config()
    
    config = st.session_state.model_config
    
    # Handle provider management
    with st.expander("Manage Providers", expanded=False):
        st.subheader("Provider Management")
        
        # Display current providers
        st.write("#### Current Providers")
        
        provider_list = list(SUPPORTED_PROVIDERS.keys())
        
        # Create a table of providers with edit/remove buttons
        for i, provider in enumerate(provider_list):
            provider_info = SUPPORTED_PROVIDERS.get(provider, {})
            
            # Determine if this provider can be edited or removed
            is_system = provider in ["OpenAI", "Anthropic", "Cohere"]
            is_editable = provider_info.get("is_editable", False) or not is_system
            is_removable = not is_system
            
            # Create a row for this provider
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            
            with col1:
                # Provider type icon
                if provider == "OpenAI":
                    icon = "🟢"  # Green circle for OpenAI
                elif provider == "Anthropic":
                    icon = "🟣"  # Purple circle for Anthropic
                elif provider == "Cohere":
                    icon = "🔵"  # Blue circle for Cohere
                else:
                    icon = "⚪"  # Default
                
                st.markdown(f"**{icon} {provider}**")
            
            with col2:
                client_type = provider_info.get("client_type", "unknown")
                provider_type = provider_info.get("provider_type", "unknown")
                st.write(f"Type: {provider_type} ({client_type})")
            
            with col3:
                if is_editable:
                    if st.button("Edit", key=f"edit_provider_{i}"):
                        # Store provider to edit in session state
                        st.session_state.editing_provider = provider
                        st.rerun()
            
            with col4:
                if is_removable:
                    if st.button("Remove", key=f"remove_provider_{i}"):
                        # Ask for confirmation
                        st.session_state.confirm_remove_provider = provider
                        st.rerun()
        
        # Confirmation dialog for removing provider
        if "confirm_remove_provider" in st.session_state and st.session_state.confirm_remove_provider:
            provider_to_remove = st.session_state.confirm_remove_provider
            
            st.warning(f"Are you sure you want to remove the provider '{provider_to_remove}'? This will also remove all associated models and settings.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Remove", key="confirm_remove"):
                    # Remove the provider
                    if provider_to_remove in SUPPORTED_PROVIDERS:
                        # Remove from SUPPORTED_PROVIDERS
                        del SUPPORTED_PROVIDERS[provider_to_remove]
                        
                        # Remove from configuration
                        if provider_to_remove in config.get("default_models", {}):
                            del config["default_models"][provider_to_remove]
                        
                        if provider_to_remove in config.get("model_settings", {}):
                            del config["model_settings"][provider_to_remove]
                        
                        if provider_to_remove in config.get("provider_settings", {}):
                            del config["provider_settings"][provider_to_remove]
                        
                        if provider_to_remove == config.get("default_provider"):
                            # Set a new default provider
                            remaining_providers = list(SUPPORTED_PROVIDERS.keys())
                            if remaining_providers:
                                config["default_provider"] = remaining_providers[0]
                        
                        # Save configuration
                        save_model_config(config)
                        st.session_state.model_config = config
                        
                        # Clear confirmation
                        st.session_state.confirm_remove_provider = None
                        
                        st.success(f"Provider '{provider_to_remove}' removed successfully.")
                        st.rerun()
            
            with col2:
                if st.button("Cancel", key="cancel_remove"):
                    # Clear confirmation
                    st.session_state.confirm_remove_provider = None
                    st.rerun()
        
        # Add new provider form
        st.write("#### Add New Provider")
        
        with st.form(key="add_provider_form"):
            new_provider_name = st.text_input("Provider Name", placeholder="e.g., MyCompany AI")
            
            new_provider_type = st.selectbox(
                "Provider Type",
                options=["custom"],
                format_func=lambda x: "Custom" if x == "custom" else x.title()
            )
            
            new_client_type = st.selectbox(
                "Client Type",
                options=["openai_compatible", "custom"],
                format_func=lambda x: "OpenAI Compatible" if x == "openai_compatible" else x.title(),
                help="OpenAI Compatible providers can use the OpenAI client with a custom base URL"
            )
            
            api_key_env = st.text_input(
                "API Key Environment Variable",
                placeholder="e.g., MY_COMPANY_API_KEY",
                help="The environment variable name for the API key"
            )
            
            api_base_env = st.text_input(
                "API Base URL Environment Variable",
                placeholder="e.g., MY_COMPANY_API_BASE",
                help="The environment variable name for the API base URL"
            )
            
            submit_provider = st.form_submit_button("Add Provider")
            
            if submit_provider and new_provider_name:
                # Check if provider already exists
                if new_provider_name in SUPPORTED_PROVIDERS:
                    st.error(f"Provider '{new_provider_name}' already exists.")
                else:
                    # Create new provider
                    SUPPORTED_PROVIDERS[new_provider_name] = {
                        "models": [],
                        "api_key_env": api_key_env,
                        "api_base_env": api_base_env,
                        "provider_type": new_provider_type,
                        "client_type": new_client_type,
                        "is_editable": True
                    }
                    
                    # Add to configuration
                    if "provider_settings" not in config:
                        config["provider_settings"] = {}
                    
                    config["provider_settings"][new_provider_name] = {
                        "api_base": "https://api.example.com/v1"
                    }
                    
                    if "default_models" not in config:
                        config["default_models"] = {}
                    
                    # No default model yet
                    config["default_models"][new_provider_name] = ""
                    
                    # Save configuration
                    save_model_config(config)
                    st.session_state.model_config = config
                    
                    st.success(f"Provider '{new_provider_name}' added. Now you can add models to it.")
                    st.rerun()
        
        # Edit provider form
        if "editing_provider" in st.session_state and st.session_state.editing_provider:
            provider_to_edit = st.session_state.editing_provider
            provider_info = SUPPORTED_PROVIDERS.get(provider_to_edit, {})
            
            st.write(f"#### Edit Provider: {provider_to_edit}")
            
            with st.form(key="edit_provider_form"):
                edit_api_key_env = st.text_input(
                    "API Key Environment Variable",
                    value=provider_info.get("api_key_env", ""),
                    help="The environment variable name for the API key"
                )
                
                edit_api_base_env = st.text_input(
                    "API Base URL Environment Variable",
                    value=provider_info.get("api_base_env", ""),
                    help="The environment variable name for the API base URL"
                )
                
                submit_edit = st.form_submit_button("Save Changes")
                
                if submit_edit:
                    # Update provider info
                    SUPPORTED_PROVIDERS[provider_to_edit]["api_key_env"] = edit_api_key_env
                    SUPPORTED_PROVIDERS[provider_to_edit]["api_base_env"] = edit_api_base_env
                    
                    # Clear editing state
                    st.session_state.editing_provider = None
                    
                    st.success(f"Provider '{provider_to_edit}' updated successfully.")
                    st.rerun()
            
            if st.button("Cancel Editing", key="cancel_edit_provider"):
                # Clear editing state
                st.session_state.editing_provider = None
                st.rerun()
    
    # Enhanced provider selection UI
    st.subheader("Select Provider")
    
    # Get provider info
    available_providers = list(SUPPORTED_PROVIDERS.keys())
    default_provider = config.get("default_provider", available_providers[0])
    
    # Create provider cards in a grid layout
    cols_per_row = min(4, len(available_providers))  # Maximum 4 cards per row
    rows = (len(available_providers) + cols_per_row - 1) // cols_per_row  # Calculate rows needed
    
    for row in range(rows):
        # Create columns for this row
        row_cols = st.columns(cols_per_row)
        
        # Fill columns with providers
        for col_idx in range(cols_per_row):
            provider_idx = row * cols_per_row + col_idx
            
            # Check if we have a provider for this slot
            if provider_idx < len(available_providers):
                provider = available_providers[provider_idx]
                
                with row_cols[col_idx]:
                    # Check if this is the default provider
                    is_default = provider == default_provider
                    card_border = "3px solid #4169E1" if is_default else "1px solid rgba(65, 105, 225, 0.2)"
                    card_bg = "rgba(65, 105, 225, 0.1)" if is_default else "rgba(65, 105, 225, 0.02)"
                    
                    # Provider logo/icon based on name
                    if provider == "OpenAI":
                        icon = "🟢"  # Green circle for OpenAI
                    elif provider == "Anthropic":
                        icon = "🟣"  # Purple circle for Anthropic
                    elif provider == "Cohere":
                        icon = "🔵"  # Blue circle for Cohere
                    else:
                        icon = "⚪"  # Default
                    
                    # Create a clickable provider card
                    html = f"""
                    <div class="provider-card" style="border: {card_border}; background-color: {card_bg};">
                        <div class="provider-icon">{icon}</div>
                        <div class="provider-name">{provider}</div>
                        <div class="provider-status">{"✓ Default" if is_default else ""}</div>
                    </div>
                    <style>
                    .provider-card {{
                        padding: 16px;
                        border-radius: 8px;
                        text-align: center;
                        cursor: pointer;
                        height: 100%;
                        transition: all 0.2s ease;
                        box-shadow: {f"0 4px 8px rgba(65, 105, 225, 0.2)" if is_default else "none"};
                    }}
                    .provider-card:hover {{
                        background-color: rgba(65, 105, 225, 0.15);
                        border-color: rgba(65, 105, 225, 0.4);
                    }}
                    .provider-icon {{
                        font-size: 24px;
                        margin-bottom: 8px;
                    }}
                    .provider-name {{
                        font-weight: 500;
                        margin-bottom: 4px;
                    }}
                    .provider-status {{
                        font-size: 12px;
                        color: #4169E1;
                    }}
                    </style>
                    """
                    st.markdown(html, unsafe_allow_html=True)
                    
                    # Button to select this provider
                    if st.button(f"Select {provider}", key=f"select_{provider}", 
                                disabled=is_default,
                                help=f"Use {provider} as your AI model provider"):
                        config["default_provider"] = provider
                        save_model_config(config)
                        st.session_state.model_config = config
                        st.success(f"{provider} set as the default provider")
                        st.rerun()
    
    st.divider()
    
    # Provider selection (keeping the dropdown as an alternative)
    selected_provider = st.selectbox(
        "Active Provider", 
        available_providers,
        index=available_providers.index(default_provider) if default_provider in available_providers else 0
    )
    
    # Get provider information
    provider_info = SUPPORTED_PROVIDERS.get(selected_provider, {})
    models = provider_info.get("models", [])
    api_key_env = provider_info.get("api_key_env")
    
    # API Key management
    st.subheader("API Key Management")
    
    # Check for existing API key
    api_key = get_api_key_from_env(selected_provider)
    saved_key = config.get("api_keys", {}).get(selected_provider, "")
    key_status = ""
    
    if api_key:
        key_status = "✅ API key found in environment"
    elif saved_key:
        key_status = "⚠️ API key found in configuration but not in environment"
    else:
        key_status = "❌ No API key configured"
    
    st.info(key_status)
    
    # API Key input
    api_key_tabs = st.tabs(["Enter Key", "Load from Environment", "Load from File"])
    
    with api_key_tabs[0]:
        manual_key = st.text_input(
            "API Key", 
            type="password", 
            help=f"Enter your {selected_provider} API key"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Validate Key", key="validate_manual_key"):
                if not manual_key:
                    st.warning("Please enter an API key")
                else:
                    valid, message = validate_api_key(selected_provider, manual_key)
                    if valid:
                        st.success(message)
                    else:
                        st.error(message)
        
        with col2:
            if st.button("Save Key", key="save_manual_key"):
                if not manual_key:
                    st.warning("Please enter an API key")
                else:
                    valid, message = validate_api_key(selected_provider, manual_key)
                    if valid:
                        # Save to configuration
                        config["api_keys"][selected_provider] = manual_key
                        save_model_config(config)
                        
                        # Set in environment
                        set_api_key_in_env(selected_provider, manual_key)
                        
                        st.success(f"{selected_provider} API key saved and active")
                        st.session_state.model_config = config
                    else:
                        st.error(f"Invalid API key: {message}")
    
    with api_key_tabs[1]:
        env_var = provider_info.get("api_key_env", "")
        st.info(f"Environment variable: {env_var}")
        
        env_key = os.environ.get(env_var, "")
        if env_key:
            st.success(f"API key found in environment variable {env_var}")
            
            if st.button("Save to Configuration", key="save_env_key"):
                config["api_keys"][selected_provider] = env_key
                save_model_config(config)
                st.success(f"{selected_provider} API key saved to configuration")
                st.session_state.model_config = config
        else:
            st.warning(f"No API key found in environment variable {env_var}")
    
    with api_key_tabs[2]:
        file_path = st.text_input(
            ".env File Path",
            value=str(Path.home() / ".env"),
            help="Path to a .env file containing API keys"
        )
        
        if st.button("Load from File", key="load_file_key"):
            try:
                env_path = Path(file_path)
                if not env_path.exists():
                    st.error(f"File does not exist: {file_path}")
                else:
                    with open(env_path, "r") as f:
                        env_content = f.read()
                    
                    import re
                    env_var = provider_info.get("api_key_env", "")
                    api_key_match = re.search(f"{env_var}=([^\n]+)", env_content)
                    
                    if api_key_match:
                        file_key = api_key_match.group(1).strip()
                        valid, message = validate_api_key(selected_provider, file_key)
                        
                        if valid:
                            # Save to configuration
                            config["api_keys"][selected_provider] = file_key
                            save_model_config(config)
                            
                            # Set in environment
                            set_api_key_in_env(selected_provider, file_key)
                            
                            st.success(f"{selected_provider} API key loaded from file and set")
                            st.session_state.model_config = config
                        else:
                            st.error(f"Invalid API key from file: {message}")
                    else:
                        st.warning(f"No {env_var} found in {file_path}")
            except Exception as e:
                st.error(f"Error loading from file: {str(e)}")
    
    # Model selection
    st.subheader("Model Selection")
    
    default_model_id = config.get("default_models", {}).get(
        selected_provider, 
        models[0]["id"] if models else ""
    )
    
    # Add model management expander
    with st.expander("Manage Models", expanded=False):
        st.subheader(f"Model Management for {selected_provider}")
        
        # Display current models in a table
        if models:
            st.write("#### Current Models")
            
            for i, model in enumerate(models):
                # Create a row for this model
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                
                with col1:
                    st.markdown(f"**{model['name']}** ({model['id']})")
                
                with col2:
                    st.write(f"Tokens: {model['tokens']:,}")
                
                with col3:
                    # Check if this model can be edited
                    if provider_info.get("is_editable", False):
                        if st.button("Edit", key=f"edit_model_{i}"):
                            # Store model to edit in session state
                            st.session_state.editing_model = {
                                "provider": selected_provider,
                                "index": i,
                                "model": model
                            }
                            st.rerun()
                
                with col4:
                    # Check if this model can be removed
                    if provider_info.get("is_editable", False):
                        if st.button("Remove", key=f"remove_model_{i}"):
                            # Ask for confirmation
                            st.session_state.confirm_remove_model = {
                                "provider": selected_provider,
                                "index": i,
                                "model": model
                            }
                            st.rerun()
            
            # Confirmation dialog for removing model
            if "confirm_remove_model" in st.session_state and st.session_state.confirm_remove_model:
                removal_info = st.session_state.confirm_remove_model
                
                if removal_info["provider"] == selected_provider:
                    model_to_remove = removal_info["model"]
                    model_idx = removal_info["index"]
                    
                    st.warning(f"Are you sure you want to remove the model '{model_to_remove['name']}'?")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Yes, Remove", key="confirm_remove_model"):
                            # Remove the model from SUPPORTED_PROVIDERS
                            if 0 <= model_idx < len(SUPPORTED_PROVIDERS[selected_provider]["models"]):
                                # Get the model ID before removing
                                model_id = SUPPORTED_PROVIDERS[selected_provider]["models"][model_idx]["id"]
                                
                                # Remove from SUPPORTED_PROVIDERS
                                del SUPPORTED_PROVIDERS[selected_provider]["models"][model_idx]
                                
                                # Remove from configuration
                                if selected_provider in config.get("model_settings", {}) and model_id in config["model_settings"][selected_provider]:
                                    del config["model_settings"][selected_provider][model_id]
                                
                                # If this was the default model, update the default
                                if model_id == config.get("default_models", {}).get(selected_provider):
                                    if SUPPORTED_PROVIDERS[selected_provider]["models"]:
                                        # Set first available model as default
                                        config["default_models"][selected_provider] = SUPPORTED_PROVIDERS[selected_provider]["models"][0]["id"]
                                    else:
                                        # No models left
                                        config["default_models"][selected_provider] = ""
                                
                                # Save configuration
                                save_model_config(config)
                                st.session_state.model_config = config
                                
                                # Clear confirmation
                                st.session_state.confirm_remove_model = None
                                
                                st.success(f"Model '{model_to_remove['name']}' removed successfully.")
                                st.rerun()
                    
                    with col2:
                        if st.button("Cancel", key="cancel_remove_model"):
                            # Clear confirmation
                            st.session_state.confirm_remove_model = None
                            st.rerun()
        else:
            st.info(f"No models defined for {selected_provider}. Add a model below.")
        
        # Add new model form
        st.write("#### Add New Model")
        
        with st.form(key="add_model_form"):
            new_model_id = st.text_input("Model ID", placeholder="e.g., my-custom-model")
            new_model_name = st.text_input("Display Name", placeholder="e.g., My Custom Model")
            new_model_tokens = st.number_input("Context Size (tokens)", min_value=1, value=16384)
            new_model_description = st.text_area("Description", placeholder="Describe the capabilities and purpose of this model")
            
            submit_model = st.form_submit_button("Add Model")
            
            if submit_model and new_model_id and new_model_name:
                # Check if model already exists
                if any(model["id"] == new_model_id for model in models):
                    st.error(f"Model with ID '{new_model_id}' already exists.")
                else:
                    # Create new model definition
                    new_model = {
                        "id": new_model_id,
                        "name": new_model_name,
                        "tokens": new_model_tokens,
                        "description": new_model_description
                    }
                    
                    # Add to SUPPORTED_PROVIDERS
                    SUPPORTED_PROVIDERS[selected_provider]["models"].append(new_model)
                    
                    # Add default settings
                    if "model_settings" not in config:
                        config["model_settings"] = {}
                    if selected_provider not in config["model_settings"]:
                        config["model_settings"][selected_provider] = {}
                    
                    # Add default settings for this model
                    config["model_settings"][selected_provider][new_model_id] = ModelSettings(
                        temperature=0.7,
                        top_p=1.0,
                        max_tokens=min(4000, new_model_tokens)
                    ).to_dict()
                    
                    # If this is the first model, set it as default
                    if not config.get("default_models", {}).get(selected_provider):
                        if "default_models" not in config:
                            config["default_models"] = {}
                        config["default_models"][selected_provider] = new_model_id
                    
                    # Save configuration
                    save_model_config(config)
                    st.session_state.model_config = config
                    
                    st.success(f"Model '{new_model_name}' added successfully.")
                    st.rerun()
        
        # Edit model form
        if "editing_model" in st.session_state and st.session_state.editing_model:
            edit_info = st.session_state.editing_model
            
            if edit_info["provider"] == selected_provider:
                model_to_edit = edit_info["model"]
                
                st.write(f"#### Edit Model: {model_to_edit['name']}")
                
                with st.form(key="edit_model_form"):
                    edit_model_name = st.text_input("Display Name", value=model_to_edit.get("name", ""))
                    edit_model_tokens = st.number_input("Context Size (tokens)", min_value=1, value=model_to_edit.get("tokens", 16384))
                    edit_model_description = st.text_area("Description", value=model_to_edit.get("description", ""))
                    
                    submit_edit_model = st.form_submit_button("Save Changes")
                    
                    if submit_edit_model:
                        # Update the model
                        model_idx = edit_info["index"]
                        
                        if 0 <= model_idx < len(SUPPORTED_PROVIDERS[selected_provider]["models"]):
                            # Update the model definition
                            SUPPORTED_PROVIDERS[selected_provider]["models"][model_idx].update({
                                "name": edit_model_name,
                                "tokens": edit_model_tokens,
                                "description": edit_model_description
                            })
                            
                            # Clear editing state
                            st.session_state.editing_model = None
                            
                            st.success(f"Model '{edit_model_name}' updated successfully.")
                            st.rerun()
                
                if st.button("Cancel Editing", key="cancel_edit_model"):
                    # Clear editing state
                    st.session_state.editing_model = None
                    st.rerun()
    
    # Create visual model selection cards
    if models:
        st.write("Available models for " + selected_provider + ":")
        
        # Organize models into rows of 2 or 3 depending on count
        cols_per_row = 2 if len(models) <= 4 else 3
        rows = (len(models) + cols_per_row - 1) // cols_per_row  # Calculate rows needed
        
        for row in range(rows):
            # Create columns for this row
            row_cols = st.columns(cols_per_row)
            
            # Fill columns with models
            for col_idx in range(cols_per_row):
                model_idx = row * cols_per_row + col_idx
                
                # Check if we have a model for this slot
                if model_idx < len(models):
                    model = models[model_idx]
                    
                    with row_cols[col_idx]:
                        # Check if this is the default model
                        is_default = model["id"] == default_model_id
                        card_border = "3px solid #4169E1" if is_default else "1px solid rgba(65, 105, 225, 0.2)"
                        card_bg = "rgba(65, 105, 225, 0.1)" if is_default else "rgba(65, 105, 225, 0.02)"
                        
                        # Create model card
                        html = f"""
                        <div class="model-card" style="border: {card_border}; background-color: {card_bg};">
                            <div class="model-name">{model["name"]}</div>
                            <div class="model-tokens">{model["tokens"]:,} tokens</div>
                            <div class="model-description">{model["description"]}</div>
                            <div class="model-status">{"✓ Current Default" if is_default else ""}</div>
                        </div>
                        <style>
                        .model-card {{
                            padding: 16px;
                            border-radius: 8px;
                            height: 100%;
                            transition: all 0.2s ease;
                            box-shadow: {f"0 4px 8px rgba(65, 105, 225, 0.2)" if is_default else "none"};
                            margin-bottom: 16px;
                        }}
                        .model-card:hover {{
                            background-color: rgba(65, 105, 225, 0.15);
                            border-color: rgba(65, 105, 225, 0.4);
                        }}
                        .model-name {{
                            font-weight: 600;
                            font-size: 16px;
                            margin-bottom: 4px;
                            color: #343D4A;
                        }}
                        .model-tokens {{
                            color: #626E7F;
                            font-size: 14px;
                            margin-bottom: 8px;
                        }}
                        .model-description {{
                            font-size: 13px;
                            color: #626E7F;
                            margin-bottom: 12px;
                            min-height: 40px;
                        }}
                        .model-status {{
                            font-size: 12px;
                            color: #4169E1;
                            font-weight: 500;
                        }}
                        </style>
                        """
                        st.markdown(html, unsafe_allow_html=True)
                        
                        # Button to select this model
                        if st.button(f"Use {model['name']}", key=f"select_model_{model['id']}",
                                    disabled=is_default):
                            # Set as default model
                            config["default_models"][selected_provider] = model["id"]
                            if selected_provider == config["default_provider"]:
                                st.success(f"{model['name']} set as the default model for {selected_provider}")
                            else:
                                st.success(f"{model['name']} set as the default model when using {selected_provider}")
                            
                            save_model_config(config)
                            st.session_state.model_config = config
                            st.rerun()
        
        # Show dropdown as alternative selection method
        st.markdown("---")
        model_options = [(m["id"], f"{m['name']} - {m['description']}") for m in models]
        selected_model_idx = next((i for i, (id, _) in enumerate(model_options) if id == default_model_id), 0)
        
        selected_model = st.selectbox(
            "Select Model via Dropdown",
            options=[id for id, _ in model_options],
            format_func=lambda x: next((desc for id, desc in model_options if id == x), x),
            index=selected_model_idx if model_options else 0
        )
        
        # Show currently selected model details
        st.markdown("### Current Default Model")
        model_info = get_current_model_info(selected_provider, default_model_id)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", model_info["name"])
        with col2:
            st.metric("Max Tokens", f"{model_info['tokens']:,}")
        
        st.info(model_info["description"])
        
    else:
        st.warning(f"No models available for {selected_provider}")
        selected_model = None
    
    # Set as default provider
    if selected_provider != config.get("default_provider"):
        if st.button("Set as Default Provider", key="set_default_provider"):
            config["default_provider"] = selected_provider
            save_model_config(config)
            st.session_state.model_config = config
            st.success(f"{selected_provider} set as the default provider")
    
    # Provider-specific settings
    provider_info = SUPPORTED_PROVIDERS.get(selected_provider, {})
    if provider_info.get("provider_type") == "custom":
        st.subheader("Provider Configuration")
        
        # Get provider settings from config
        provider_settings = config.get("provider_settings", {}).get(selected_provider, {})
        
        # Azure OpenAI specific settings
        if selected_provider == "Azure OpenAI":
            # API version
            api_version = st.text_input(
                "API Version", 
                value=provider_settings.get("api_version", "2023-12-01-preview"),
                help="The Azure OpenAI API version to use"
            )
            
            # Model to deployment mapping
            st.write("#### Deployment Mappings")
            st.write("Map model IDs to Azure deployment names:")
            
            # Get existing deployments or create empty dict
            deployments = provider_settings.get("deployments", {})
            
            # Create a form for each model
            for model in provider_info.get("models", []):
                model_id = model["id"]
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**{model['name']}**")
                with col2:
                    deployments[model_id] = st.text_input(
                        f"Deployment for {model_id}",
                        value=deployments.get(model_id, model_id.replace("azure-", "")),
                        key=f"deployment_{model_id}",
                        help="The Azure deployment name for this model"
                    )
            
            # Save button for Azure settings
            if st.button("Save Azure Configuration", key="save_azure_config"):
                # Update provider settings
                if "provider_settings" not in config:
                    config["provider_settings"] = {}
                if selected_provider not in config["provider_settings"]:
                    config["provider_settings"][selected_provider] = {}
                
                # Update settings
                config["provider_settings"][selected_provider]["api_version"] = api_version
                config["provider_settings"][selected_provider]["deployments"] = deployments
                
                # Save configuration
                save_model_config(config)
                st.session_state.model_config = config
                st.success(f"Azure OpenAI configuration saved")
        
        # Custom provider settings
        elif selected_provider == "Custom Provider" or provider_info.get("is_editable"):
            # API base URL
            api_base = st.text_input(
                "API Base URL", 
                value=provider_settings.get("api_base", "https://api.example.com/v1"),
                help="The base URL for the API"
            )
            
            # Custom models section
            st.write("#### Custom Models")
            
            if provider_info.get("is_editable"):
                # Get existing models or use default
                custom_models = provider_info.get("models", [{"id": "custom-model", "name": "Custom Model", "tokens": 16384, "description": "Your custom model"}])
                
                # Allow adding a new model
                with st.form(key="add_model_form"):
                    st.write("Add a new model:")
                    new_model_id = st.text_input("Model ID", value="", key="new_model_id", help="The identifier for the model")
                    new_model_name = st.text_input("Display Name", value="", key="new_model_name", help="The name to display for the model")
                    new_model_tokens = st.number_input("Context Tokens", value=16384, key="new_model_tokens", help="The maximum context size in tokens")
                    new_model_desc = st.text_area("Description", value="", key="new_model_desc", help="A description of the model")
                    
                    submit = st.form_submit_button("Add Model")
                    if submit and new_model_id and new_model_name:
                        # Add to the SUPPORTED_PROVIDERS dictionary
                        new_model = {
                            "id": new_model_id,
                            "name": new_model_name,
                            "tokens": new_model_tokens,
                            "description": new_model_desc
                        }
                        
                        # Update the provider's models list
                        SUPPORTED_PROVIDERS[selected_provider]["models"].append(new_model)
                        
                        # Add default model settings
                        if "model_settings" not in config:
                            config["model_settings"] = {}
                        if selected_provider not in config["model_settings"]:
                            config["model_settings"][selected_provider] = {}
                        
                        config["model_settings"][selected_provider][new_model_id] = ModelSettings(
                            temperature=0.7, top_p=1.0, max_tokens=min(4000, new_model_tokens)
                        ).to_dict()
                        
                        # Save configuration
                        save_model_config(config)
                        st.session_state.model_config = config
                        st.success(f"Model {new_model_name} added to {selected_provider}")
                        st.rerun()
            
            # Save button for Custom settings
            if st.button("Save Custom Configuration", key="save_custom_config"):
                # Update provider settings
                if "provider_settings" not in config:
                    config["provider_settings"] = {}
                if selected_provider not in config["provider_settings"]:
                    config["provider_settings"][selected_provider] = {}
                
                # Update settings
                config["provider_settings"][selected_provider]["api_base"] = api_base
                
                # Save configuration
                save_model_config(config)
                st.session_state.model_config = config
                st.success(f"Custom provider configuration saved")
        
        st.divider()
    
    # Integration method selection
    st.subheader("Integration Method")
    
    integration_methods = [
        "agent_model", 
        "global_client", 
        "model_provider"
    ]
    
    integration_method_labels = {
        "agent_model": "Agent.model (Default - per-agent model selection)",
        "global_client": "set_default_openai_client (Global client for OpenAI-compatible APIs)",
        "model_provider": "ModelProvider (Custom provider implementation)"
    }
    
    # Get current method
    current_method = config.get("integration_method", "agent_model")
    
    # Create radio buttons for selection
    integration_method = st.radio(
        "Choose integration method",
        options=integration_methods,
        index=integration_methods.index(current_method),
        format_func=lambda x: integration_method_labels.get(x, x),
        help="Select how to integrate custom providers with the Agent SDK"
    )
    
    # Show help text based on selected method
    if integration_method == "agent_model":
        st.info("""
        **Agent.model** lets you specify the model directly on each Agent instance.
        
        This approach provides the most flexibility, allowing you to mix and match different 
        providers for different agents. Each agent is configured with its specific model.
        """)
    elif integration_method == "global_client":
        st.info("""
        **set_default_openai_client** creates a global AsyncOpenAI instance for all agents.
        
        This approach is useful for providers with OpenAI-compatible APIs like Azure OpenAI.
        You can set custom base_url and api_key globally. This works best when all your
        agents use the same provider.
        """)
    elif integration_method == "model_provider":
        st.warning("""
        **ModelProvider** creates a custom provider implementation at the Runner.run level.
        
        This is the most advanced approach, requiring implementation of the Model and ModelProvider
        interfaces. Use this when you need complete control over how models are instantiated
        and how requests are made.
        
        Note: This implementation provides skeleton code that would need to be customized
        for production use.
        """)
    
    # Save button for integration method
    if st.button("Save Integration Method", key="save_integration_method"):
        # Update config
        config["integration_method"] = integration_method
        
        # Save configuration
        save_model_config(config)
        st.session_state.model_config = config
        st.success(f"Integration method saved: {integration_method_labels[integration_method]}")
        
        # Apply the integration method if possible
        if integration_method == "global_client":
            if setup_global_openai_client(config):
                st.success("Global OpenAI client configured successfully")
            else:
                st.warning("Could not setup global OpenAI client - check provider settings")
    
    st.divider()
        
    # Advanced model parameters
    st.subheader("Model Parameters")
    
    # Get model settings for the selected model
    if selected_model:
        # Get the current settings
        model_settings = get_model_settings(config, selected_provider, selected_model)
        
        # Create a more structured parameter editing interface
        st.write(f"Configure parameters for {model_info['name']}:")
        
        # Use tabs to organize parameters by category
        param_tabs = st.tabs([
            "Basic Parameters", 
            "Advanced Parameters",
            "Tool Behavior",
            "Output Settings",
            "Tracing"
        ])
        
        # Basic Parameters Tab
        with param_tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,  # Some models support values above 1.0
                    value=model_settings.temperature if model_settings.temperature is not None else 0.7,
                    step=0.05,
                    help="Controls randomness. Lower values are more deterministic, higher values more creative."
                )
                
                top_p = st.slider(
                    "Top P",
                    min_value=0.0,
                    max_value=1.0,
                    value=model_settings.top_p if model_settings.top_p is not None else 1.0,
                    step=0.05,
                    help="Controls diversity via nucleus sampling. 1.0 considers all tokens."
                )
            
            with col2:
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=1,
                    max_value=model_info.get("tokens", 4096),
                    value=model_settings.max_tokens if model_settings.max_tokens is not None else 1000,
                    help="Maximum number of output tokens to generate."
                )
                
                # Truncation setting
                truncation_options = [None, "auto", "disabled"]
                truncation_index = 0 if model_settings.truncation is None else truncation_options.index(model_settings.truncation)
                
                truncation = st.selectbox(
                    "Truncation Strategy",
                    options=truncation_options,
                    index=truncation_index,
                    format_func=lambda x: "Default" if x is None else x,
                    help="How to handle context window constraints. 'auto' truncates as needed, 'disabled' errors instead."
                )
        
        # Advanced Parameters Tab
        with param_tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                frequency_penalty = st.slider(
                    "Frequency Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=model_settings.frequency_penalty if model_settings.frequency_penalty is not None else 0.0,
                    step=0.1,
                    help="Reduces repetition by penalizing tokens that have appeared frequently. Negative values encourage repetition."
                )
            
            with col2:
                presence_penalty = st.slider(
                    "Presence Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=model_settings.presence_penalty if model_settings.presence_penalty is not None else 0.0,
                    step=0.1,
                    help="Reduces repetition by penalizing tokens that have appeared at all. Negative values encourage repetition."
                )
        
        # Tool Behavior Tab
        with param_tabs[2]:
            # Tool choice options
            tool_choice_options = [None, "auto", "required", "none"]
            tool_choice_labels = ["Default", "Auto", "Required", "None"]
            tool_choice_index = 0
            
            # Find index for the current value
            if model_settings.tool_choice is not None:
                try:
                    tool_choice_index = tool_choice_options.index(model_settings.tool_choice)
                except ValueError:
                    # If it's a custom value not in the list
                    tool_choice_options.append(model_settings.tool_choice)
                    tool_choice_labels.append(f"Custom: {model_settings.tool_choice}")
                    tool_choice_index = len(tool_choice_options) - 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                tool_choice = st.selectbox(
                    "Tool Choice",
                    options=tool_choice_options,
                    index=tool_choice_index,
                    format_func=lambda x, labels=tool_choice_labels, options=tool_choice_options: 
                        labels[options.index(x)] if x in options else str(x),
                    help="Controls how the model uses tools. 'auto' lets the model decide, 'required' forces tool use, 'none' prevents tool use."
                )
            
            with col2:
                parallel_tool_calls = st.selectbox(
                    "Parallel Tool Calls",
                    options=[None, True, False],
                    index=0 if model_settings.parallel_tool_calls is None else (1 if model_settings.parallel_tool_calls else 2),
                    format_func=lambda x: "Default" if x is None else ("Enabled" if x else "Disabled"),
                    help="Whether the model can call multiple tools in parallel."
                )
        
        # Output Settings Tab
        with param_tabs[3]:
            store = st.selectbox(
                "Store Responses",
                options=[None, True, False],
                index=0 if model_settings.store is None else (1 if model_settings.store else 2),
                format_func=lambda x: "Default" if x is None else ("Yes" if x else "No"),
                help="Whether to store the generated model response for later retrieval."
            )
            
            # Add a description of these settings
            st.markdown("""
            ### Parameter Notes:
            
            - **Temperature** controls the randomness of output. Higher values make output more creative but less predictable.
            - **Top P** controls diversity by only considering tokens that make up the specified probability mass.
            - **Frequency penalty** reduces word repetition (positive values) or encourages it (negative values).
            - **Presence penalty** penalizes tokens that have appeared at all.
            
            Not all parameters are supported by all models and providers.
            """)
            
        # Tracing Tab
        with param_tabs[4]:
            # Convert enum to options list
            tracing_options = [None, ModelTracing.DISABLED, ModelTracing.ENABLED, ModelTracing.ENABLED_WITHOUT_DATA]
            tracing_labels = ["Default", "Disabled", "Enabled (Full)", "Enabled (No Data)"]
            
            # Find index for current value
            tracing_index = 0  # Default
            if model_settings.tracing is not None:
                try:
                    tracing_index = tracing_options.index(model_settings.tracing)
                except ValueError:
                    # If somehow the value isn't in our list
                    pass
            
            # Tracing setting
            tracing = st.selectbox(
                "Tracing Mode",
                options=tracing_options,
                index=tracing_index,
                format_func=lambda x: "Default" if x is None else (
                    "Disabled" if x == ModelTracing.DISABLED else (
                        "Enabled (Full)" if x == ModelTracing.ENABLED else "Enabled (No Data)"
                    )
                ),
                help="Controls the level of tracing information collected during model execution."
            )
            
            # Add explanation of tracing options
            st.info("""
            **Tracing Options:**
            
            - **Disabled**: No tracing information is collected
            - **Enabled (Full)**: Full tracing with all inputs and outputs
            - **Enabled (No Data)**: Tracing is enabled but input/output data is excluded
            
            Tracing can be useful for debugging and monitoring model behavior, but may have performance implications.
            """)
            
            # Visual indicator of data usage implications
            if tracing == ModelTracing.DISABLED:
                st.success("⚡ Optimal performance, no additional data stored")
            elif tracing == ModelTracing.ENABLED:
                st.warning("🔍 Full trace data will be collected (including inputs/outputs)")
            elif tracing == ModelTracing.ENABLED_WITHOUT_DATA:
                st.info("🔄 Trace structure collected without sensitive input/output data")
        
        # Save button for all parameters
        if st.button("Save Parameters", key="save_params"):
            # Create ModelSettings object from the UI values
            updated_settings = ModelSettings(
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                truncation=truncation,
                max_tokens=max_tokens,
                store=store,
                tracing=tracing
            )
            
            # Ensure model_settings structure exists in config
            if "model_settings" not in config:
                config["model_settings"] = {}
            if selected_provider not in config["model_settings"]:
                config["model_settings"][selected_provider] = {}
            
            # Save settings for the provider and model
            config["model_settings"][selected_provider][selected_model] = updated_settings.to_dict()
            
            # Save configuration
            save_model_config(config)
            st.session_state.model_config = config
            st.success(f"Parameters for {model_info['name']} saved")
            
        # Reset button to restore defaults
        if st.button("Reset to Defaults", key="reset_params"):
            # Default settings for the provider
            default_settings = {
                "OpenAI": ModelSettings(temperature=0.7, top_p=1.0, max_tokens=2000),
                "Anthropic": ModelSettings(temperature=0.7, top_p=0.9, max_tokens=2000),
                "Cohere": ModelSettings(temperature=0.8, top_p=0.95, max_tokens=2000),
            }.get(selected_provider, ModelSettings())
            
            # Update config
            if "model_settings" in config and selected_provider in config["model_settings"]:
                config["model_settings"][selected_provider][selected_model] = default_settings.to_dict()
                
                # Save configuration
                save_model_config(config)
                st.session_state.model_config = config
                st.success(f"Parameters for {model_info['name']} reset to defaults")
                st.rerun()
    else:
        st.warning("Please select a model to configure parameters")

def get_default_model() -> Tuple[str, str, ModelSettings]:
    """Get the default model provider, model ID, and settings.
    
    Returns:
        Tuple[str, str, ModelSettings]: Provider name, model ID, and model settings
    """
    # Initialize if needed
    if "model_config" not in st.session_state:
        st.session_state.model_config = initialize_model_config()
    
    config = st.session_state.model_config
    provider = config.get("default_provider", "OpenAI")
    model_id = config.get("default_models", {}).get(provider, "")
    
    # If no model ID is specified, get the first one for the provider
    if not model_id and provider in SUPPORTED_PROVIDERS:
        models = SUPPORTED_PROVIDERS[provider].get("models", [])
        if models:
            model_id = models[0]["id"]
    
    # Get model settings
    settings = get_model_settings(config, provider, model_id)
    
    return provider, model_id, settings

# Function to setup global OpenAI client (Method 1)
def setup_global_openai_client(config: Dict) -> bool:
    """
    Setup a global OpenAI client for LLM providers with compatible API.
    
    This function configures an OpenAI-compatible client globally.
    It's useful for providers like Azure OpenAI or other compatible APIs.
    
    Args:
        config: Configuration dictionary with provider settings
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        from openai import AsyncOpenAI
        import agents
        
        # Get current provider info
        provider = config.get("default_provider", "OpenAI")
        provider_info = SUPPORTED_PROVIDERS.get(provider, {})
        
        # Skip for non-OpenAI compatible providers
        if provider_info.get("client_type") != "openai" and provider_info.get("client_type") != "openai_compatible":
            print(f"Provider {provider} is not OpenAI compatible, skipping global client setup")
            return False
        
        # Get API key and base URL
        api_key = os.environ.get(provider_info.get("api_key_env", ""))
        api_base = os.environ.get(provider_info.get("api_base_env", ""))
        
        if not api_key:
            api_key = config.get("api_keys", {}).get(provider)
        
        if not api_key:
            print(f"API key not found for {provider}")
            return False
        
        # For non-official providers, ensure we have a base URL
        if provider_info.get("provider_type") == "custom" and not api_base:
            api_base = config.get("provider_settings", {}).get(provider, {}).get("api_base")
            
        if provider_info.get("provider_type") == "custom" and not api_base:
            print(f"API base URL not found for custom provider {provider}")
            return False
            
        # Create client
        client_args = {"api_key": api_key}
        
        # Add base URL if specified
        if api_base:
            client_args["base_url"] = api_base
            
        # Add Azure-specific arguments if needed
        if provider == "Azure OpenAI":
            api_version = config.get("provider_settings", {}).get(provider, {}).get("api_version", "2023-12-01-preview")
            client_args["api_version"] = api_version
            
        # Setup global client
        openai_client = AsyncOpenAI(**client_args)
        agents.set_default_openai_client(openai_client)
        
        # Set tracing based on configuration
        model_settings = get_model_settings(
            config, 
            provider, 
            config.get("default_models", {}).get(provider, "")
        )
        
        # Apply tracing settings
        if model_settings.tracing == ModelTracing.DISABLED:
            agents.set_tracing_disabled()
        
        return True
    except Exception as e:
        print(f"Error setting up global OpenAI client: {str(e)}")
        return False

# Create a custom model provider (Method 2)
def create_custom_model_provider(config: Dict) -> Any:
    """
    Create a custom model provider for use with Runner.run.
    
    This creates a provider implementation that can be passed to Runner.run
    to handle all models in that run.
    
    Args:
        config: Configuration dictionary with provider settings
        
    Returns:
        ModelProvider: A custom model provider implementation
    """
    try:
        # Get current provider info
        provider = config.get("default_provider", "OpenAI")
        provider_info = SUPPORTED_PROVIDERS.get(provider, {})
        
        # For official providers, we don't need a custom implementation
        if provider_info.get("provider_type") == "official":
            return None
            
        # Get API key and base URL
        api_key = os.environ.get(provider_info.get("api_key_env", ""))
        api_base = os.environ.get(provider_info.get("api_base_env", ""))
        
        if not api_key:
            api_key = config.get("api_keys", {}).get(provider)
            
        if not api_base:
            api_base = config.get("provider_settings", {}).get(provider, {}).get("api_base")
        
        # For Azure OpenAI, ensure we have deployment mappings
        if provider == "Azure OpenAI":
            # Get Azure-specific configuration
            deployments = config.get("provider_settings", {}).get(provider, {}).get("deployments", {})
            api_version = config.get("provider_settings", {}).get(provider, {}).get("api_version", "2023-12-01-preview")
            
            # Create a custom AzureOpenAI model provider
            from openai import AsyncAzureOpenAI
            
            # This is a simplified example - actual implementation would be more complex
            class AzureOpenAIModelProvider(ModelProvider):
                def __init__(self, api_key, api_base, api_version, deployments):
                    self.api_key = api_key
                    self.api_base = api_base
                    self.api_version = api_version
                    self.deployments = deployments
                    self.client = AsyncAzureOpenAI(
                        api_key=api_key,
                        azure_endpoint=api_base,
                        api_version=api_version
                    )
                
                def get_model(self, model_name: Optional[str]) -> Model:
                    # Look up deployment name for this model
                    deployment = self.deployments.get(model_name, model_name)
                    return AzureOpenAIModel(self.client, deployment)
            
            # This would be a complete Model implementation
            class AzureOpenAIModel(Model):
                def __init__(self, client, deployment):
                    self.client = client
                    self.deployment = deployment
                
                async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing):
                    # Implementation would call self.client.chat.completions.create with proper deployment
                    raise NotImplementedError("This is a simplified example - actual implementation required")
                
                async def stream_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing):
                    # Implementation would call self.client.chat.completions.create with stream=True
                    raise NotImplementedError("This is a simplified example - actual implementation required")
            
            return AzureOpenAIModelProvider(api_key, api_base, api_version, deployments)
        
        # For other custom OpenAI-compatible providers
        if provider_info.get("client_type") == "openai_compatible":
            from openai import AsyncOpenAI
            
            # This is a simplified example
            class OpenAICompatibleProvider(ModelProvider):
                def __init__(self, api_key, api_base):
                    self.api_key = api_key
                    self.api_base = api_base
                    self.client = AsyncOpenAI(
                        api_key=api_key,
                        base_url=api_base
                    )
                
                def get_model(self, model_name: Optional[str]) -> Model:
                    return OpenAICompatibleModel(self.client, model_name)
            
            # This would be a complete Model implementation
            class OpenAICompatibleModel(Model):
                def __init__(self, client, model_name):
                    self.client = client
                    self.model_name = model_name
                
                async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing):
                    # Implementation would call self.client.chat.completions.create
                    raise NotImplementedError("This is a simplified example - actual implementation required")
                
                async def stream_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing):
                    # Implementation would call self.client.chat.completions.create with stream=True
                    raise NotImplementedError("This is a simplified example - actual implementation required")
            
            return OpenAICompatibleProvider(api_key, api_base)
        
        # If we got here, we don't have a suitable implementation
        return None
    except Exception as e:
        print(f"Error creating custom model provider: {str(e)}")
        return None

def create_agent_with_model(agent_name: str, tools: List, instructions: str) -> Any:
    """
    Create an agent using the configured default model (Method 3).
    
    This function creates an agent with the selected model and configured settings.
    It's the most direct way to specify the model for a specific agent instance.
    
    Args:
        agent_name: Name of the agent
        tools: List of tools available to the agent
        instructions: Instructions for the agent
        
    Returns:
        Agent: The created agent object
    """
    from agents import Agent, RunConfig, ModelSettings as AgentModelSettings
    
    # Get configuration
    if "model_config" not in st.session_state:
        st.session_state.model_config = initialize_model_config()
    
    config = st.session_state.model_config
    
    # Get provider, model and settings
    provider, model_id, settings = get_default_model()
    
    # Convert our ModelSettings to the Agent SDK's ModelSettings
    agent_settings = AgentModelSettings(
        temperature=settings.temperature,
        top_p=settings.top_p,
        frequency_penalty=settings.frequency_penalty,
        presence_penalty=settings.presence_penalty,
        tool_choice=settings.tool_choice,
        parallel_tool_calls=settings.parallel_tool_calls,
        truncation=settings.truncation,
        max_tokens=settings.max_tokens,
        store=settings.store
    )
    
    # Note: Tracing setting can't be directly passed through ModelSettings
    # It would typically be handled at the runner level
    
    # Get provider info
    provider_info = SUPPORTED_PROVIDERS.get(provider, {})
    
    # For custom providers with special model name handling
    if provider_info.get("provider_type") == "custom":
        # For Azure OpenAI, we need to use the deployment name
        if provider == "Azure OpenAI":
            deployments = config.get("provider_settings", {}).get(provider, {}).get("deployments", {})
            deployment_name = deployments.get(model_id, model_id)
            
            # Using method 3 (Agent.model) for custom model name
            # Create the agent with the deployment name
            agent = Agent(
                model=deployment_name,
                tools=tools,
                instructions=instructions,
                name=agent_name,
                model_settings=agent_settings
            )
            return agent
    
    # Regular case - use the model ID directly
    agent = Agent(
        model=model_id,
        tools=tools,
        instructions=instructions,
        name=agent_name,
        model_settings=agent_settings
    )
    
    return agent

if __name__ == "__main__":
    render_model_management()