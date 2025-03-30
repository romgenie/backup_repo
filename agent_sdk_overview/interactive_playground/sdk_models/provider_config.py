"""
Provider configuration manager for the SDK Playground.
Handles API keys, endpoints, and other provider-specific settings.
"""
import os
import streamlit as st
from typing import Dict, Optional
from .model_definitions import ModelDefinition, PROVIDERS


class ProviderConfig:
    """Manager for model provider configurations."""
    
    @staticmethod
    def load_from_env():
        """Load provider configurations from environment variables."""
        configs = {
            "openai": {
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
                "base_url": os.environ.get("OPENAI_BASE_URL", ""),
            },
            "azure_openai": {
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
                "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
            },
            "anthropic": {
                "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            },
            "custom": {
                "api_key": "",
                "base_url": "",
            },
        }
        
        # Store in session state if not already there
        if "provider_configs" not in st.session_state:
            st.session_state.provider_configs = configs
        
        return configs
    
    @staticmethod
    def get_config(provider_id: str) -> Dict:
        """Get configuration for a specific provider."""
        if "provider_configs" not in st.session_state:
            ProviderConfig.load_from_env()
            
        return st.session_state.provider_configs.get(provider_id, {})
    
    @staticmethod
    def validate_config(model: ModelDefinition, config: Dict) -> bool:
        """
        Validate that the configuration is complete for the given model.
        Returns True if valid, False otherwise.
        """
        provider = model.provider
        
        # Check for required API key
        if provider.requires_key and not config.get("api_key"):
            return False
            
        # Check for required endpoint
        if provider.requires_endpoint and not config.get("endpoint"):
            return False
            
        return True
    
    @staticmethod
    def get_provider_client(model: ModelDefinition, config: Dict):
        """
        Get an appropriate client for the provider.
        This function would integrate with the actual SDK client creation.
        """
        provider_id = model.provider.id
        
        # This is a placeholder - in a real implementation, this would
        # create and return the actual SDK client
        if provider_id == "openai":
            # Example: return OpenAIProvider(api_key=config["api_key"])
            return {
                "type": "openai",
                "api_key": config.get("api_key"),
                "base_url": config.get("base_url") or None,
                "model_id": model.id,
            }
        elif provider_id == "azure_openai":
            # Example: return AzureOpenAIProvider(...)
            return {
                "type": "azure_openai",
                "api_key": config.get("api_key"),
                "endpoint": config.get("endpoint"),
                "api_version": config.get("api_version"),
                "model_id": model.id,
            }
        elif provider_id == "anthropic":
            # Example: return AnthropicProvider(api_key=config["api_key"])
            return {
                "type": "anthropic",
                "api_key": config.get("api_key"),
                "model_id": model.id,
            }
        else:
            # Generic custom provider
            return {
                "type": "custom",
                "api_key": config.get("api_key"),
                "base_url": config.get("base_url"),
                "model_id": model.id,
            }