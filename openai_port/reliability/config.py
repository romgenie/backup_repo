"""
Configuration system for reliability processing.

This module provides a flexible configuration system for customizing
reliability processing behavior.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ValidatorConfig(BaseModel):
    """Configuration for a specific validator."""
    
    enabled: bool = True
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    validation_level: int = Field(default=10, ge=0, le=10)
    skip_if_not_present: bool = True
    additional_instructions: Optional[str] = None


class ValidationConfig(BaseModel):
    """Configuration for validation processing."""
    
    url_validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    number_validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    information_validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    code_validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    run_all_validators: bool = False
    parallel_validation: bool = True
    validation_timeout: float = 30.0


class EditorConfig(BaseModel):
    """Configuration for the editor agent."""
    
    enabled: bool = True
    additional_instructions: Optional[str] = None
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    handle_partial_validation: bool = True
    preserve_structure: bool = True


class CacheConfig(BaseModel):
    """Configuration for caching."""
    
    enabled: bool = True
    expiry_time: int = 3600  # 1 hour
    max_cache_size: int = 1000
    cache_validation_results: bool = True
    invalidate_on_context_change: bool = True


class ContextConfig(BaseModel):
    """Configuration for context management."""
    
    enabled: bool = True
    max_conversation_history: int = 100
    expiry_time: int = 3600  # 1 hour
    inherit_trusted_sources: bool = True
    inherit_suspicious_points: bool = True


class SubtaskConfig(BaseModel):
    """Configuration for subtask management."""
    
    enabled: bool = True
    validate_final_subtask_only: bool = True
    pass_context_between_subtasks: bool = True
    intermediate_validation_level: int = Field(default=5, ge=0, le=10)


class LanguageConfig(BaseModel):
    """Configuration for language transformation."""
    
    enabled: bool = True
    preserve_formatting: bool = True
    translate_suspicious_content: bool = False


class ErrorConfig(BaseModel):
    """Configuration for error handling."""
    
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    log_errors: bool = True
    fail_on_validation_error: bool = False


class ReliabilityConfig(BaseModel):
    """
    Comprehensive configuration for reliability processing.
    
    This class provides a flexible configuration system for customizing
    all aspects of reliability processing.
    """
    
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    editor: EditorConfig = Field(default_factory=EditorConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    subtask: SubtaskConfig = Field(default_factory=SubtaskConfig)
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    error: ErrorConfig = Field(default_factory=ErrorConfig)
    
    # General settings
    overall_reliability_level: int = Field(default=10, ge=0, le=10)
    default_model: Optional[str] = None
    debug_mode: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ReliabilityConfig":
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            ReliabilityConfig instance
        """
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ReliabilityConfig":
        """
        Create a configuration from a JSON string.
        
        Args:
            json_str: JSON string with configuration values
            
        Returns:
            ReliabilityConfig instance
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, file_path: str) -> "ReliabilityConfig":
        """
        Create a configuration from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            ReliabilityConfig instance
        """
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return self.model_dump()
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert configuration to a JSON string.
        
        Args:
            indent: Number of spaces for indentation
            
        Returns:
            JSON string representation of the configuration
        """
        return self.model_dump_json(indent=indent)
    
    def save_to_file(self, file_path: str, indent: int = 2):
        """
        Save configuration to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
            indent: Number of spaces for indentation
        """
        with open(file_path, "w") as f:
            f.write(self.to_json(indent=indent))
    
    def update(self, config_dict: Dict[str, Any]) -> "ReliabilityConfig":
        """
        Update configuration with values from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values to update
            
        Returns:
            Updated ReliabilityConfig instance
        """
        # Create a dictionary representation of the current config
        current_config = self.to_dict()
        
        # Update with new values
        for key, value in config_dict.items():
            if isinstance(value, dict) and key in current_config and isinstance(current_config[key], dict):
                # If both are dictionaries, recursively update
                current_config[key].update(value)
            else:
                # Otherwise, replace the value
                current_config[key] = value
        
        # Create a new config with the updated values
        return self.from_dict(current_config)


class ConfigurationManager:
    """Manager for reliability configurations."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the configuration manager state."""
        self.configs = {}
        self.default_config = ReliabilityConfig()
    
    def get_config(self, config_id: Optional[str] = None) -> ReliabilityConfig:
        """
        Get a configuration by ID, or the default configuration if not found.
        
        Args:
            config_id: ID of the configuration to get
            
        Returns:
            ReliabilityConfig instance
        """
        if config_id is None or config_id not in self.configs:
            return self.default_config
        
        return self.configs[config_id]
    
    def set_config(self, config: ReliabilityConfig, config_id: str):
        """
        Set a configuration with the given ID.
        
        Args:
            config: Configuration to set
            config_id: ID to associate with the configuration
        """
        self.configs[config_id] = config
    
    def set_default_config(self, config: ReliabilityConfig):
        """
        Set the default configuration.
        
        Args:
            config: Configuration to set as default
        """
        self.default_config = config
    
    def load_from_file(self, file_path: str, config_id: Optional[str] = None):
        """
        Load a configuration from a file.
        
        Args:
            file_path: Path to the configuration file
            config_id: ID to associate with the configuration
        """
        config = ReliabilityConfig.from_file(file_path)
        
        if config_id is None:
            self.default_config = config
        else:
            self.configs[config_id] = config
    
    def save_to_file(self, file_path: str, config_id: Optional[str] = None):
        """
        Save a configuration to a file.
        
        Args:
            file_path: Path to save the configuration to
            config_id: ID of the configuration to save
        """
        config = self.get_config(config_id)
        config.save_to_file(file_path)


# Predefined configurations for common use cases
def get_default_config() -> ReliabilityConfig:
    """
    Get the default configuration.
    
    Returns:
        Default ReliabilityConfig
    """
    return ReliabilityConfig()


def get_high_reliability_config() -> ReliabilityConfig:
    """
    Get a configuration with high reliability settings.
    
    Returns:
        High reliability ReliabilityConfig
    """
    config = ReliabilityConfig()
    
    # Set validation confidence thresholds higher
    config.validation.url_validator.confidence_threshold = 0.9
    config.validation.number_validator.confidence_threshold = 0.9
    config.validation.information_validator.confidence_threshold = 0.9
    config.validation.code_validator.confidence_threshold = 0.9
    
    # Enable all validators
    config.validation.run_all_validators = True
    
    # Set editor confidence threshold higher
    config.editor.confidence_threshold = 0.8
    
    # Increase retries
    config.error.max_retries = 5
    
    return config


def get_fast_config() -> ReliabilityConfig:
    """
    Get a configuration optimized for speed.
    
    Returns:
        Fast ReliabilityConfig
    """
    config = ReliabilityConfig()
    
    # Lower validation confidence thresholds
    config.validation.url_validator.confidence_threshold = 0.5
    config.validation.number_validator.confidence_threshold = 0.5
    config.validation.information_validator.confidence_threshold = 0.5
    config.validation.code_validator.confidence_threshold = 0.5
    
    # Skip validators if content not present
    config.validation.url_validator.skip_if_not_present = True
    config.validation.number_validator.skip_if_not_present = True
    config.validation.code_validator.skip_if_not_present = True
    
    # Enable caching
    config.cache.enabled = True
    
    # Reduce retries
    config.error.max_retries = 1
    
    return config


def get_balanced_config() -> ReliabilityConfig:
    """
    Get a configuration with balanced reliability and speed.
    
    Returns:
        Balanced ReliabilityConfig
    """
    config = ReliabilityConfig()
    
    # Set moderate validation confidence thresholds
    config.validation.url_validator.confidence_threshold = 0.7
    config.validation.number_validator.confidence_threshold = 0.7
    config.validation.information_validator.confidence_threshold = 0.7
    config.validation.code_validator.confidence_threshold = 0.7
    
    # Skip validators for code if not present
    config.validation.code_validator.skip_if_not_present = True
    
    # Set moderate editor confidence threshold
    config.editor.confidence_threshold = 0.6
    
    # Enable caching with moderate expiry
    config.cache.enabled = True
    config.cache.expiry_time = 1800  # 30 minutes
    
    return config
