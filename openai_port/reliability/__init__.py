"""
Reliability module for OpenAI Agents SDK.

This module provides functionality to reduce hallucinations in agent responses
by validating outputs against their context and removing suspicious content.
"""

from .models import ValidationPoint, ValidationResult, SourceReliability
from .processor import ReliabilityProcessor
from .runner import ReliableRunner
from .reliable_agent import ReliableAgent
from .context_manager import ContextManager
from .subtask_manager import SubtaskManager
from .caching import CacheManager, save_to_cache_with_expiry, get_from_cache_with_expiry
from .error_handling import (
    ReliabilityError, ValidationError, ProcessingError, TransformationError,
    RetryConfiguration, retry_async, retry_sync, ErrorLogger, log_error, get_errors
)
from .config import (
    ReliabilityConfig, ConfigurationManager, ValidatorConfig, ValidationConfig,
    EditorConfig, CacheConfig, ContextConfig, SubtaskConfig, LanguageConfig, ErrorConfig,
    get_default_config, get_high_reliability_config, get_fast_config, get_balanced_config
)
from .language import LanguageTransformer, transform_to_language, SupportedLanguages
from .validators import (
    analyze_content, should_validate, find_urls_in_text, contains_urls,
    extract_numbers_from_text, contains_numbers, contains_code
)

__all__ = [
    # Core classes
    "ReliabilityProcessor",
    "ReliableRunner",
    "ReliableAgent",
    
    # Models
    "ValidationPoint",
    "ValidationResult",
    "SourceReliability",
    
    # Managers
    "ContextManager",
    "SubtaskManager",
    "CacheManager",
    
    # Configuration
    "ReliabilityConfig",
    "ConfigurationManager",
    "ValidatorConfig",
    "ValidationConfig",
    "EditorConfig",
    "CacheConfig", 
    "ContextConfig",
    "SubtaskConfig",
    "LanguageConfig",
    "ErrorConfig",
    "get_default_config",
    "get_high_reliability_config",
    "get_fast_config",
    "get_balanced_config",
    
    # Error handling
    "ReliabilityError",
    "ValidationError",
    "ProcessingError", 
    "TransformationError",
    "RetryConfiguration",
    "retry_async",
    "retry_sync",
    "ErrorLogger",
    "log_error",
    "get_errors",
    
    # Caching
    "save_to_cache_with_expiry",
    "get_from_cache_with_expiry",
    
    # Language transformation
    "LanguageTransformer",
    "transform_to_language",
    "SupportedLanguages",
    
    # Validators
    "analyze_content",
    "should_validate",
    "find_urls_in_text",
    "contains_urls",
    "extract_numbers_from_text",
    "contains_numbers",
    "contains_code",
]
