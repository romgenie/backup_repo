"""
Reliability processor for reducing hallucinations in agent responses.

This module provides the ReliabilityProcessor class, which orchestrates the
validation process for agent responses, detecting and removing potentially
hallucinated content.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from agents import Agent, Runner, ItemHelpers

from .models import ValidationPoint, ValidationResult, SourceReliability
from .prompts import (
    URL_VALIDATION_PROMPT,
    NUMBER_VALIDATION_PROMPT,
    INFORMATION_VALIDATION_PROMPT,
    CODE_VALIDATION_PROMPT,
    EDITOR_TASK_PROMPT,
)
from .validators import (
    analyze_content,
    should_validate,
    find_urls_in_text,
    contains_urls,
    contains_numbers,
    contains_code,
)
from .context_manager import ContextManager
from .subtask_manager import SubtaskManager
from .caching import CacheManager, save_validation_result, get_validation_result
from .error_handling import (
    ValidationError,
    ProcessingError,
    TransformationError,
    RetryConfiguration,
    retry_async,
    log_error,
)
from .language import LanguageTransformer, transform_to_language
from .config import ReliabilityConfig, ConfigurationManager


T = TypeVar('T')


class ReliabilityProcessor:
    """
    Processor for validating agent responses and removing hallucinations.
    
    The ReliabilityProcessor orchestrates validator agents to check different
    aspects of agent responses (URLs, numbers, information, code) and removes
    or replaces suspicious content.
    """
    
    def __init__(
        self,
        config: Optional[ReliabilityConfig] = None,
        config_id: Optional[str] = None,
    ):
        """
        Initialize the ReliabilityProcessor.
        
        Args:
            config: Configuration for reliability processing
            config_id: ID of the configuration to use
        """
        self.config_manager = ConfigurationManager()
        self.context_manager = ContextManager()
        self.subtask_manager = SubtaskManager()
        self.cache_manager = CacheManager()
        
        if config:
            if config_id:
                self.config_manager.set_config(config, config_id)
                self.config_id = config_id
            else:
                self.config_manager.set_default_config(config)
                self.config_id = None
        else:
            self.config_id = config_id
        
        self.config = self.config_manager.get_config(self.config_id)
    
    def _get_config(self) -> ReliabilityConfig:
        """
        Get the current configuration.
        
        Returns:
            Current ReliabilityConfig
        """
        return self.config_manager.get_config(self.config_id)
    
    @staticmethod
    async def process_result(
        result: T,
        context_items: List[Any],
        task_description: str,
        reliability_level: int = 10,
        model: Optional[str] = None,
        target_language: Optional[str] = None,
        config_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> T:
        """
        Process a result to check for hallucinations and clean up suspicious content.
        
        Args:
            result: The result from an agent run
            context_items: Context items used during the agent run
            task_description: The original task description
            reliability_level: Level of reliability checking (0-10)
            model: Model to use for validation agents
            target_language: Target language for transformation
            config_id: ID of the configuration to use
            task_id: ID of the task (for subtask management)
            
        Returns:
            Processed result with suspicious content removed
        """
        # Create processor instance
        processor = ReliabilityProcessor(config_id=config_id)
        
        # Get configuration
        config = processor._get_config()
        
        # Override reliability level if provided
        if reliability_level is not None:
            actual_reliability_level = reliability_level
        else:
            actual_reliability_level = config.overall_reliability_level
        
        # Skip reliability processing if level is 0
        if actual_reliability_level == 0:
            return result
        
        # Get context manager and cache manager
        context_manager = processor.context_manager
        cache_manager = processor.cache_manager
        
        # Try to get cached result
        if config.cache.enabled:
            cache_key = cache_manager.generate_cache_key({
                "result": result,
                "context": context_items,
                "task_description": task_description,
                "reliability_level": actual_reliability_level,
            })
            
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Add context items to context manager
        for i, item in enumerate(context_items):
            if isinstance(item, dict) and "role" in item and "content" in item:
                context_manager.add_conversation_item(item["role"], item["content"])
        
        # Add task description to context manager
        context_manager.add_conversation_item("system", task_description)
        
        # Check if this is part of a subtask
        subtask_manager = processor.subtask_manager
        is_subtask = task_id is not None
        
        if is_subtask:
            # Get subtask information
            subtask = subtask_manager.get_subtask(task_id)
            
            # If not a subtask that should be validated, return result as is
            if subtask and not subtask_manager.should_validate(task_id):
                return result
            
            # If it is a subtask that should be validated, get the appropriate level
            if subtask:
                actual_reliability_level = subtask_manager.get_validation_level(task_id)
        
        # Only perform full validation at appropriate level
        if actual_reliability_level > 0:
            try:
                # Start processing
                return await processor._process_result_internal(
                    result,
                    context_items,
                    task_description,
                    actual_reliability_level,
                    model,
                    target_language,
                    task_id,
                )
            except Exception as e:
                # Log error
                log_error(e, {
                    "result_type": type(result).__name__,
                    "context_items_count": len(context_items),
                    "task_description": task_description,
                    "reliability_level": actual_reliability_level,
                })
                
                # If configured to fail on validation error, re-raise
                if config.error.fail_on_validation_error:
                    raise
                
                # Otherwise, return original result
                return result
        
        # If no validation performed, return original result
        return result
    
    async def _process_result_internal(
        self,
        result: T,
        context_items: List[Any],
        task_description: str,
        reliability_level: int,
        model: Optional[str] = None,
        target_language: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> T:
        """
        Internal method for processing a result.
        
        Args:
            result: The result from an agent run
            context_items: Context items used during the agent run
            task_description: The original task description
            reliability_level: Level of reliability checking (0-10)
            model: Model to use for validation agents
            target_language: Target language for transformation
            task_id: ID of the task (for subtask management)
            
        Returns:
            Processed result with suspicious content removed
        """
        # Get configuration
        config = self._get_config()
        
        # Use configured model if not provided
        if model is None:
            model = config.default_model
        
        # Extract context and prepare for validation
        context_strings = []
        context_strings.append(f"Given Task: {task_description}")
        
        # Process context items
        for item in context_items:
            if isinstance(item, dict) and "role" in item and "content" in item:
                role = item["role"]
                content = item["content"]
                context_strings.append(f"\n\nContext From {role.capitalize()} (Trusted Source): ```{content}```")
            else:
                text = str(item)
                context_strings.append(f"\n\nContext That Came From User (Trusted Source): ```{text}```")
        
        # Add additional context from context manager if enabled
        if config.context.enabled:
            recent_conversation = self.context_manager.get_recent_conversation(
                config.context.max_conversation_history
            )
            
            for item in recent_conversation:
                if item.get("role") and item.get("content"):
                    role = item["role"]
                    content = item["content"]
                    context_strings.append(f"\n\nPrevious Conversation From {role.capitalize()} (Trusted Source): ```{content}```")
        
        # Add additional context from subtask manager if enabled
        if config.subtask.enabled and task_id:
            subtask_context = self.subtask_manager.get_context_for_subtask(task_id)
            
            for item in subtask_context:
                if isinstance(item, dict) and "result" in item:
                    context_strings.append(f"\n\nPrevious Subtask Result (Trusted Source): ```{item['result']}```")
        
        # Add the current result to context
        result_str = str(result)
        context_strings.append(f"\nCurrent AI Response (Untrusted Source, last AI response that we are checking now): {result_str}")
        
        # Create validation result structure
        validation_result = ValidationResult(
            url_validation=ValidationPoint(is_suspicious=False, feedback="", suspicious_points=[]),
            number_validation=ValidationPoint(is_suspicious=False, feedback="", suspicious_points=[]),
            information_validation=ValidationPoint(is_suspicious=False, feedback="", suspicious_points=[]),
            code_validation=ValidationPoint(is_suspicious=False, feedback="", suspicious_points=[]),
        )
        
        # Analyze content to determine which validators to run
        content_analysis = analyze_content("\n".join(context_strings))
        
        # Create validation agents and tasks
        validation_agents = {}
        validation_tasks = []
        validation_types = []
        
        # URL validation (skip if not present or disabled)
        if (config.validation.url_validator.enabled and
                (config.validation.run_all_validators or
                 (config.validation.url_validator.skip_if_not_present and
                  content_analysis["url_analysis"]["has_urls"]))):
            
            # Try to get cached validation result
            cached_validation = None
            if config.cache.enabled and config.cache.cache_validation_results:
                cached_validation = get_validation_result(
                    result_str,
                    {"validation_type": "url", "context": context_strings}
                )
            
            if cached_validation:
                validation_result.url_validation = cached_validation
            else:
                url_validator = Agent(
                    name="URL Validator",
                    instructions=URL_VALIDATION_PROMPT + (
                        config.validation.url_validator.additional_instructions or ""
                    ),
                    output_type=ValidationPoint,
                )
                validation_agents["url_validation"] = url_validator
                validation_types.append("url_validation")
                validation_tasks.append("\n".join(context_strings))
        else:
            # Set default validation point for URLs if not validating
            validation_result.url_validation = ValidationPoint(
                is_suspicious=False,
                feedback="URL validation skipped",
                suspicious_points=[],
                source_reliability=SourceReliability.UNKNOWN,
                verification_method="skipped",
                confidence_score=1.0
            )
        
        # Number validation (skip if not present or disabled)
        if (config.validation.number_validator.enabled and
                (config.validation.run_all_validators or
                 (config.validation.number_validator.skip_if_not_present and
                  content_analysis["number_analysis"]["has_numbers"]))):
            
            # Try to get cached validation result
            cached_validation = None
            if config.cache.enabled and config.cache.cache_validation_results:
                cached_validation = get_validation_result(
                    result_str,
                    {"validation_type": "number", "context": context_strings}
                )
            
            if cached_validation:
                validation_result.number_validation = cached_validation
            else:
                number_validator = Agent(
                    name="Number Validator",
                    instructions=NUMBER_VALIDATION_PROMPT + (
                        config.validation.number_validator.additional_instructions or ""
                    ),
                    output_type=ValidationPoint,
                )
                validation_agents["number_validation"] = number_validator
                validation_types.append("number_validation")
                validation_tasks.append("\n".join(context_strings))
        else:
            # Set default validation point for numbers if not validating
            validation_result.number_validation = ValidationPoint(
                is_suspicious=False,
                feedback="Number validation skipped",
                suspicious_points=[],
                source_reliability=SourceReliability.UNKNOWN,
                verification_method="skipped",
                confidence_score=1.0
            )
        
        # Information validation (always run if enabled)
        if config.validation.information_validator.enabled:
            # Try to get cached validation result
            cached_validation = None
            if config.cache.enabled and config.cache.cache_validation_results:
                cached_validation = get_validation_result(
                    result_str,
                    {"validation_type": "information", "context": context_strings}
                )
            
            if cached_validation:
                validation_result.information_validation = cached_validation
            else:
                information_validator = Agent(
                    name="Information Validator",
                    instructions=INFORMATION_VALIDATION_PROMPT + (
                        config.validation.information_validator.additional_instructions or ""
                    ),
                    output_type=ValidationPoint,
                )
                validation_agents["information_validation"] = information_validator
                validation_types.append("information_validation")
                validation_tasks.append("\n".join(context_strings))
        else:
            # Set default validation point for information if not validating
            validation_result.information_validation = ValidationPoint(
                is_suspicious=False,
                feedback="Information validation skipped",
                suspicious_points=[],
                source_reliability=SourceReliability.UNKNOWN,
                verification_method="skipped",
                confidence_score=1.0
            )
        
        # Code validation (skip if not present or disabled)
        if (config.validation.code_validator.enabled and
                (config.validation.run_all_validators or
                 (config.validation.code_validator.skip_if_not_present and
                  content_analysis["code_analysis"]["has_code"]))):
            
            # Try to get cached validation result
            cached_validation = None
            if config.cache.enabled and config.cache.cache_validation_results:
                cached_validation = get_validation_result(
                    result_str,
                    {"validation_type": "code", "context": context_strings}
                )
            
            if cached_validation:
                validation_result.code_validation = cached_validation
            else:
                code_validator = Agent(
                    name="Code Validator",
                    instructions=CODE_VALIDATION_PROMPT + (
                        config.validation.code_validator.additional_instructions or ""
                    ),
                    output_type=ValidationPoint,
                )
                validation_agents["code_validation"] = code_validator
                validation_types.append("code_validation")
                validation_tasks.append("\n".join(context_strings))
        else:
            # Set default validation point for code if not validating
            validation_result.code_validation = ValidationPoint(
                is_suspicious=False,
                feedback="Code validation skipped",
                suspicious_points=[],
                source_reliability=SourceReliability.UNKNOWN,
                verification_method="skipped",
                confidence_score=1.0
            )
        
        # Run all validators in parallel
        if validation_tasks and validation_types:
            try:
                # Set timeout for validation
                validation_timeout = config.validation.validation_timeout
                
                if config.validation.parallel_validation:
                    # Run validators in parallel
                    validation_coroutines = []
                    for i, validation_type in enumerate(validation_types):
                        validation_coroutines.append(
                            asyncio.wait_for(
                                Runner.run(
                                    validation_agents[validation_type],
                                    validation_tasks[i],
                                    model=model,
                                ),
                                timeout=validation_timeout
                            )
                        )
                    
                    # Wait for all validation tasks to complete
                    validation_results = await asyncio.gather(
                        *validation_coroutines,
                        return_exceptions=True
                    )
                    
                    # Process validation results
                    for i, validation_type in enumerate(validation_types):
                        result_output = validation_results[i]
                        
                        if isinstance(result_output, Exception):
                            # Log error and set default validation point
                            log_error(result_output, {
                                "validation_type": validation_type,
                                "task_description": task_description,
                            })
                            
                            # Set default validation point
                            setattr(validation_result, validation_type, ValidationPoint(
                                is_suspicious=False,
                                feedback=f"Validation failed: {str(result_output)}",
                                suspicious_points=[],
                                source_reliability=SourceReliability.UNKNOWN,
                                verification_method="failed",
                                confidence_score=0.5
                            ))
                        else:
                            # Set validation result
                            validator_result = result_output.final_output
                            setattr(validation_result, validation_type, validator_result)
                            
                            # Cache validation result if enabled
                            if config.cache.enabled and config.cache.cache_validation_results:
                                save_validation_result(
                                    validator_result,
                                    {"validation_type": validation_type, "context": context_strings},
                                    config.cache.expiry_time
                                )
                else:
                    # Run validators sequentially
                    for i, validation_type in enumerate(validation_types):
                        try:
                            # Run validator
                            result_output = await asyncio.wait_for(
                                Runner.run(
                                    validation_agents[validation_type],
                                    validation_tasks[i],
                                    model=model,
                                ),
                                timeout=validation_timeout
                            )
                            
                            # Set validation result
                            validator_result = result_output.final_output
                            setattr(validation_result, validation_type, validator_result)
                            
                            # Cache validation result if enabled
                            if config.cache.enabled and config.cache.cache_validation_results:
                                save_validation_result(
                                    validator_result,
                                    {"validation_type": validation_type, "context": context_strings},
                                    config.cache.expiry_time
                                )
                        except Exception as e:
                            # Log error and set default validation point
                            log_error(e, {
                                "validation_type": validation_type,
                                "task_description": task_description,
                            })
                            
                            # Set default validation point
                            setattr(validation_result, validation_type, ValidationPoint(
                                is_suspicious=False,
                                feedback=f"Validation failed: {str(e)}",
                                suspicious_points=[],
                                source_reliability=SourceReliability.UNKNOWN,
                                verification_method="failed",
                                confidence_score=0.5
                            ))
            except Exception as e:
                # Log error
                log_error(e, {
                    "validation_task": "running_validators",
                    "task_description": task_description,
                })
                
                # Set default validation points for all types
                for validation_type in validation_types:
                    setattr(validation_result, validation_type, ValidationPoint(
                        is_suspicious=False,
                        feedback=f"Validation failed: {str(e)}",
                        suspicious_points=[],
                        source_reliability=SourceReliability.UNKNOWN,
                        verification_method="failed",
                        confidence_score=0.5
                    ))
        
        # Calculate overall suspicion
        validation_result.calculate_suspicion()
        
        # Store validation result in context manager
        validation_id = f"validation_{hash(result_str)}"
        self.context_manager.add_validation_result(validation_id, validation_result)
        
        # If this is a subtask, store validation result in subtask manager
        if task_id:
            self.subtask_manager.set_validation_result(task_id, validation_result)
        
        # If suspicious content found and editor is enabled, run editor agent
        processed_result = result
        
        if validation_result.any_suspicion and config.editor.enabled:
            try:
                # Create editor agent
                editor_agent = Agent(
                    name="Information Editor Agent",
                    instructions=EDITOR_TASK_PROMPT.format(
                        validation_feedback=validation_result.overall_feedback
                    ) + f"\nOLD AI Response: {result_str}" + (
                        config.editor.additional_instructions or ""
                    ),
                    output_type=type(result) if not isinstance(result, str) else str,
                )
                
                # Create editor context
                editor_context = [
                    {"role": "system", "content": "You are a content editor that removes suspicious information."},
                    {"role": "user", "content": "\n".join(context_strings)}
                ]
                
                # Run editor agent
                retry_config = RetryConfiguration(
                    max_retries=config.error.max_retries,
                    retry_delay=config.error.retry_delay,
                    backoff_factor=config.error.backoff_factor,
                )
                
                editor_result = await retry_async(
                    Runner.run,
                    args=[editor_agent, editor_context, model],
                    config=retry_config
                )
                
                # Extract result
                processed_result = editor_result.final_output
            except Exception as e:
                # Log error
                log_error(e, {
                    "validation_task": "editor_agent",
                    "task_description": task_description,
                })
                
                # Use original result if editor fails
                processed_result = result
        
        # Apply language transformation if enabled and target language specified
        if config.language.enabled and target_language:
            try:
                # Transform result to target language
                language_transformer = LanguageTransformer(target_language)
                processed_result = await language_transformer.transform(processed_result, model)
            except Exception as e:
                # Log error
                log_error(e, {
                    "validation_task": "language_transformation",
                    "task_description": task_description,
                    "target_language": target_language,
                })
                
                # Use unmodified result if transformation fails
                # (already processed by editor if suspicious content was found)
        
        # Cache the processed result if caching is enabled
        if config.cache.enabled:
            cache_key = self.cache_manager.generate_cache_key({
                "result": result,
                "context": context_items,
                "task_description": task_description,
                "reliability_level": reliability_level,
            })
            
            self.cache_manager.set(
                cache_key,
                processed_result,
                config.cache.expiry_time
            )
        
        # Return the processed result
        return processed_result
