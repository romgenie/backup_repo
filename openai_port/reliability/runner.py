"""
Reliable runner for agents with hallucination prevention.

This module provides the ReliableRunner class, which wraps the standard Runner
with additional reliability processing to reduce hallucinations.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from agents import Agent, Runner, RunResult, TResponseInputItem

from .processor import ReliabilityProcessor
from .context_manager import ContextManager
from .subtask_manager import SubtaskManager
from .config import ReliabilityConfig, ConfigurationManager, get_default_config
from .error_handling import retry_async, RetryConfiguration, log_error

T = TypeVar('T')


class ReliableRunner:
    """
    A runner for agents with built-in reliability processing.
    
    ReliableRunner wraps the standard Runner with additional functionality
    to validate agent responses and remove potentially hallucinated content.
    """
    
    @staticmethod
    async def run(
        agent: Agent[T],
        input: Union[str, List[TResponseInputItem]],
        reliability_level: Optional[int] = None,
        model: Optional[str] = None,
        target_language: Optional[str] = None,
        config: Optional[ReliabilityConfig] = None,
        config_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> RunResult:
        """
        Run an agent with reliability processing.
        
        Args:
            agent: The agent to run
            input: The input to the agent (string or message list)
            reliability_level: Level of reliability checking (0-10)
            model: Model to use for agent and validators
            target_language: Target language for response transformation
            config: Configuration for reliability processing
            config_id: ID of configuration to use
            task_id: ID of task (for subtask management)
            
        Returns:
            The run result with hallucinations removed
        """
        # Get configuration
        config_manager = ConfigurationManager()
        if config:
            if config_id:
                config_manager.set_config(config, config_id)
            else:
                config_manager.set_default_config(config)
        
        actual_config = config_manager.get_config(config_id)
        
        # If reliability level is 0, skip reliability processing
        if reliability_level == 0 or (reliability_level is None and actual_config.overall_reliability_level == 0):
            if model:
                return await Runner.run(agent, input, model)
            else:
                return await Runner.run(agent, input)
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Create subtask if using subtask management
        subtask_manager = SubtaskManager()
        context_manager = ContextManager()
        
        if actual_config.subtask.enabled:
            task_description = ""
            if isinstance(input, str):
                task_description = input
            else:
                # Extract description from the first user message
                for item in input:
                    if item.get("role") == "user":
                        task_description = item.get("content", "")
                        break
            
            # Create subtask
            subtask = subtask_manager.create_subtask(
                description=task_description,
                parent_id=None,  # Root task
                is_final=True,   # Single task is final
            )
            
            # Use generated task ID
            task_id = subtask.task_id
        
        # Create retry configuration
        retry_config = RetryConfiguration(
            max_retries=actual_config.error.max_retries,
            retry_delay=actual_config.error.retry_delay,
            backoff_factor=actual_config.error.backoff_factor,
        )
        
        # Run the agent with retry
        try:
            # Pass only the required arguments to Runner.run
            if model:
                args = [agent, input, model]
            else:
                args = [agent, input]
                
            result = await retry_async(
                Runner.run,
                args=args,
                config=retry_config
            )
        except Exception as e:
            # Log error
            log_error(e, {
                "agent_name": agent.name,
                "task_id": task_id,
            })
            
            # Re-raise the exception
            raise
        
        # Extract context from the input
        context_items = []
        task_description = ""
        
        if isinstance(input, str):
            context_items = [{"role": "user", "content": input}]
            task_description = input
            
            # Add to context manager
            context_manager.add_conversation_item("user", input)
        else:
            context_items = input
            
            # Get the first user message as the task description
            for item in input:
                if item.get("role") == "user":
                    task_description = item.get("content", "")
                    
                    # Add to context manager
                    context_manager.add_conversation_item(
                        item.get("role", "user"),
                        item.get("content", "")
                    )
                    break
            else:
                task_description = "Unknown task"
        
        # Process the result with the reliability processor
        try:
            processed_output = await ReliabilityProcessor.process_result(
                result.final_output,
                context_items,
                task_description,
                reliability_level=reliability_level,
                model=model,
                target_language=target_language,
                config_id=config_id,
                task_id=task_id,
            )
            
            # Add processed output to context manager
            context_manager.add_conversation_item(
                "assistant",
                str(processed_output),
                agent_id=agent.name
            )
            
            # Complete subtask if using subtask management
            if actual_config.subtask.enabled:
                subtask_manager.complete_subtask(task_id, processed_output)
            
            # Update the result with the processed output
            # Note: This assumes that _final_output is an attribute of RunResult.
            # This may need to be updated for future SDK versions.
            try:
                result._final_output = processed_output
            except AttributeError:
                # If _final_output is not an attribute, log a warning
                print("Warning: Could not update RunResult._final_output. "
                      "The processed output will not be reflected in the result.")
        
        except Exception as e:
            # Log error
            log_error(e, {
                "agent_name": agent.name,
                "task_id": task_id,
                "validation_task": "process_result",
            })
            
            # If fail_on_validation_error is enabled, re-raise
            if actual_config.error.fail_on_validation_error:
                raise
            
            # Otherwise, use the original result
            # Complete subtask if using subtask management
            if actual_config.subtask.enabled:
                subtask_manager.complete_subtask(task_id, result.final_output)
        
        return result
    
    @staticmethod
    def run_sync(
        agent: Agent[T],
        input: Union[str, List[TResponseInputItem]],
        reliability_level: Optional[int] = None,
        model: Optional[str] = None,
        target_language: Optional[str] = None,
        config: Optional[ReliabilityConfig] = None,
        config_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> RunResult:
        """
        Synchronous version of run.
        
        Args:
            agent: The agent to run
            input: The input to the agent (string or message list)
            reliability_level: Level of reliability checking (0-10)
            model: Model to use for agent and validators
            target_language: Target language for response transformation
            config: Configuration for reliability processing
            config_id: ID of configuration to use
            task_id: ID of task (for subtask management)
            
        Returns:
            The run result with hallucinations removed
        """
        return asyncio.run(ReliableRunner.run(
            agent, input, reliability_level, model, target_language, config, config_id, task_id
        ))
    
    @staticmethod
    async def run_reliably(
        agent: Agent[T],
        input: Union[str, List[TResponseInputItem]],
        model: Optional[str] = None,
        target_language: Optional[str] = None,
        config: Optional[ReliabilityConfig] = None,
        config_id: Optional[str] = None,
    ) -> T:
        """
        Run an agent with reliability processing and return just the final output.
        
        This is a convenience method that runs the agent with reliability processing
        and returns only the final output, not the full RunResult.
        
        Args:
            agent: The agent to run
            input: The input to the agent (string or message list)
            model: Model to use for agent and validators
            target_language: Target language for response transformation
            config: Configuration for reliability processing
            config_id: ID of configuration to use
            
        Returns:
            The processed final output
        """
        result = await ReliableRunner.run(
            agent, input, reliability_level=10, model=model, 
            target_language=target_language, config=config, config_id=config_id
        )
        return result.final_output
    
    @staticmethod
    def run_reliably_sync(
        agent: Agent[T],
        input: Union[str, List[TResponseInputItem]],
        model: Optional[str] = None,
        target_language: Optional[str] = None,
        config: Optional[ReliabilityConfig] = None,
        config_id: Optional[str] = None,
    ) -> T:
        """
        Synchronous version of run_reliably.
        
        Args:
            agent: The agent to run
            input: The input to the agent (string or message list)
            model: Model to use for agent and validators
            target_language: Target language for response transformation
            config: Configuration for reliability processing
            config_id: ID of configuration to use
            
        Returns:
            The processed final output
        """
        return asyncio.run(ReliableRunner.run_reliably(
            agent, input, model, target_language, config, config_id
        ))
    
    @staticmethod
    async def run_subtasks(
        agent: Agent[T],
        subtasks: List[str],
        model: Optional[str] = None,
        target_language: Optional[str] = None,
        config: Optional[ReliabilityConfig] = None,
        config_id: Optional[str] = None,
    ) -> T:
        """
        Run a sequence of subtasks with reliability processing.
        
        This method runs a sequence of subtasks, with the last subtask being treated
        as the final subtask for comprehensive validation.
        
        Args:
            agent: The agent to run
            subtasks: List of subtask descriptions
            model: Model to use for agent and validators
            target_language: Target language for response transformation
            config: Configuration for reliability processing
            config_id: ID of configuration to use
            
        Returns:
            The processed final output from the last subtask
        """
        # Get configuration
        config_manager = ConfigurationManager()
        if config:
            if config_id:
                config_manager.set_config(config, config_id)
            else:
                config_manager.set_default_config(config)
        
        actual_config = config_manager.get_config(config_id)
        
        # Initialize subtask manager
        subtask_manager = SubtaskManager()
        
        # Create a sequence of subtasks
        root_task = subtask_manager.start_sequence(subtasks[0])
        task_id = root_task.task_id
        
        # Run the first subtask
        first_result = await ReliableRunner.run(
            agent, subtasks[0], 
            reliability_level=actual_config.subtask.intermediate_validation_level, 
            model=model, task_id=task_id
        )
        
        # Store the first result
        first_output = first_result.final_output
        subtask_manager.complete_subtask(task_id, first_output)
        
        last_output = first_output
        
        # Run subsequent subtasks
        for i, subtask_desc in enumerate(subtasks[1:]):
            is_final = i == len(subtasks) - 2  # Last subtask
            
            # Create subtask
            subtask = subtask_manager.add_to_sequence(
                parent_id=task_id,
                description=subtask_desc,
                is_final=is_final
            )
            
            # Create context with previous results
            context = subtask_manager.get_context_for_subtask(subtask.task_id)
            
            # Create input with previous results
            if isinstance(subtask_desc, str):
                input_with_context = [
                    {"role": "system", "content": "Previous results: " + str(context)},
                    {"role": "user", "content": subtask_desc}
                ]
            else:
                # Assume it's already a list of messages
                input_with_context = subtask_desc
            
            # Run the subtask
            validation_level = 10 if is_final else actual_config.subtask.intermediate_validation_level
            
            subtask_result = await ReliableRunner.run(
                agent, input_with_context, 
                reliability_level=validation_level, 
                model=model, task_id=subtask.task_id
            )
            
            # Store the result
            subtask_output = subtask_result.final_output
            subtask_manager.complete_subtask(subtask.task_id, subtask_output)
            
            last_output = subtask_output
        
        # Apply language transformation if needed
        if target_language:
            from .language import transform_to_language
            last_output = await transform_to_language(last_output, target_language, model)
        
        return last_output
    
    @staticmethod
    def run_subtasks_sync(
        agent: Agent[T],
        subtasks: List[str],
        model: Optional[str] = None,
        target_language: Optional[str] = None,
        config: Optional[ReliabilityConfig] = None,
        config_id: Optional[str] = None,
    ) -> T:
        """
        Synchronous version of run_subtasks.
        
        Args:
            agent: The agent to run
            subtasks: List of subtask descriptions
            model: Model to use for agent and validators
            target_language: Target language for response transformation
            config: Configuration for reliability processing
            config_id: ID of configuration to use
            
        Returns:
            The processed final output from the last subtask
        """
        return asyncio.run(ReliableRunner.run_subtasks(
            agent, subtasks, model, target_language, config, config_id
        ))
