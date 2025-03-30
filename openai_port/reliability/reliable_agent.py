"""
ReliableAgent class for agents with built-in reliability processing.

This module provides the ReliableAgent class, which extends the standard Agent
with built-in reliability processing to reduce hallucinations.
"""

import asyncio
from typing import Any, Generic, List, Optional, Type, TypeVar, Union, Dict

from agents import Agent, ModelSettings, TResponseInputItem, RunResult

from .runner import ReliableRunner
from .config import ReliabilityConfig, ConfigurationManager, get_default_config
from .context_manager import ContextManager
from .subtask_manager import SubtaskManager

T = TypeVar('T')


class ReliableAgent(Agent[T], Generic[T]):
    """
    An agent with built-in reliability processing.
    
    ReliableAgent extends the standard Agent with built-in reliability
    processing to reduce hallucinations in agent responses.
    """
    
    def __init__(
        self,
        *args,
        reliability_level: Optional[int] = None,
        target_language: Optional[str] = None,
        config: Optional[ReliabilityConfig] = None,
        config_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a ReliableAgent.
        
        Args:
            *args: Arguments to pass to the Agent constructor
            reliability_level: Level of reliability checking (0-10)
            target_language: Target language for response transformation
            config: Configuration for reliability processing
            config_id: ID of configuration to use
            **kwargs: Keyword arguments to pass to the Agent constructor
        """
        super().__init__(*args, **kwargs)
        self.reliability_level = reliability_level
        self.target_language = target_language
        self.config_id = config_id
        
        # Initialize configuration
        self.config_manager = ConfigurationManager()
        if config:
            if config_id:
                self.config_manager.set_config(config, config_id)
            else:
                self.config_manager.set_default_config(config)
                self.config = config
        else:
            self.config = self.config_manager.get_config(config_id)
        
        # Initialize managers
        self.context_manager = ContextManager()
        self.subtask_manager = SubtaskManager()
    
    async def run(
        self,
        input: Union[str, List[TResponseInputItem]],
        model: Optional[str] = None,
    ) -> RunResult:
        """
        Run the agent with reliability processing.
        
        Args:
            input: The input to the agent (string or message list)
            model: Model to use for agent and validators
            
        Returns:
            The run result with reliability processing applied
        """
        return await ReliableRunner.run(
            self, input, 
            reliability_level=self.reliability_level,
            model=model,
            target_language=self.target_language,
            config_id=self.config_id
        )
    
    def run_sync(
        self,
        input: Union[str, List[TResponseInputItem]],
        model: Optional[str] = None,
    ) -> RunResult:
        """
        Synchronous version of run.
        
        Args:
            input: The input to the agent (string or message list)
            model: Model to use for agent and validators
            
        Returns:
            The run result with reliability processing applied
        """
        return ReliableRunner.run_sync(
            self, input, 
            reliability_level=self.reliability_level,
            model=model,
            target_language=self.target_language,
            config_id=self.config_id
        )
    
    async def run_reliably(
        self,
        input: Union[str, List[TResponseInputItem]],
        model: Optional[str] = None,
    ) -> T:
        """
        Run the agent with reliability processing and return just the final output.
        
        Args:
            input: The input to the agent (string or message list)
            model: Model to use for agent and validators
            
        Returns:
            The processed final output
        """
        result = await self.run(input, model)
        return result.final_output
    
    def run_reliably_sync(
        self,
        input: Union[str, List[TResponseInputItem]],
        model: Optional[str] = None,
    ) -> T:
        """
        Synchronous version of run_reliably.
        
        Args:
            input: The input to the agent (string or message list)
            model: Model to use for agent and validators
            
        Returns:
            The processed final output
        """
        result = self.run_sync(input, model)
        return result.final_output
    
    async def run_subtasks(
        self,
        subtasks: List[str],
        model: Optional[str] = None,
    ) -> T:
        """
        Run a sequence of subtasks with reliability processing.
        
        Args:
            subtasks: List of subtask descriptions
            model: Model to use for agent and validators
            
        Returns:
            The processed final output from the last subtask
        """
        return await ReliableRunner.run_subtasks(
            self, subtasks, model, 
            self.target_language, None, self.config_id
        )
    
    def run_subtasks_sync(
        self,
        subtasks: List[str],
        model: Optional[str] = None,
    ) -> T:
        """
        Synchronous version of run_subtasks.
        
        Args:
            subtasks: List of subtask descriptions
            model: Model to use for agent and validators
            
        Returns:
            The processed final output from the last subtask
        """
        return ReliableRunner.run_subtasks_sync(
            self, subtasks, model, 
            self.target_language, None, self.config_id
        )
    
    def set_reliability_level(self, level: int):
        """
        Set the reliability level.
        
        Args:
            level: Reliability level (0-10)
        """
        self.reliability_level = level
    
    def set_target_language(self, language: str):
        """
        Set the target language for response transformation.
        
        Args:
            language: Target language
        """
        self.target_language = language
    
    def set_config(self, config: ReliabilityConfig):
        """
        Set the reliability configuration.
        
        Args:
            config: Reliability configuration
        """
        self.config = config
        
        if self.config_id:
            self.config_manager.set_config(config, self.config_id)
        else:
            self.config_manager.set_default_config(config)
