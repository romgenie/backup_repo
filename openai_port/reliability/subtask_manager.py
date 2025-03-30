"""
Subtask management for reliability processing.

This module provides the SubtaskManager class, which tracks and manages
subtasks for complex workflows requiring multiple agent runs.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Set, Union

from .context_manager import ContextManager


class Subtask:
    """
    Representation of a subtask within a larger task.
    
    Attributes:
        task_id: Unique identifier for the subtask
        parent_id: ID of the parent task (if any)
        description: Description of the subtask
        is_final: Whether this is the final subtask
        status: Current status of the subtask
        created_at: Timestamp when subtask was created
        completed_at: Timestamp when subtask was completed
        result: Result of the subtask execution
        validation_result: Result of validation for this subtask
    """
    
    def __init__(
        self,
        description: str,
        parent_id: Optional[str] = None,
        is_final: bool = False,
    ):
        """
        Initialize a subtask.
        
        Args:
            description: Description of the subtask
            parent_id: ID of the parent task (if any)
            is_final: Whether this is the final subtask
        """
        self.task_id = str(uuid.uuid4())
        self.parent_id = parent_id
        self.description = description
        self.is_final = is_final
        self.status = "created"
        self.created_at = time.time()
        self.completed_at = None
        self.result = None
        self.validation_result = None
    
    def complete(self, result: Any):
        """
        Mark the subtask as completed with the given result.
        
        Args:
            result: Result of the subtask execution
        """
        self.status = "completed"
        self.completed_at = time.time()
        self.result = result
    
    def set_validation_result(self, validation_result: Any):
        """
        Set the validation result for this subtask.
        
        Args:
            validation_result: Result of validation
        """
        self.validation_result = validation_result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert subtask to dictionary representation.
        
        Returns:
            Dictionary representation of the subtask
        """
        return {
            "task_id": self.task_id,
            "parent_id": self.parent_id,
            "description": self.description,
            "is_final": self.is_final,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class SubtaskManager:
    """
    Manager for subtasks within a complex task.
    
    The SubtaskManager tracks parent-child relationships between tasks,
    manages task sequences, and applies appropriate reliability processing.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SubtaskManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the subtask manager state."""
        self.subtasks = {}
        self.active_tasks = {}
        self.task_sequences = {}
        self.context_manager = ContextManager()
    
    def reset(self):
        """Reset the subtask manager state."""
        self._initialize()
    
    def create_subtask(
        self,
        description: str,
        parent_id: Optional[str] = None,
        is_final: bool = False,
    ) -> Subtask:
        """
        Create a new subtask.
        
        Args:
            description: Description of the subtask
            parent_id: ID of the parent task (if any)
            is_final: Whether this is the final subtask
            
        Returns:
            The created subtask
        """
        subtask = Subtask(description, parent_id, is_final)
        self.subtasks[subtask.task_id] = subtask
        
        # If this is part of a sequence, add it to the sequence
        if parent_id and parent_id in self.task_sequences:
            self.task_sequences[parent_id].append(subtask.task_id)
        elif parent_id:
            self.task_sequences[parent_id] = [subtask.task_id]
        else:
            # This is a root task, create a new sequence
            self.task_sequences[subtask.task_id] = []
        
        self.active_tasks[subtask.task_id] = subtask
        return subtask
    
    def start_sequence(self, description: str) -> Subtask:
        """
        Start a new task sequence.
        
        Args:
            description: Description of the root task
            
        Returns:
            The root subtask
        """
        return self.create_subtask(description)
    
    def add_to_sequence(
        self,
        parent_id: str,
        description: str,
        is_final: bool = False,
    ) -> Subtask:
        """
        Add a subtask to an existing sequence.
        
        Args:
            parent_id: ID of the parent task
            description: Description of the subtask
            is_final: Whether this is the final subtask
            
        Returns:
            The created subtask
        """
        return self.create_subtask(description, parent_id, is_final)
    
    def complete_subtask(self, task_id: str, result: Any) -> bool:
        """
        Mark a subtask as completed.
        
        Args:
            task_id: ID of the subtask
            result: Result of the subtask execution
            
        Returns:
            True if successful, False otherwise
        """
        if task_id not in self.subtasks:
            return False
        
        subtask = self.subtasks[task_id]
        subtask.complete(result)
        
        # If this is an active task, remove it
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        
        # Add result to context manager
        context_key = f"subtask_result_{task_id}"
        self.context_manager.add_agent_context(context_key, result)
        
        return True
    
    def set_validation_result(self, task_id: str, validation_result: Any) -> bool:
        """
        Set the validation result for a subtask.
        
        Args:
            task_id: ID of the subtask
            validation_result: Result of validation
            
        Returns:
            True if successful, False otherwise
        """
        if task_id not in self.subtasks:
            return False
        
        subtask = self.subtasks[task_id]
        subtask.set_validation_result(validation_result)
        
        # Add validation result to context manager
        validation_id = f"subtask_validation_{task_id}"
        self.context_manager.add_validation_result(validation_id, validation_result)
        
        return True
    
    def get_subtask(self, task_id: str) -> Optional[Subtask]:
        """
        Get a subtask by ID.
        
        Args:
            task_id: ID of the subtask
            
        Returns:
            The subtask, or None if not found
        """
        return self.subtasks.get(task_id)
    
    def get_sequence(self, root_id: str) -> List[Subtask]:
        """
        Get all subtasks in a sequence.
        
        Args:
            root_id: ID of the root task
            
        Returns:
            List of subtasks in the sequence
        """
        sequence = []
        
        # Add the root task
        if root_id in self.subtasks:
            sequence.append(self.subtasks[root_id])
        
        # Add all subtasks in the sequence
        if root_id in self.task_sequences:
            for task_id in self.task_sequences[root_id]:
                if task_id in self.subtasks:
                    sequence.append(self.subtasks[task_id])
        
        return sequence
    
    def get_final_subtask(self, root_id: str) -> Optional[Subtask]:
        """
        Get the final subtask in a sequence.
        
        Args:
            root_id: ID of the root task
            
        Returns:
            The final subtask, or None if not found
        """
        sequence = self.get_sequence(root_id)
        
        # Filter for final subtasks
        final_subtasks = [s for s in sequence if s.is_final]
        
        # Return the last final subtask
        if final_subtasks:
            return final_subtasks[-1]
        
        # If no final subtasks, return the last subtask in the sequence
        if sequence:
            return sequence[-1]
        
        return None
    
    def get_previous_results(self, task_id: str) -> List[Any]:
        """
        Get results from previous subtasks in the same sequence.
        
        Args:
            task_id: ID of the current subtask
            
        Returns:
            List of results from previous subtasks
        """
        results = []
        
        subtask = self.get_subtask(task_id)
        if not subtask:
            return results
        
        # If this is a root task, there are no previous results
        if not subtask.parent_id:
            return results
        
        # Get the sequence
        sequence = self.get_sequence(subtask.parent_id)
        
        # Find the current task in the sequence
        current_index = -1
        for i, s in enumerate(sequence):
            if s.task_id == task_id:
                current_index = i
                break
        
        # Get results from previous tasks
        if current_index > 0:
            for i in range(current_index):
                if sequence[i].result is not None:
                    results.append(sequence[i].result)
        
        return results
    
    def should_validate(self, task_id: str) -> bool:
        """
        Determine if a subtask should be validated.
        
        In general, only final subtasks or subtasks without children
        should be validated to avoid unnecessary validation.
        
        Args:
            task_id: ID of the subtask
            
        Returns:
            True if the subtask should be validated, False otherwise
        """
        subtask = self.get_subtask(task_id)
        if not subtask:
            return False
        
        # If this is a final subtask, it should be validated
        if subtask.is_final:
            return True
        
        # If this is a subtask without children, it should be validated
        if task_id in self.task_sequences and not self.task_sequences[task_id]:
            return True
        
        # Otherwise, it should not be validated
        return False
    
    def get_validation_level(self, task_id: str) -> int:
        """
        Get the appropriate validation level for a subtask.
        
        Final subtasks should receive comprehensive validation (level 10),
        while intermediate subtasks may receive lighter validation.
        
        Args:
            task_id: ID of the subtask
            
        Returns:
            Validation level (0-10)
        """
        subtask = self.get_subtask(task_id)
        if not subtask:
            return 0
        
        # If this is a final subtask, use level 10
        if subtask.is_final:
            return 10
        
        # If this is a subtask without children, use level 10
        if task_id in self.task_sequences and not self.task_sequences[task_id]:
            return 10
        
        # For intermediate subtasks, use level 5
        return 5
    
    def get_context_for_subtask(self, task_id: str) -> List[Any]:
        """
        Get relevant context for a subtask.
        
        This includes previous subtask results and other context information.
        
        Args:
            task_id: ID of the subtask
            
        Returns:
            List of context items
        """
        context = []
        
        # Add previous results
        previous_results = self.get_previous_results(task_id)
        context.extend(previous_results)
        
        # Add recent conversation history
        recent_conversation = self.context_manager.get_recent_conversation()
        context.extend(recent_conversation)
        
        return context
