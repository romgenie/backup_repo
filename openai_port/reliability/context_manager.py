"""
Context management for reliability processing.

This module provides the ContextManager class, which maintains context
information across multiple agent runs to improve validation accuracy.
"""

import time
from typing import Any, Dict, List, Optional, Set, Union
import uuid

class ContextManager:
    """
    Singleton class for managing context across multiple agent runs.
    
    The ContextManager maintains a history of interactions, validation results,
    and other metadata that can be used to improve validation accuracy.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ContextManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the context manager state."""
        self.conversation_history = []
        self.validation_history = {}
        self.trusted_sources = set()
        self.suspicious_points = set()
        self.context_by_agent = {}
        self.last_accessed = time.time()
        self.session_id = str(uuid.uuid4())
        self.expiry_time = 3600  # 1 hour default
    
    def reset(self):
        """Reset the context manager state."""
        self._initialize()
    
    def add_conversation_item(self, role: str, content: str, agent_id: Optional[str] = None):
        """
        Add a conversation item to the history.
        
        Args:
            role: The role of the sender (user, assistant, system)
            content: The content of the message
            agent_id: Optional ID of the agent that generated the message
        """
        item = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "agent_id": agent_id,
        }
        self.conversation_history.append(item)
        self.last_accessed = time.time()
    
    def add_validation_result(self, validation_id: str, result: Any):
        """
        Add a validation result to the history.
        
        Args:
            validation_id: Unique identifier for the validation
            result: The validation result
        """
        self.validation_history[validation_id] = {
            "result": result,
            "timestamp": time.time(),
        }
        
        # Update trusted sources and suspicious points
        if hasattr(result, "url_validation") and not result.url_validation.is_suspicious:
            # Extract URLs from the content as trusted sources
            from .validators import find_urls_in_text
            content = str(result.url_validation)
            urls = find_urls_in_text(content)
            self.trusted_sources.update(urls)
        
        if hasattr(result, "suspicious_points"):
            self.suspicious_points.update(result.suspicious_points)
        
        self.last_accessed = time.time()
    
    def get_validation_result(self, validation_id: str) -> Optional[Any]:
        """
        Get a validation result from the history.
        
        Args:
            validation_id: Unique identifier for the validation
            
        Returns:
            The validation result, or None if not found or expired
        """
        if validation_id not in self.validation_history:
            return None
        
        entry = self.validation_history[validation_id]
        # Check if expired
        if time.time() - entry["timestamp"] > self.expiry_time:
            del self.validation_history[validation_id]
            return None
        
        return entry["result"]
    
    def add_agent_context(self, agent_id: str, context: Any):
        """
        Add context information for a specific agent.
        
        Args:
            agent_id: ID of the agent
            context: Context information
        """
        self.context_by_agent[agent_id] = {
            "context": context,
            "timestamp": time.time(),
        }
        self.last_accessed = time.time()
    
    def get_agent_context(self, agent_id: str) -> Optional[Any]:
        """
        Get context information for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            The agent context, or None if not found or expired
        """
        if agent_id not in self.context_by_agent:
            return None
        
        entry = self.context_by_agent[agent_id]
        # Check if expired
        if time.time() - entry["timestamp"] > self.expiry_time:
            del self.context_by_agent[agent_id]
            return None
        
        return entry["context"]
    
    def add_trusted_source(self, source: str):
        """
        Add a trusted source to the context.
        
        Args:
            source: URL or identifier of the trusted source
        """
        self.trusted_sources.add(source)
        self.last_accessed = time.time()
    
    def is_trusted_source(self, source: str) -> bool:
        """
        Check if a source is trusted.
        
        Args:
            source: URL or identifier to check
            
        Returns:
            True if the source is trusted, False otherwise
        """
        return source in self.trusted_sources
    
    def add_suspicious_point(self, point: str):
        """
        Add a suspicious point to the context.
        
        Args:
            point: Description of the suspicious point
        """
        self.suspicious_points.add(point)
        self.last_accessed = time.time()
    
    def is_suspicious_point(self, point: str) -> bool:
        """
        Check if a point has been identified as suspicious.
        
        Args:
            point: Description of the point to check
            
        Returns:
            True if the point is suspicious, False otherwise
        """
        return point in self.suspicious_points
    
    def get_recent_conversation(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent conversation items.
        
        Args:
            count: Maximum number of items to return
            
        Returns:
            List of recent conversation items
        """
        return self.conversation_history[-count:]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context.
        
        Returns:
            Dictionary with context summary information
        """
        return {
            "session_id": self.session_id,
            "conversation_length": len(self.conversation_history),
            "trusted_sources_count": len(self.trusted_sources),
            "suspicious_points_count": len(self.suspicious_points),
            "validation_results_count": len(self.validation_history),
            "agent_contexts_count": len(self.context_by_agent),
            "last_accessed": self.last_accessed,
        }
    
    def set_expiry_time(self, seconds: int):
        """
        Set the expiry time for context items.
        
        Args:
            seconds: Expiry time in seconds
        """
        self.expiry_time = seconds
    
    def cleanup_expired(self):
        """
        Remove expired items from the context.
        """
        now = time.time()
        
        # Clean up validation history
        expired_validations = []
        for validation_id, entry in self.validation_history.items():
            if now - entry["timestamp"] > self.expiry_time:
                expired_validations.append(validation_id)
        
        for validation_id in expired_validations:
            del self.validation_history[validation_id]
        
        # Clean up agent contexts
        expired_contexts = []
        for agent_id, entry in self.context_by_agent.items():
            if now - entry["timestamp"] > self.expiry_time:
                expired_contexts.append(agent_id)
        
        for agent_id in expired_contexts:
            del self.context_by_agent[agent_id]
        
        # Update last accessed time
        self.last_accessed = now
