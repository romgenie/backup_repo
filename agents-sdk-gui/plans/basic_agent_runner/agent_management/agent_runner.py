"""
Agent runner module for launching agents and managing their execution.

This module has been refactored into smaller components:
- agent_management/execution/ - Core execution logic
"""

# Import from the refactored execution module to maintain backward compatibility
from .execution import run_agent_with_history, get_weather, search_news

# Make these functions and tools available from this module
__all__ = ['run_agent_with_history', 'get_weather', 'search_news']