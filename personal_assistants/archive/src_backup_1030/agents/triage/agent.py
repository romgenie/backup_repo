"""
Triage agent implementation.
"""
import re
from typing import Callable, Awaitable, Any, Union, Optional
from agents import Agent
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from src.agents.triage.prompts import BASE_INSTRUCTIONS
from src.agents.account.agent import create_account_agent
from src.agents.knowledge.agent import create_knowledge_agent
from src.agents.search.agent import create_search_agent
from src.agents.vector_store.agent import create_vector_store_agent

def create_triage_agent(
    use_voice_optimization: bool = False
) -> Agent:
    """
    Create a triage agent that routes requests to specialized agents.
    
    Args:
        use_voice_optimization: Whether to use voice-optimized instructions for sub-agents
        
    Returns:
        Agent: The initialized triage agent
    """
    # Create specialized agents
    account_agent = create_account_agent(use_voice_optimization)
    knowledge_agent = create_knowledge_agent(use_voice_optimization)
    search_agent = create_search_agent(use_voice_optimization)
    vector_store_agent = create_vector_store_agent(use_voice_optimization)
    
    # Create the agent with the instructions from prompts.py
    agent = Agent(
        name="Assistant",
        instructions=prompt_with_handoff_instructions(BASE_INSTRUCTIONS),
        handoffs=[account_agent, knowledge_agent, search_agent, vector_store_agent],
    )
    
    return agent