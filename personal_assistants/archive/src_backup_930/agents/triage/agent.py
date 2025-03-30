"""
Triage agent implementation.
"""
from agents import Agent
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from src.agents.triage.prompts import BASE_INSTRUCTIONS
from src.agents.account.agent import create_account_agent
from src.agents.knowledge.agent import create_knowledge_agent
from src.agents.search.agent import create_search_agent

def create_triage_agent(use_voice_optimization: bool = False) -> Agent:
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
    
    return Agent(
        name="Assistant",
        instructions=prompt_with_handoff_instructions(BASE_INSTRUCTIONS),
        handoffs=[account_agent, knowledge_agent, search_agent],
    )