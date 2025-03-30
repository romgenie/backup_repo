"""
Account agent implementation.
"""
from agents import Agent
from src.agents.account.prompts import BASE_INSTRUCTIONS, VOICE_INSTRUCTIONS
from src.tools.account.account_info import get_account_info

def create_account_agent(use_voice_optimization: bool = False) -> Agent:
    """
    Create an account agent that provides account information.
    
    Args:
        use_voice_optimization: Whether to use voice-optimized instructions
        
    Returns:
        Agent: The initialized account agent
    """
    instructions = VOICE_INSTRUCTIONS if use_voice_optimization else BASE_INSTRUCTIONS
    
    return Agent(
        name="AccountAgent",
        instructions=instructions,
        tools=[get_account_info],
    )