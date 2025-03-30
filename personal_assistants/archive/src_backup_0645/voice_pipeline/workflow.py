"""
Voice workflow implementation.
"""
from agents import Agent
from agents.voice import SingleAgentVoiceWorkflow

def create_voice_workflow(agent: Agent) -> SingleAgentVoiceWorkflow:
    """
    Create a voice workflow for the specified agent.
    
    Args:
        agent: The agent to use in the workflow
        
    Returns:
        SingleAgentVoiceWorkflow: The configured workflow
    """
    return SingleAgentVoiceWorkflow(agent)