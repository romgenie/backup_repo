"""
Agents module for the voice assistant application.
"""
from src.agents.account.agent import create_account_agent
from src.agents.knowledge.agent import create_knowledge_agent
from src.agents.search.agent import create_search_agent
from src.agents.triage.agent import create_triage_agent

__all__ = [
    'create_account_agent',
    'create_knowledge_agent',
    'create_search_agent',
    'create_triage_agent',
]