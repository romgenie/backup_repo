"""
Prompts for the search agent.
"""
from src.config.settings import VOICE_SYSTEM_PROMPT

# Base instructions for the search agent
BASE_INSTRUCTIONS = """
You immediately provide an input to the WebSearchTool to find up-to-date information on the user's query.
"""

# Voice-optimized instructions for the search agent
VOICE_INSTRUCTIONS = VOICE_SYSTEM_PROMPT + BASE_INSTRUCTIONS