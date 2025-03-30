"""
Prompts for the knowledge agent.
"""
from src.config.settings import VOICE_SYSTEM_PROMPT

# Base instructions for the knowledge agent
BASE_INSTRUCTIONS = """
You answer user questions on our product portfolio with concise, helpful responses using the FileSearchTool.
"""

# Voice-optimized instructions for the knowledge agent
VOICE_INSTRUCTIONS = VOICE_SYSTEM_PROMPT + BASE_INSTRUCTIONS