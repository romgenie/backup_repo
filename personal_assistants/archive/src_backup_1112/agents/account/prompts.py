"""
Prompts for the account agent.
"""
from src.config.settings import VOICE_SYSTEM_PROMPT

# Base instructions for the account agent
BASE_INSTRUCTIONS = """
You provide account information based on a user ID using the get_account_info tool.
"""

# Voice-optimized instructions for the account agent
VOICE_INSTRUCTIONS = VOICE_SYSTEM_PROMPT + BASE_INSTRUCTIONS