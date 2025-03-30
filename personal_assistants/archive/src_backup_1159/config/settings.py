"""
Configuration settings for the personal assistant application.
Loads values from both .env and .env_personalization files if available.
"""
from typing import Optional, Dict, Any, cast
import os
from dotenv import load_dotenv

# Load environment variables from .env file (API keys, etc.)
load_dotenv()

# Load personalization settings if available
personalization_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env_personalization')
if os.path.exists(personalization_path):
    load_dotenv(dotenv_path=personalization_path)

# OpenAI API configuration
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

# Vector store configuration
VECTOR_STORE_ID: Optional[str] = os.getenv("VECTOR_STORE_ID")

# Audio settings
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))

# Organization and branding
ORGANIZATION_NAME: str = os.getenv("ORGANIZATION_NAME", "Personal Assistant")
ORGANIZATION_DESCRIPTION: str = os.getenv("ORGANIZATION_DESCRIPTION", "A personal AI assistant that helps with various tasks")

# Assistant customization
ASSISTANT_NAME: str = os.getenv("ASSISTANT_NAME", "Assistant")
GREETING_MESSAGE: str = os.getenv("GREETING_MESSAGE", "Hello! How can I help you today?")
FAREWELL_MESSAGE: str = os.getenv("FAREWELL_MESSAGE", "Goodbye! Let me know if you need anything else.")

# Menu display options
DISPLAY_TEXT_MODE: bool = os.getenv("DISPLAY_TEXT_MODE", "true").lower() == "true"
DISPLAY_VOICE_MODES: bool = os.getenv("DISPLAY_VOICE_MODES", "true").lower() == "true" 
DISPLAY_VECTOR_STORE_MODES: bool = os.getenv("DISPLAY_VECTOR_STORE_MODES", "true").lower() == "true"
DISPLAY_VECTOR_STORE_WARNINGS: bool = os.getenv("DISPLAY_VECTOR_STORE_WARNINGS", "true").lower() == "true"

# Application defaults
DEFAULT_MODE: str = os.getenv("DEFAULT_MODE", "text")
DEFAULT_VOICE_PROFILE: str = os.getenv("DEFAULT_VOICE_PROFILE", "upbeat")
AUTO_START: bool = os.getenv("AUTO_START", "false").lower() == "true"

# Knowledge paths
KNOWLEDGE_DIRECTORY: str = os.getenv("KNOWLEDGE_DIRECTORY", "/Users/timgregg/Desktop/personal_assistants/src/data/knowledge")

# TTS settings profiles
TTS_PROFILES: Dict[str, Dict[str, str]] = {
    "default": {
        "instructions": (
            f"Personality: {os.getenv('VOICE_DEFAULT_PERSONALITY', 'helpful, friendly assistant')}. "
            f"Tone: {os.getenv('VOICE_DEFAULT_TONE', 'Clear, professional, and informative')}. "
            "Pronunciation: Well-articulated with natural pacing."
        )
    },
    "upbeat": {
        "instructions": (
            f"Personality: {os.getenv('VOICE_UPBEAT_PERSONALITY', 'upbeat, friendly, persuasive guide')}. "
            f"Tone: {os.getenv('VOICE_UPBEAT_TONE', 'Friendly, clear, and reassuring')}. "
            "Pronunciation: Clear, articulate, and steady, maintaining a natural flow. "
            "Tempo: Speak relatively fast, include brief pauses before questions. "
            "Emotion: Warm and supportive, conveying empathy and care."
        )
    },
    "character": {
        "instructions": (
            f"Personality: {os.getenv('VOICE_CHARACTER_PERSONALITY', 'Sophisticated, thoughtful, and slightly formal')}. "
            f"Tone: {os.getenv('VOICE_CHARACTER_TONE', 'Poised, measured, and distinguished')}. "
            "Pronunciation: Clear, deliberate, and with a slightly formal cadence. "
            "Pause: Strategic pauses after important phrases to add weight."
        )
    }
}

# Voice system prompts
VOICE_SYSTEM_PROMPT = """
[Output Structure]
Your output will be delivered in an audio voice response, please ensure that every response meets these guidelines:
1. Use a friendly, human tone that will sound natural when spoken aloud.
2. Keep responses short and segmented—ideally one to two concise sentences per step.
3. Avoid technical jargon; use plain language so that instructions are easy to understand.
4. Provide only essential details so as not to overwhelm the listener.
"""