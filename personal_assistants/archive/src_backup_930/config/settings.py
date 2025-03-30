"""
Configuration settings for the voice assistant application.
Loads values from .env file if available.
"""
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

# Vector store configuration
VECTOR_STORE_ID: Optional[str] = os.getenv("VECTOR_STORE_ID")

# Audio settings
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))

# Application defaults
DEFAULT_MODE: str = os.getenv("DEFAULT_MODE", "text")
DEFAULT_VOICE_PROFILE: str = os.getenv("DEFAULT_VOICE_PROFILE", "upbeat")

# TTS settings profiles
TTS_PROFILES = {
    "default": {
        "instructions": (
            "Personality: helpful, friendly assistant. "
            "Tone: Clear, professional, and informative. "
            "Pronunciation: Well-articulated with natural pacing."
        )
    },
    "upbeat": {
        "instructions": (
            "Personality: upbeat, friendly, persuasive guide. "
            "Tone: Friendly, clear, and reassuring, creating a calm atmosphere. "
            "Pronunciation: Clear, articulate, and steady, maintaining a natural flow. "
            "Tempo: Speak relatively fast, include brief pauses before questions. "
            "Emotion: Warm and supportive, conveying empathy and care."
        )
    },
    "character": {
        "instructions": (
            "Affect: Deep, commanding, and slightly dramatic, with an archaic quality. "
            "Tone: Noble, heroic, and formal, capturing the essence of medieval knights. "
            "Emotion: Excitement, anticipation, and a sense of mystery. "
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