"""
Text-to-speech settings for the voice assistant.
"""
from typing import Dict, Any
from agents.voice import TTSModelSettings
from src.config.settings import TTS_PROFILES

def get_tts_settings(profile: str = "default") -> TTSModelSettings:
    """
    Get TTS model settings for a specific profile.
    
    Args:
        profile: The name of the TTS profile to use
        
    Returns:
        TTSModelSettings: The configured settings
    """
    if profile not in TTS_PROFILES:
        profile = "default"
        
    return TTSModelSettings(
        instructions=TTS_PROFILES[profile]["instructions"]
    )