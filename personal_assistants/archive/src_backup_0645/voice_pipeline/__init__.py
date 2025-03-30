"""
Voice pipeline module for the voice assistant application.
"""
from src.voice_pipeline.pipeline import create_voice_pipeline
from src.voice_pipeline.workflow import create_voice_workflow
from src.voice_pipeline.tts_settings import get_tts_settings

__all__ = [
    'create_voice_pipeline',
    'create_voice_workflow',
    'get_tts_settings',
]