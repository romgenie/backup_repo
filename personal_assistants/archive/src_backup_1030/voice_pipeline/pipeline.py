"""
Voice pipeline implementation.
"""
from typing import Optional
from agents.voice import VoicePipeline, VoicePipelineConfig, SingleAgentVoiceWorkflow
from src.voice_pipeline.tts_settings import get_tts_settings

def create_voice_pipeline(
    workflow: SingleAgentVoiceWorkflow,
    tts_profile: str = "default"
) -> VoicePipeline:
    """
    Create a voice pipeline with the specified workflow and TTS profile.
    
    Args:
        workflow: The voice workflow to use
        tts_profile: The name of the TTS profile to use
        
    Returns:
        VoicePipeline: The configured pipeline
    """
    tts_settings = get_tts_settings(tts_profile)
    config = VoicePipelineConfig(tts_settings=tts_settings)
    
    return VoicePipeline(workflow=workflow, config=config)