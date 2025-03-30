"""
Audio processing utilities.
"""
from typing import List, Callable, Any, Tuple, Optional

# Try to import audio dependencies
try:
    import numpy as np
    import sounddevice as sd
    from agents.voice import AudioInput
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio features not available. Install with: pip install 'openai-agents[voice]'")
    
    # Define dummy classes for type hints
    class DummyAudioInput:
        def __init__(self, buffer=None):
            pass
            
    AudioInput = DummyAudioInput

def record_audio(
    samplerate: int = 16000,
    stop_condition: Callable[[], bool] = None
) -> Tuple[np.ndarray, int]:
    """
    Record audio from the microphone.
    
    Args:
        samplerate: Sample rate to use for recording
        stop_condition: Callable that returns True when recording should stop
        
    Returns:
        Tuple containing the recorded audio buffer and sample rate
    """
    recorded_chunks = []

    # Define callback function for audio stream
    def callback(indata, frames, time, status):
        recorded_chunks.append(indata.copy())

    # Start recording stream
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', callback=callback):
        if stop_condition is None:
            # Default stop condition is to wait for user to press Enter
            input("Press Enter to stop recording...")
        else:
            while not stop_condition():
                # Keep the stream open until stop condition is met
                sd.sleep(100)

    # Concatenate all recorded chunks
    if recorded_chunks:
        recording = np.concatenate(recorded_chunks, axis=0)
        return recording, samplerate
    
    # Return empty buffer if no audio was recorded
    return np.array([], dtype=np.int16), samplerate

def create_audio_input(audio_buffer: np.ndarray) -> AudioInput:
    """
    Create an AudioInput object from an audio buffer.
    
    Args:
        audio_buffer: NumPy array containing audio data
        
    Returns:
        AudioInput: Object for use with voice pipelines
    """
    return AudioInput(buffer=audio_buffer)

def play_audio(audio_buffer: np.ndarray, samplerate: int = 16000) -> None:
    """
    Play audio buffer through the default audio output.
    
    Args:
        audio_buffer: NumPy array containing audio data
        samplerate: Sample rate of the audio data
    """
    if len(audio_buffer) > 0:
        sd.play(audio_buffer, samplerate=samplerate)
        sd.wait()  # Wait until playback is finished
    else:
        print("Warning: Empty audio buffer, nothing to play.")