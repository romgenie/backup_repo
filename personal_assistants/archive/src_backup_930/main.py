"""
Main entry point for the voice assistant application.
"""
import numpy as np
import asyncio
from agents import Runner, trace, set_default_openai_key
from agents.voice import AudioInput
from src.config.settings import OPENAI_API_KEY
from src.agents import create_triage_agent
from src.voice_pipeline import create_voice_workflow, create_voice_pipeline
from src.utils.audio import record_audio, play_audio

async def run_text_assistant():
    """Run the assistant in text mode."""
    # Create the triage agent
    triage_agent = create_triage_agent(use_voice_optimization=False)
    
    print("ACME Shop Text Assistant")
    print("Type 'exit' to quit")
    print("---")
    
    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        
        # Process the input
        with trace("ACME Shop Text Assistant"):
            result = await Runner.run(triage_agent, user_input)
        
        # Display the result
        print(f"Assistant: {result.final_output}")
        print("---")

async def run_voice_assistant(tts_profile: str = "upbeat"):
    """
    Run the assistant in voice mode.
    
    Args:
        tts_profile: The TTS profile to use ("default", "upbeat", "character")
    """
    # Create the triage agent with voice optimization
    triage_agent = create_triage_agent(use_voice_optimization=True)
    
    print("ACME Shop Voice Assistant")
    print("Press Enter to start/stop recording")
    print("Type 'exit' to quit")
    print("---")
    
    while True:
        # Check if user wants to exit
        cmd = input("Press Enter to speak (or type 'exit' to quit): ")
        if cmd.lower() == "exit":
            print("Exiting...")
            break
        
        # Record audio
        print("Listening...")
        recording, samplerate = record_audio()
        
        # Check if we got audio
        if len(recording) == 0:
            print("No audio detected. Please try again.")
            continue
        
        # Process the audio
        audio_input = AudioInput(buffer=recording)
        
        # Create workflow and pipeline
        workflow = create_voice_workflow(triage_agent)
        pipeline = create_voice_pipeline(workflow, tts_profile=tts_profile)
        
        # Process the audio
        with trace("ACME Shop Voice Assistant"):
            result = await pipeline.run(audio_input)
        
        # Collect audio chunks from response
        print("Assistant is responding...")
        response_chunks = []
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                response_chunks.append(event.data)
        
        # Combine and play audio
        if response_chunks:
            response_audio = np.concatenate(response_chunks, axis=0)
            play_audio(response_audio, samplerate=samplerate)
        
        print("---")

async def main():
    """Main function."""
    # Set the OpenAI API key
    if OPENAI_API_KEY:
        set_default_openai_key(OPENAI_API_KEY)
    else:
        print("Warning: OPENAI_API_KEY not set. Please set it in your .env file.")
        return
    
    # Check for command line arguments
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "text":
            await run_text_assistant()
            return
        elif mode == "voice":
            profile = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_VOICE_PROFILE
            await run_voice_assistant(tts_profile=profile)
            return
    
    # Display menu if no command line arguments
    from src.config.settings import DEFAULT_MODE, DEFAULT_VOICE_PROFILE
    
    print("ACME Shop Assistant")
    print("1. Text Mode")
    print("2. Voice Mode (Default Voice)")
    print("3. Voice Mode (Upbeat Voice)")
    print("4. Voice Mode (Character Voice)")
    print("5. Exit")
    
    # Get user choice
    choice = input("Enter your choice (1-5): ")
    
    if choice == "1":
        await run_text_assistant()
    elif choice == "2":
        await run_voice_assistant(tts_profile="default")
    elif choice == "3":
        await run_voice_assistant(tts_profile="upbeat")
    elif choice == "4":
        await run_voice_assistant(tts_profile="character")
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    asyncio.run(main())