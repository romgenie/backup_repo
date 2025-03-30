"""
Main entry point for the voice assistant application.
"""
import asyncio
from agents import Runner, trace, set_default_openai_key
from src.config.settings import OPENAI_API_KEY
from src.agents import create_triage_agent
from src.utils.vector_store_manager import initialize_vector_store, get_vector_store_info

# Try to import voice features, but make them optional
try:
    import numpy as np
    from agents.voice import AudioInput
    from src.voice_pipeline import create_voice_workflow, create_voice_pipeline
    from src.utils.audio import record_audio, play_audio
    VOICE_AVAILABLE = True
except ImportError:
    print("Voice features not available. Install with: pip install 'openai-agents[voice]'")
    VOICE_AVAILABLE = False

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
    if not VOICE_AVAILABLE:
        print("Voice features are not available.")
        print("Please install them with: pip install 'openai-agents[voice]'")
        return
    
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

async def check_vector_store() -> bool:
    """
    Check if a valid vector store ID exists.
    
    Returns:
        bool: True if a valid vector store ID exists, False otherwise
    """
    from src.config.settings import VECTOR_STORE_ID
    return VECTOR_STORE_ID is not None and VECTOR_STORE_ID != "your_vector_store_id_here"

# This function is now in src.utils.vector_store_manager

async def setup_vector_store(force: bool = False):
    """
    Setup vector store for knowledge base.
    
    Args:
        force: If True, create a new vector store even if one already exists
        
    Returns:
        str: Vector store ID if successful, None otherwise
    """
    if not force and await check_vector_store():
        print("Vector store already configured.")
        return
        
    print("Setting up vector store...")
    try:
        vector_store_id = await initialize_vector_store(force_create=force)
        if vector_store_id:
            print(f"Vector store initialized with ID: {vector_store_id}")
        else:
            print("No vector store was created or configured.")
        return vector_store_id
    except Exception as e:
        print(f"Error setting up vector store: {e}")
        return None


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
        if sys.argv[1] == "setup":
            # Setup vector store (force creation even if one exists)
            await setup_vector_store(force=True)
            return
        elif sys.argv[1] == "info":
            # Display vector store information
            info = await get_vector_store_info()
            print(info)
            return
        elif sys.argv[1] == "text":
            # Run text assistant without forcing vector store creation
            if not await check_vector_store():
                print("Warning: No vector store configured. Knowledge agent will have limited functionality.")
                print("Run 'python -m src.main setup' to set up a vector store.")
                print("")
            
            await run_text_assistant()
            return
        elif sys.argv[1] == "voice":
            # Check if voice is available
            if not VOICE_AVAILABLE:
                print("Voice features are not available.")
                print("Please install them with: pip install 'openai-agents[voice]'")
                return
                
            # Run voice assistant without forcing vector store creation
            if not await check_vector_store():
                print("Warning: No vector store configured. Knowledge agent will have limited functionality.")
                print("Run 'python -m src.main setup' to set up a vector store.")
                print("")
                
            # Get voice profile
            from src.config.settings import DEFAULT_VOICE_PROFILE
            profile = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_VOICE_PROFILE
            await run_voice_assistant(tts_profile=profile)
            return
    
    # Display menu if no command line arguments
    from src.config.settings import DEFAULT_MODE, DEFAULT_VOICE_PROFILE
    
    # Check for valid vector store
    has_vector_store = await check_vector_store()
    
    print("ACME Shop Assistant")
    print("1. Text Mode")
    
    if VOICE_AVAILABLE:
        print("2. Voice Mode (Default Voice)")
        print("3. Voice Mode (Upbeat Voice)")
        print("4. Voice Mode (Character Voice)")
    else:
        print("Voice modes (2-4) not available - install with: pip install 'openai-agents[voice]'")
    
    print("5. Setup Vector Store")
    print("6. Vector Store Information")
    print("7. Exit")
    
    if not has_vector_store:
        print("\nNote: No vector store is configured. Knowledge agent will have limited functionality.")
        print("Select option 5 to set up a vector store.")
    
    # Get user choice
    choice = input("\nEnter your choice (1-7): ")
    
    if choice == "1":
        # Run without forcing vector store creation
        await run_text_assistant()
    elif choice == "2":
        # Run without forcing vector store creation
        await run_voice_assistant(tts_profile="default")
    elif choice == "3":
        # Run without forcing vector store creation
        await run_voice_assistant(tts_profile="upbeat")
    elif choice == "4":
        # Run without forcing vector store creation
        await run_voice_assistant(tts_profile="character")
    elif choice == "5":
        # Explicitly setup vector store (with force=True to allow recreation)
        await setup_vector_store(force=True)
    elif choice == "6":
        # Display vector store information
        info = await get_vector_store_info()
        print(info)
    elif choice == "7":
        print("Exiting...")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    asyncio.run(main())