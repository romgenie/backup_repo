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
    
    # Use personalized name from settings
    from src.config.settings import ORGANIZATION_NAME, ORGANIZATION_DESCRIPTION, ASSISTANT_NAME
    print(f"{ORGANIZATION_NAME} - Text Mode")
    print("Type 'exit' to quit")
    print("---")
    
    # Initialize conversation history to maintain context between turns
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.lower() == "exit":
            from src.config.settings import FAREWELL_MESSAGE
            print(f"{FAREWELL_MESSAGE}")
            break
        
        # Add user message to conversation history
        if conversation_history:
            # If we have existing history, add the new user message
            conversation_input = conversation_history + [{"role": "user", "content": user_input}]
        else:
            # For the first message, just use the text directly
            conversation_input = user_input
        
        # Process the input with the full conversation history
        with trace(f"{ORGANIZATION_NAME} Text Assistant"):
            result = await Runner.run(triage_agent, conversation_input)
        
        # Update conversation history with the result for the next turn
        conversation_history = result.to_input_list()
        
        # Display the result
        from src.config.settings import ASSISTANT_NAME
        print(f"{ASSISTANT_NAME}: {result.final_output}")
        print("---")

async def run_voice_input_text_output():
    """
    Run the assistant in voice input, text output mode.
    User speaks, but assistant responds with text only.
    """
    if not VOICE_AVAILABLE:
        print("Voice features are not available.")
        print("Please install them with: pip install 'openai-agents[voice]'")
        return
    
    # Create the triage agent with voice optimization
    triage_agent = create_triage_agent(use_voice_optimization=False)
    
    # Use personalized name from settings
    from src.config.settings import ORGANIZATION_NAME, ORGANIZATION_DESCRIPTION, ASSISTANT_NAME
    print(f"{ORGANIZATION_NAME} - Voice Input, Text Output Mode")
    print("Press Enter to start/stop recording")
    print("Type 'exit' to quit")
    print("---")
    
    # Import OpenAI client for transcription
    from openai import OpenAI
    from src.config.settings import OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Initialize conversation history to maintain context between turns
    conversation_history = []
    
    while True:
        # Check if user wants to exit
        cmd = input("Press Enter to speak (or type 'exit' to quit): ")
        if cmd.lower() == "exit":
            from src.config.settings import FAREWELL_MESSAGE
            print(f"{FAREWELL_MESSAGE}")
            break
        
        # Record audio
        print("Listening...")
        recording, samplerate = record_audio()
        
        # Check if we got audio
        if len(recording) == 0:
            print("No audio detected. Please try again.")
            continue
        
        # Convert audio to WAV file format for OpenAI API
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_filename = temp_audio.name
            sf.write(temp_filename, recording, samplerate)
        
        try:
            # Transcribe audio using OpenAI Whisper API
            with open(temp_filename, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            user_text = transcript.text
            
            # If we got a transcription, send it to the agent
            if user_text.strip():
                print(f"You said: {user_text}")
                
                # Add user message to conversation history
                if conversation_history:
                    # If we have existing history, add the new user message
                    conversation_input = conversation_history + [{"role": "user", "content": user_text}]
                else:
                    # For the first message, just use the text directly
                    conversation_input = user_text
                
                # Process with the triage agent (text mode) including full conversation history
                with trace(f"{ORGANIZATION_NAME} Voice Input Assistant"):
                    result = await Runner.run(triage_agent, conversation_input)
                
                # Update conversation history with the result for the next turn
                conversation_history = result.to_input_list()
                
                # Display the text response
                print(f"{ASSISTANT_NAME}: {result.final_output}")
            else:
                print("No speech detected. Please try again.")
                
        except Exception as e:
            print(f"Error processing speech: {str(e)}")
        
        # Clean up temp file
        import os
        try:
            os.unlink(temp_filename)
        except:
            pass
            
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
    
    # Use personalized name from settings
    from src.config.settings import ORGANIZATION_NAME, ORGANIZATION_DESCRIPTION, ASSISTANT_NAME
    print(f"{ORGANIZATION_NAME} - Voice Mode")
    print("Press Enter to start/stop recording")
    print("Type 'exit' to quit")
    print("---")
    
    # Initialize conversation history and workflow/pipeline
    conversation_history = None
    workflow = None
    pipeline = None
    
    while True:
        # Check if user wants to exit
        cmd = input("Press Enter to speak (or type 'exit' to quit): ")
        if cmd.lower() == "exit":
            from src.config.settings import FAREWELL_MESSAGE
            print(f"{FAREWELL_MESSAGE}")
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
        
        # Create workflow and pipeline if this is the first interaction
        # or if we need to recreate them
        if workflow is None or pipeline is None:
            workflow = create_voice_workflow(triage_agent)
            pipeline = create_voice_pipeline(workflow, tts_profile=tts_profile)
        
        # Process the audio with conversation history if available
        with trace(f"{ORGANIZATION_NAME} Voice Assistant"):
            if conversation_history is not None:
                # Include conversation history in the run
                result = await pipeline.run(audio_input, conversation_history=conversation_history)
            else:
                # First interaction, no history yet
                result = await pipeline.run(audio_input)
        
        # Update conversation history for the next turn
        conversation_history = result.conversation_history
        
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
    from src.utils.vector_store_cache import has_valid_vector_store
    return has_valid_vector_store()

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


async def init_vector_store_cache():
    """Initialize the vector store cache at startup"""
    # Import the vector store cache utilities
    from src.utils.vector_store_cache import auto_set_active_store, refresh_cache_manually
    
    # Auto-set an active store if none is set
    auto_set_active_store()
    
    # We'll do a full refresh in the background 
    # This is non-blocking so doesn't impact startup time
    import asyncio
    
    async def background_refresh():
        # Run in the background to refresh cache without blocking startup
        try:
            # Directly refresh the cache using our manual function
            # This avoids using the agent which might have issues with function_tool calling
            with trace("Vector Store Cache Init"):
                refresh_cache_manually()
        except Exception:
            # Silently fail if something goes wrong
            pass
            
    # Start the background task
    asyncio.create_task(background_refresh())

async def main():
    """Main function."""
    # Set the OpenAI API key
    if OPENAI_API_KEY:
        set_default_openai_key(OPENAI_API_KEY)
    else:
        print("Warning: OPENAI_API_KEY not set. Please set it in your .env file.")
        return
        
    # Initialize the vector store cache
    await init_vector_store_cache()
    
    # Check for auto-start mode
    from src.config.settings import DEFAULT_MODE, DEFAULT_VOICE_PROFILE, AUTO_START
    
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
            # Run text assistant with optional vector store warnings
            from src.config.settings import DISPLAY_VECTOR_STORE_WARNINGS
            if not await check_vector_store() and DISPLAY_VECTOR_STORE_WARNINGS:
                print("Warning: No vector store configured. Document search functionality will be limited.")
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
                
            # Run voice assistant with optional vector store warnings
            from src.config.settings import DISPLAY_VECTOR_STORE_WARNINGS
            if not await check_vector_store() and DISPLAY_VECTOR_STORE_WARNINGS:
                print("Warning: No vector store configured. Document search functionality will be limited.")
                print("Run 'python -m src.main setup' to set up a vector store.")
                print("")
            
            # Get voice profile
            from src.config.settings import DEFAULT_VOICE_PROFILE
            profile = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_VOICE_PROFILE
            await run_voice_assistant(tts_profile=profile)
            return
    
    # Auto-start if enabled
    if AUTO_START:
        if DEFAULT_MODE == "text":
            await run_text_assistant()
            return
        elif DEFAULT_MODE == "voice":
            if VOICE_AVAILABLE:
                await run_voice_assistant(tts_profile=DEFAULT_VOICE_PROFILE)
                return
            else:
                print("Voice mode not available. Falling back to text mode.")
                await run_text_assistant()
                return
                
    # Display menu if no command line arguments or auto-start is disabled
    from src.config.settings import (
        ORGANIZATION_NAME, ORGANIZATION_DESCRIPTION, ASSISTANT_NAME,
        DISPLAY_TEXT_MODE, DISPLAY_VOICE_MODES, DISPLAY_VECTOR_STORE_MODES
    )
    
    # Check for valid vector store
    has_vector_store = await check_vector_store()
    
    print(f"{ORGANIZATION_NAME}")
    print(f"{ORGANIZATION_DESCRIPTION}")
    
    menu_items = []
    menu_item_count = 1
    
    # Add text mode if enabled
    if DISPLAY_TEXT_MODE:
        menu_items.append({"id": menu_item_count, "name": "Text Mode", "action": "text"})
        print(f"{menu_item_count}. Text Mode")
        menu_item_count += 1
    
    # Add voice modes if enabled and available
    if DISPLAY_VOICE_MODES:
        if VOICE_AVAILABLE:
            voice_options = [
                {"id": menu_item_count, "name": "Voice Mode (Default Voice)", "action": "voice_default"},
                {"id": menu_item_count + 1, "name": "Voice Mode (Upbeat Voice)", "action": "voice_upbeat"},
                {"id": menu_item_count + 2, "name": "Voice Mode (Character Voice)", "action": "voice_character"},
                {"id": menu_item_count + 3, "name": "Voice Input, Text Output", "action": "voice_text_only"}
            ]
            
            for option in voice_options:
                menu_items.append(option)
                print(f"{option['id']}. {option['name']}")
                menu_item_count += 1
        else:
            print("Voice modes not available - install with: pip install 'openai-agents[voice]'")
    
    # Add vector store options if enabled
    if DISPLAY_VECTOR_STORE_MODES:
        vector_options = [
            {"id": menu_item_count, "name": "Setup Vector Store", "action": "setup_vector"},
            {"id": menu_item_count + 1, "name": "Vector Store Information", "action": "vector_info"}
        ]
        
        for option in vector_options:
            menu_items.append(option)
            print(f"{option['id']}. {option['name']}")
            menu_item_count += 1
    
    # Add exit option
    exit_option = {"id": menu_item_count, "name": "Exit", "action": "exit"}
    menu_items.append(exit_option)
    print(f"{exit_option['id']}. Exit")
    
    # Show vector store warning if enabled
    from src.config.settings import DISPLAY_VECTOR_STORE_WARNINGS
    if not has_vector_store and DISPLAY_VECTOR_STORE_MODES and DISPLAY_VECTOR_STORE_WARNINGS:
        print("\nNote: No vector store is configured. Document search functionality will be limited.")
        for item in menu_items:
            if item["action"] == "setup_vector":
                print(f"Select option {item['id']} to set up a vector store.")
                break
    
    # Get user choice
    max_choice = menu_items[-1]["id"]
    choice = input(f"\nEnter your choice (1-{max_choice}): ")
    
    try:
        choice_num = int(choice)
        selected_option = next((item for item in menu_items if item["id"] == choice_num), None)
        
        if not selected_option:
            print("Invalid choice.")
            return
            
        action = selected_option["action"]
        
        if action == "text":
            # Run text assistant
            await run_text_assistant()
        elif action == "voice_default":
            # Run voice assistant with default profile
            await run_voice_assistant(tts_profile="default")
        elif action == "voice_upbeat":
            # Run voice assistant with upbeat profile
            await run_voice_assistant(tts_profile="upbeat")
        elif action == "voice_character":
            # Run voice assistant with character profile
            await run_voice_assistant(tts_profile="character")
        elif action == "voice_text_only":
            # Run voice input with text output mode
            await run_voice_input_text_output()
        elif action == "setup_vector":
            # Explicitly setup vector store
            await setup_vector_store(force=True)
        elif action == "vector_info":
            # Display vector store information
            info = await get_vector_store_info()
            print(info)
        elif action == "exit":
            from src.config.settings import FAREWELL_MESSAGE
            print(f"{FAREWELL_MESSAGE}")
        else:
            print("Invalid action.")
    except ValueError:
        print("Please enter a valid number.")

if __name__ == "__main__":
    asyncio.run(main())