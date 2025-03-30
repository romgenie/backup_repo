"""
Main entry point for the voice assistant application.
"""
import asyncio
import json
import logging
import re
from agents import Runner, trace, set_default_openai_key
from src.config.settings import OPENAI_API_KEY
from src.agents import create_triage_agent
from src.utils.vector_store_manager import initialize_vector_store, get_vector_store_info

# Configure logging for context debugging
logging.basicConfig(level=logging.INFO)
context_logger = logging.getLogger("context_debug")
context_logger.setLevel(logging.INFO)

# Try to import voice features, but make them optional
try:
    import numpy as np
    import json
    import os
    import tempfile
    import soundfile as sf
    from src.utils.audio import record_audio, play_audio
    
    # Only need basic Runner for our simplified approach
    from agents import Runner
    
    VOICE_AVAILABLE = True
    print("Basic voice features are available.")
except ImportError as e:
    print(f"Voice features not available. Error: {str(e)}")
    print("Install with: pip install numpy sounddevice soundfile")
    VOICE_AVAILABLE = False

async def run_text_assistant():
    """Run the assistant in text mode with context preservation for workflows."""
    # Create the triage agent
    triage_agent = create_triage_agent(use_voice_optimization=False)
    
    # Use personalized name from settings
    from src.config.settings import ORGANIZATION_NAME, ORGANIZATION_DESCRIPTION, ASSISTANT_NAME
    print(f"{ORGANIZATION_NAME} - Text Mode")
    print("Type 'exit' to quit")
    print("---")
    
    # Initialize conversation for context preservation
    conversation_id = None
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.lower() == "exit":
            from src.config.settings import FAREWELL_MESSAGE
            print(f"{FAREWELL_MESSAGE}")
            break
        
        # Save to history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Enhanced special handling for affirmative responses to vector store questions
        is_affirmative = user_input.lower().strip() in ["yes", "yes please", "sure", "ok", "okay", "yep", "yeah", "yup", "do it", "go ahead"]
        last_assistant_message = next((msg for msg in reversed(conversation_history) if msg["role"] == "assistant"), None) if conversation_history else None
        
        # Check for vector store questions or suggestions
        vector_store_question = False
        if last_assistant_message:
            last_msg_content = last_assistant_message["content"].lower()
            vector_store_question = any(phrase in last_msg_content for phrase in 
                                    ["add to vector store", "add to your vector store", "add them to vector store", 
                                     "add them to your vector store", "add to a vector store", "create a new vector store",
                                     "would you like me to add", "should i add"])
        
        if is_affirmative and vector_store_question and last_assistant_message:
            context_logger.info("Detected affirmative response to vector store question")
            
            # Extract paper information from the conversation history
            paper_info = []
            paper_titles = []
            
            # Look through previous messages for paper information
            for msg in conversation_history:
                if msg["role"] == "assistant" and "downloaded" in msg["content"].lower():
                    # Try to extract paper information
                    content = msg["content"]
                    
                    # Look for PDF file paths
                    pdf_paths = re.findall(r'/[\w/\-\.]+\.pdf', content)
                    
                    # Look for paper titles (usually in quotes or after "paper" or "titled")
                    title_patterns = [
                        r'["\'](.*?)["\']',  # Anything in quotes
                        r'paper (?:titled|called) ["\'](.*?)["\']',  # After "paper titled" or "paper called"
                        r'downloaded ["\'](.*?)["\']',  # After "downloaded"
                        r'paper ["\']?(.*?)["\']? by',  # Before "by"
                    ]
                    
                    titles = []
                    for pattern in title_patterns:
                        titles.extend(re.findall(pattern, content))
                    
                    # If we found paths but no titles, try to extract titles from paths
                    if pdf_paths and not titles:
                        for path in pdf_paths:
                            # Extract filename and remove extension
                            filename = re.search(r'/([^/]+)\.pdf$', path)
                            if filename:
                                titles.append(filename.group(1).replace('_', ' '))
                    
                    # If we found titles, remember them for later
                    paper_titles.extend(titles)
                    
                    # If we found both paths and titles, combine them
                    if pdf_paths and titles:
                        # Pair paths with titles as best we can
                        for i, path in enumerate(pdf_paths):
                            if i < len(titles):
                                paper_info.append({
                                    "title": titles[i],
                                    "file_path": path,
                                    "authors": "N/A"  # Default when we can't extract authors
                                })
            
            # Check what kind of papers were discussed in the last assistant message
            last_msg = last_assistant_message["content"].lower()
            paper_type = None
            
            if "karma" in last_msg:
                paper_type = "KARMA"
            elif "transform" in last_msg:
                paper_type = "transformer"
            elif "bert" in last_msg:
                paper_type = "BERT"
            elif "attention" in last_msg:
                paper_type = "Attention"
            elif paper_titles:
                # If we found paper titles, mention them specifically
                paper_type = ", ".join(paper_titles[:3])
                if len(paper_titles) > 3:
                    paper_type += f", and {len(paper_titles) - 3} more"
            
            # Create enhanced response based on extracted information
            if paper_info:
                # If we have detailed paper information with file paths, include it in a machine-readable format
                # This helps the ArXiv research agent properly handoff to Vector Store agent
                context_logger.info(f"Found {len(paper_info)} papers with file paths")
                user_input = f"Yes, please add these papers to the vector store: {json.dumps(paper_info)}"
            elif paper_type:
                # If we know the paper type but not details, be more specific than just "yes please"
                context_logger.info(f"Using paper type: {paper_type}")
                user_input = f"Yes, please add the {paper_type} papers to the vector store."
            else:
                # Fallback to generic but still explicit instruction
                context_logger.info("Using generic paper reference")
                user_input = "Yes, please add these papers to the vector store that you just mentioned."
            
            # Update the history with our enhanced response
            conversation_history[-1]["content"] = user_input
            context_logger.info(f"Enhanced affirmative response to: {user_input}")
        
        # Process the input with conversation context
        with trace(f"{ORGANIZATION_NAME} Text Assistant"):
            # If we have a conversation ID, use it to maintain context
            if conversation_id:
                result = await Runner.run(
                    triage_agent, 
                    user_input, 
                    conversation_id=conversation_id
                )
            else:
                # First message in conversation
                result = await Runner.run(triage_agent, user_input)
                # Store conversation ID for future messages if available
                # Note: Some versions of the SDK may not return conversation_id
                if hasattr(result, 'conversation_id'):
                    conversation_id = result.conversation_id
        
        # Display the result
        from src.config.settings import ASSISTANT_NAME
        print(f"{ASSISTANT_NAME}: {result.final_output}")
        print("---")
        
        # Add to conversation history for future context
        conversation_history.append({"role": "assistant", "content": result.final_output})

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
    
    # Initialize conversation for context preservation
    conversation_id = None
    
    # Track conversation context - especially for debugging
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
                
                # Add to conversation history for tracking
                conversation_history.append({"role": "user", "content": user_text})
                
                # Process with the triage agent (text mode)
                with trace(f"{ORGANIZATION_NAME} Voice Input Assistant"):
                    # Log the context information for debugging
                    context_logger.info(f"Context before run - conversation_id: {conversation_id}, " +
                                       f"history_length: {len(conversation_history)}, " +
                                       f"last_user_message: {conversation_history[-1]['content'] if conversation_history else None}")
                    
                    # Include the previous message context for OpenAI's system
                    # This adds extra redundancy in case the conversation_id mechanism fails
                    messages = []
                    
                    # Add the last few messages from history to provide redundant context
                    # This helps ensure context even if the OpenAI conversation_id has issues
                    if len(conversation_history) >= 3:
                        # Get the last couple of turns (up to 2 turns = 4 messages)
                        extra_context = conversation_history[-4:]
                        for msg in extra_context:
                            if msg["role"] != "system":  # Skip system messages
                                messages.append({
                                    "role": msg["role"],
                                    "content": msg["content"]
                                })
                    
                    # Add the current user message
                    messages.append({
                        "role": "user",
                        "content": user_text
                    })
                    
                    # Log the constructed messages for debugging
                    context_logger.info(f"Constructed {len(messages)} messages for additional context")
                    
                    # If we have a conversation ID, use it to maintain context
                    if conversation_id:
                        context_logger.info(f"Using existing conversation_id: {conversation_id}")
                        
                        try:
                            # Ensure we explicitly pass conversation context
                            result = await Runner.run(
                                triage_agent, 
                                messages if len(messages) > 1 else user_text,  # Use messages if we have context
                                conversation_id=conversation_id
                            )
                        except Exception as e:
                            # If there's an error with the conversation_id approach, try without it
                            context_logger.warning(f"Error with conversation_id, trying without: {str(e)}")
                            result = await Runner.run(triage_agent, messages if len(messages) > 1 else user_text)
                            # Reset conversation_id to get a new one
                            conversation_id = None
                    else:
                        # First message in conversation
                        result = await Runner.run(triage_agent, user_text)
                    
                    # Log all attributes of the result for debugging
                    context_logger.info(f"Result attributes: {dir(result)}")
                    
                    # Store conversation ID for future messages if available
                    if hasattr(result, 'conversation_id'):
                        context_logger.info(f"Found conversation_id attribute: {result.conversation_id}")
                        if result.conversation_id:
                            conversation_id = result.conversation_id
                            context_logger.info(f"Using conversation_id: {conversation_id}")
                        else:
                            context_logger.warning("conversation_id attribute exists but is None")
                    else:
                        context_logger.warning("Result does not have conversation_id attribute")
                        
                        # Try alternative attribute names
                        for attr_name in ['thread_id', 'context_id', 'id']:
                            if hasattr(result, attr_name):
                                context_logger.info(f"Found alternative id attribute: {attr_name}={getattr(result, attr_name)}")
                                conversation_id = getattr(result, attr_name)
                    
                    # Log the presence of message history if available
                    if hasattr(result, 'messages'):
                        if result.messages:
                            context_logger.info(f"Result contains {len(result.messages)} messages in history")
                            # Log a sample of the messages
                            if len(result.messages) > 0:
                                context_logger.info(f"Sample message: {result.messages[0]}")
                        else:
                            context_logger.warning("messages attribute exists but is empty")
                    else:
                        context_logger.warning("Result does not have messages attribute")
                    
                    # Log the context after running
                    context_logger.info(f"Context after run - conversation_id: {conversation_id}, " +
                                       f"has_response: {hasattr(result, 'final_output')}")
                
                # Add response to conversation history for tracking
                conversation_history.append({"role": "assistant", "content": result.final_output})
                
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
    Run the assistant in voice mode with direct text-to-speech.
    
    Args:
        tts_profile: The TTS profile to use (kept for API compatibility)
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
    
    # Import OpenAI client for transcription and TTS
    from openai import OpenAI
    from src.config.settings import OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Include logging message for transparency
    context_logger.info("Using simplified speech interface (without VoicePipeline)")
    print("Using simplified speech interface. Context will be preserved between turns.")
    
    # Initialize conversation for context preservation
    conversation_id = None
    
    # Track conversation history for redundant context
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
                
                # Add to conversation history for tracking
                conversation_history.append({"role": "user", "content": user_text})
                
                # Prepare message context
                messages = []
                
                # Add previous messages for context
                if len(conversation_history) > 1:
                    # Include up to 10 previous messages
                    history_size = min(10, len(conversation_history) - 1)
                    context_messages = conversation_history[:-1][-history_size:]
                    context_logger.info(f"Adding {len(context_messages)} messages for context")
                    
                    for msg in context_messages:
                        if msg["role"] != "system":  # Skip system messages
                            messages.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                
                # Enhanced special handling for affirmative responses to vector store questions
                is_affirmative = user_text.lower().strip() in ["yes", "yes please", "sure", "ok", "okay", "yep", "yeah", "yup", "do it", "go ahead"]
                last_assistant_message = next((msg for msg in reversed(conversation_history) if msg["role"] == "assistant"), None)
                
                # Check for vector store questions or suggestions
                vector_store_question = False
                if last_assistant_message:
                    last_msg_content = last_assistant_message["content"].lower()
                    vector_store_question = any(phrase in last_msg_content for phrase in 
                                            ["add to vector store", "add to your vector store", "add them to vector store", 
                                             "add them to your vector store", "add to a vector store", "create a new vector store",
                                             "would you like me to add", "should i add"])
                
                # Special affirmative response handling
                if is_affirmative and vector_store_question and last_assistant_message:
                    context_logger.info("Detected affirmative response to vector store question - using special handling")
                    
                    # Extract paper information from the conversation history
                    paper_info = []
                    paper_titles = []
                    
                    # Look through previous messages for paper information
                    for msg in conversation_history:
                        if msg["role"] == "assistant" and "downloaded" in msg["content"].lower():
                            # Try to extract paper information
                            content = msg["content"]
                            
                            # Look for PDF file paths
                            pdf_paths = re.findall(r'/[\w/\-\.]+\.pdf', content)
                            
                            # Look for paper titles (usually in quotes or after "paper" or "titled")
                            title_patterns = [
                                r'["\'](.*?)["\']',  # Anything in quotes
                                r'paper (?:titled|called) ["\'](.*?)["\']',  # After "paper titled" or "paper called"
                                r'downloaded ["\'](.*?)["\']',  # After "downloaded"
                                r'paper ["\']?(.*?)["\']? by',  # Before "by"
                            ]
                            
                            titles = []
                            for pattern in title_patterns:
                                titles.extend(re.findall(pattern, content))
                            
                            # If we found paths but no titles, try to extract titles from paths
                            if pdf_paths and not titles:
                                for path in pdf_paths:
                                    # Extract filename and remove extension
                                    filename = re.search(r'/([^/]+)\.pdf$', path)
                                    if filename:
                                        titles.append(filename.group(1).replace('_', ' '))
                            
                            # If we found titles, remember them for later
                            paper_titles.extend(titles)
                            
                            # If we found both paths and titles, combine them
                            if pdf_paths and titles:
                                # Pair paths with titles as best we can
                                for i, path in enumerate(pdf_paths):
                                    if i < len(titles):
                                        paper_info.append({
                                            "title": titles[i],
                                            "file_path": path,
                                            "authors": "N/A"  # Default when we can't extract authors
                                        })
                    
                    # Check what kind of papers were discussed in the last assistant message
                    last_msg = last_assistant_message["content"].lower()
                    paper_type = None
                    
                    if "karma" in last_msg:
                        paper_type = "KARMA"
                    elif "transform" in last_msg:
                        paper_type = "transformer"
                    elif "bert" in last_msg:
                        paper_type = "BERT"
                    elif "attention" in last_msg:
                        paper_type = "Attention"
                    elif paper_titles:
                        # If we found paper titles, mention them specifically
                        paper_type = ", ".join(paper_titles[:3])
                        if len(paper_titles) > 3:
                            paper_type += f", and {len(paper_titles) - 3} more"
                    
                    # Create enhanced response based on extracted information
                    if paper_info:
                        # If we have detailed paper information with file paths, include it in a machine-readable format
                        # This helps the ArXiv research agent properly handoff to Vector Store agent
                        context_logger.info(f"Found {len(paper_info)} papers with file paths")
                        papers_context = f"Yes, please add these papers to the vector store: {json.dumps(paper_info)}"
                    elif paper_type:
                        # If we know the paper type but not details, be more specific than just "yes please"
                        context_logger.info(f"Using paper type: {paper_type}")
                        papers_context = f"Yes, please add the {paper_type} papers to the vector store."
                    else:
                        # Fallback to generic but still explicit instruction
                        context_logger.info("Using generic paper reference")
                        papers_context = "Yes, please add these papers to the vector store that you just mentioned."
                    
                    context_logger.info(f"Replacing '{user_text}' with more explicit: '{papers_context}'")
                    
                    # Replace the simple affirmative with a more explicit message
                    user_text = papers_context
                    
                    # Also update the history
                    conversation_history[-1]["content"] = papers_context
                
                # Add current message
                messages.append({
                    "role": "user",
                    "content": user_text
                })
                
                # Process with the triage agent (text mode)
                with trace(f"{ORGANIZATION_NAME} Voice Assistant"):
                    context_logger.info(f"Context before run - conversation_id: {conversation_id}")
                    
                    # If we have a conversation ID, use it to maintain context
                    if conversation_id:
                        try:
                            result = await Runner.run(
                                triage_agent, 
                                messages if len(messages) > 1 else user_text,  # Use messages if we have context
                                conversation_id=conversation_id
                            )
                        except Exception as e:
                            # If there's an error with the conversation_id approach, try without it
                            context_logger.warning(f"Error with conversation_id, trying without: {str(e)}")
                            result = await Runner.run(triage_agent, messages if len(messages) > 1 else user_text)
                            # Reset conversation_id to get a new one
                            conversation_id = None
                    else:
                        # First message in conversation
                        if len(messages) > 1:
                            result = await Runner.run(triage_agent, messages)
                        else:
                            result = await Runner.run(triage_agent, user_text)
                    
                    # Store conversation ID for future messages
                    if hasattr(result, 'conversation_id') and result.conversation_id:
                        conversation_id = result.conversation_id
                        context_logger.info(f"Got conversation_id: {conversation_id}")
                    else:
                        context_logger.warning("No conversation_id in result")
                        
                    # Log context after running    
                    context_logger.info(f"Context after run - conversation_id: {conversation_id}")
                
                # Add response to conversation history for tracking
                conversation_history.append({"role": "assistant", "content": result.final_output})
                
                # Convert response to speech with OpenAI TTS API
                print("Assistant is responding...")
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                        tts_filename = temp_audio.name
                    
                    # Generate speech
                    response = client.audio.speech.create(
                        model="tts-1",
                        voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
                        input=result.final_output
                    )
                    
                    # Save to file
                    response.write_to_file(tts_filename)
                    
                    # Play the audio
                    data, audio_samplerate = sf.read(tts_filename)
                    play_audio(data, audio_samplerate)
                    
                    # Clean up
                    try:
                        os.unlink(tts_filename)
                    except Exception:
                        pass
                        
                except Exception as e:
                    context_logger.error(f"Error with TTS: {str(e)}")
                    # Just display the text if TTS fails
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