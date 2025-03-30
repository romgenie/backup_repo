"""
Triage agent implementation.
"""
import re
import json
import logging
from typing import Callable, Awaitable, Any, Union, Optional, Dict, List
from agents import Agent, Handoff, HandoffInputData
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from src.agents.triage.prompts import BASE_INSTRUCTIONS
from src.agents.vector_store.agent import create_vector_store_agent
from src.agents.arxiv_research.agent import create_arxiv_research_agent

from agents import handoff
from agents.extensions import handoff_filters

# Configure logging
logger = logging.getLogger("triage_agent")
logger.setLevel(logging.INFO)

def arxiv_to_vector_handoff_filter(input_data: HandoffInputData) -> HandoffInputData:
    """
    Custom filter that enhances handoffs from ArXiv to Vector Store agent.
    This filter examines the conversation context to identify paper information
    and ensures it's properly transferred to the Vector Store Agent.
    
    Args:
        input_data: The original handoff input data
        
    Returns:
        HandoffInputData: The enhanced handoff input data
    """
    # Log the handoff for debugging
    logger.info(f"Processing handoff from ArXiv to Vector Store agent")
    logger.info(f"Number of messages in handoff: {len(input_data.messages)}")
    
    # Check if this is a simple "yes please" response to a vector store question
    if len(input_data.messages) >= 2:
        last_user_message = next((msg for msg in reversed(input_data.messages) 
                                if msg.get("role") == "user"), None)
        last_assistant_message = next((msg for msg in reversed(input_data.messages) 
                                    if msg.get("role") == "assistant"), None)
        
        # Detect simple affirmative responses
        if (last_user_message and last_assistant_message and
            any(resp in last_user_message.get("content", "").lower() for resp in 
                ["yes", "yes please", "sure", "ok", "okay", "yes do it", "please do"])):
            
            # Check if the last assistant message was about adding to vector store
            assistant_content = last_assistant_message.get("content", "").lower()
            if any(phrase in assistant_content for phrase in 
                  ["vector store", "add to your vector store", "add them to vector store"]):
                
                logger.info("Detected affirmative response to vector store question")
                
                # Create a more explicit user message based on context
                # Find paper information in the conversation history
                paper_titles = []
                paper_paths = []
                
                # Look through ALL previous messages for paper information
                for msg in input_data.messages:
                    if msg.get("role") == "assistant":
                        # Try to extract paper information from any assistant message
                        content = msg.get("content", "")
                        
                        # Look for PDF file paths with more patterns
                        paths = re.findall(r'/[\w/\-\.]+\.pdf', content)
                        paper_paths.extend(paths)
                        
                        # Look for paper titles in quotes with multiple patterns
                        title_patterns = [
                            r'["\'](.*?)["\']',  # Anything in quotes
                            r'paper (?:titled|called) ["\'](.*?)["\']',  # After "paper titled" or "paper called"
                            r'downloaded ["\'](.*?)["\']',  # After "downloaded"
                            r'paper ["\']?(.*?)["\']? by',  # Before "by"
                        ]
                        
                        for pattern in title_patterns:
                            found_titles = re.findall(pattern, content)
                            paper_titles.extend(found_titles)
                            
                # Additionally, look for papers mentioned in the immediate handoff message
                # This is typically from ArXiv agent's transfer_to_VectorStoreAgent call
                last_assistant_content = last_assistant_message.get("content", "").lower() if last_assistant_message else ""
                
                # Check if this looks like an explicit handoff message
                if any(phrase in last_assistant_content for phrase in 
                       ["add to vector store", "transfer", "add paper", "add these paper", "add the paper"]):
                    # Extract paper paths and titles from this message with even more patterns
                    content = last_assistant_message.get("content", "")
                    
                    # Look for file paths in the handoff message
                    handoff_paths = re.findall(r'/[\w/\-\.]+\.pdf', content)
                    for path in handoff_paths:
                        if path not in paper_paths:  # Avoid duplicates
                            paper_paths.append(path)
                            
                    # Look for paper titles in the handoff message
                    for pattern in title_patterns:
                        found_titles = re.findall(pattern, content)
                        for title in found_titles:
                            if title not in paper_titles:  # Avoid duplicates
                                paper_titles.append(title)
                
                # If we found paper information, enhance the last user message
                if paper_titles or paper_paths:
                    enhanced_message = "Please add the following papers to the vector store immediately. I want them to be accessible for future searches.\n\n"
                    
                    # Add paper titles and paths when we have both and they match up
                    if len(paper_titles) == len(paper_paths) and len(paper_titles) > 0:
                        for i in range(len(paper_titles)):
                            enhanced_message += f"- '{paper_titles[i]}' at {paper_paths[i]}\n"
                    # Otherwise list what we have, but prioritize paths since they're needed for upload
                    else:
                        # Add paths first since they're most important for actual uploading
                        if paper_paths:
                            enhanced_message += "The file paths are:\n"
                            for path in paper_paths:
                                enhanced_message += f"- {path}\n"
                            
                        if paper_titles and paper_titles != paper_paths:
                            enhanced_message += "\nThe paper titles are:\n"
                            for title in paper_titles:
                                enhanced_message += f"- '{title}'\n"
                    
                    # Add explicit instructions about what to do
                    enhanced_message += "\nUse the upload_file_to_vector_store() function for each file above. Be sure to use the active vector store ID. After uploading, confirm which papers were added successfully."
                    
                    logger.info(f"Enhanced message: {enhanced_message}")
                    
                    # Add system instruction to the messages list to help guide the agent
                    system_instruction = {
                        "role": "system", 
                        "content": "The user has just asked to add papers to the vector store. Make sure to call upload_file_to_vector_store() for each file and confirm the upload. Do not respond with a generic greeting."
                    }
                    input_data.messages.append(system_instruction)
                    
                    # Replace the last user message with our enhanced version
                    for i, msg in enumerate(input_data.messages):
                        if msg == last_user_message:
                            input_data.messages[i]["content"] = enhanced_message
                            break
    
    return input_data

def create_triage_agent(
    use_voice_optimization: bool = False
) -> Agent:
    """
    Create a triage agent that routes requests to specialized agents.
    
    Args:
        use_voice_optimization: Whether to use voice-optimized instructions for sub-agents
        
    Returns:
        Agent: The initialized triage agent
    """
    # Create specialized agents
    vector_store_agent = create_vector_store_agent(use_voice_optimization)
    arxiv_research_agent = create_arxiv_research_agent(use_voice_optimization)
    
    # Define a custom handoff from arxiv_research_agent to vector_store_agent that preserves context
    # This ensures that when the arxiv agent hands off to vector store agent, full context is preserved
    # and simple affirmative responses are properly expanded
    arxiv_to_vector_handoff = handoff(
        agent=vector_store_agent,
        # Use our custom filter to enhance the handoff with paper information
        input_filter=arxiv_to_vector_handoff_filter
    )
    
    # Define a callback function that gets executed during handoff
    def on_vector_store_handoff(context):
        """Callback that gets executed when handoff to vector store happens"""
        logger.info(f"Vector store handoff triggered")
        # This is where we could add special handling for the vector store handoff
        # We can access the message history from context if needed
    
    # Add this specialized handoff to the arxiv agent
    arxiv_research_agent_with_handoffs = Agent(
        name=arxiv_research_agent.name,
        instructions=arxiv_research_agent.instructions,
        tools=arxiv_research_agent.tools,
        handoffs=[
            # Add structured data handoff with callback
            handoff(
                agent=vector_store_agent,
                input_filter=arxiv_to_vector_handoff_filter,
                on_handoff=on_vector_store_handoff
            )
        ]
    )
    
    # Create the agent with the instructions from prompts.py
    agent = Agent(
        name="Assistant",
        instructions=prompt_with_handoff_instructions(BASE_INSTRUCTIONS),
        handoffs=[vector_store_agent, arxiv_research_agent_with_handoffs],
    )
    
    return agent