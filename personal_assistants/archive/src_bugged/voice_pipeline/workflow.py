"""
Voice workflow implementation with custom context handling.
"""
from typing import Optional, Dict, Any, List, AsyncIterator
from agents import Agent, Runner
from agents.voice import SingleAgentVoiceWorkflow
import logging

logger = logging.getLogger("conversation_context")

class ContextAwareVoiceWorkflow(SingleAgentVoiceWorkflow):
    """Extended SingleAgentVoiceWorkflow with better context preservation."""
    
    def __init__(self, agent: Agent):
        super().__init__(agent)
        self.conversation_id: Optional[str] = None
        self.conversation_history: List[Dict[str, Any]] = []
        
    async def run(self, transcription: str) -> AsyncIterator[str]:
        """
        Run the workflow with explicit context management.
        
        Args:
            transcription: The transcribed text from audio input
            
        Returns:
            AsyncIterator[str]: An async iterator of response chunks
        """
        # Log that we're running with context
        if self.conversation_id:
            logger.info(f"Running with existing conversation_id: {self.conversation_id}")
        else:
            logger.info("Starting new conversation")
            
        # Set up message history for manual context management
        messages = []
        
        # Add context from our conversation history
        if len(self.conversation_history) >= 2:
            # Get the last few turns (up to 2 turns = 4 messages)
            context_length = min(len(self.conversation_history), 4)
            recent_history = self.conversation_history[-context_length:]
            logger.info(f"Adding {len(recent_history)} messages from history for context")
            
            for msg in recent_history:
                if msg.get("role") in ["user", "assistant"]:  # Skip system messages
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Add the current message
        messages.append({
            "role": "user",
            "content": transcription
        })
        
        # Add to our conversation history after constructing the messages
        self.conversation_history.append({"role": "user", "content": transcription})
        
        # Try to run with conversation_id first
        if self.conversation_id:
            try:
                logger.info(f"Running with conversation_id: {self.conversation_id}")
                result = await Runner.run(
                    self.agent,
                    messages if len(messages) > 1 else transcription,  # Use messages if we have context
                    conversation_id=self.conversation_id
                )
            except Exception as e:
                logger.error(f"Error running with conversation_id: {str(e)}")
                # Fallback to running with just the messages for context
                try:
                    logger.info("Falling back to running with just messages for context")
                    result = await Runner.run(self.agent, messages if len(messages) > 1 else transcription)
                    self.conversation_id = None
                except Exception as e2:
                    logger.error(f"Error running with messages: {str(e2)}")
                    # Ultimate fallback - just the text
                    result = await Runner.run(self.agent, transcription)
                    self.conversation_id = None
        else:
            # First message or no conversation_id
            if len(messages) > 1:
                logger.info(f"Running with {len(messages)} messages for context")
                result = await Runner.run(self.agent, messages)
            else:
                logger.info("Running with single message")
                result = await Runner.run(self.agent, transcription)
            
        # Log all attributes of the result for debugging
        logger.info(f"Result attributes: {dir(result)}")
            
        # Save conversation_id for future runs
        if hasattr(result, 'conversation_id'):
            logger.info(f"Found conversation_id attribute: {result.conversation_id}")
            if result.conversation_id:
                self.conversation_id = result.conversation_id
                logger.info(f"Saved conversation_id: {self.conversation_id}")
            else:
                logger.warning("conversation_id attribute exists but is None")
        else:
            logger.warning("Result does not have conversation_id attribute")
            
            # Try alternative attribute names
            for attr_name in ['thread_id', 'context_id', 'id']:
                if hasattr(result, attr_name):
                    logger.info(f"Found alternative id attribute: {attr_name}={getattr(result, attr_name)}")
                    self.conversation_id = getattr(result, attr_name)
            
        # Track conversation history for debugging
        if hasattr(result, 'messages'):
            if result.messages:
                logger.info(f"Result contains {len(result.messages)} messages in history")
                # Log a sample of the messages
                if len(result.messages) > 0:
                    logger.info(f"Sample message: {result.messages[0]}")
                self.conversation_history.extend(result.messages)
            else:
                logger.warning("messages attribute exists but is empty")
        else:
            logger.warning("Result does not have messages attribute")
        
        # Add the response to our conversation history
        self.conversation_history.append({"role": "assistant", "content": result.final_output})
            
        # Return the result as a single-item async iterator
        async def iterator():
            yield result.final_output
            
        return iterator()

def create_voice_workflow(agent: Agent) -> ContextAwareVoiceWorkflow:
    """
    Create a voice workflow for the specified agent with enhanced context handling.
    
    Args:
        agent: The agent to use in the workflow
        
    Returns:
        ContextAwareVoiceWorkflow: The configured workflow with context preservation
    """
    return ContextAwareVoiceWorkflow(agent)