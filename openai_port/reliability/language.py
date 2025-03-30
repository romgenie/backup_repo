"""
Language transformation for reliability processing.

This module provides the LanguageTransformer class, which transforms
responses to different languages while preserving reliability.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Set, Union

from agents import Agent, Runner, ItemHelpers
from .context_manager import ContextManager


class LanguageTransformer:
    """
    Transformer for translating responses to different languages.
    
    The LanguageTransformer handles language detection and translation
    while preserving reliability metrics.
    """
    
    def __init__(self, target_language: str):
        """
        Initialize the LanguageTransformer.
        
        Args:
            target_language: Target language for transformation
        """
        self.target_language = target_language
        self.context_manager = ContextManager()
    
    async def transform(self, content: Any, model: Optional[str] = None) -> Any:
        """
        Transform content to the target language.
        
        Args:
            content: Content to transform
            model: Model to use for transformation
            
        Returns:
            Transformed content
        """
        # Check if content is a string
        if isinstance(content, str):
            return await self._transform_text(content, model)
        
        # Check if content is a dictionary
        if isinstance(content, dict):
            return await self._transform_dict(content, model)
        
        # Check if content is a list
        if isinstance(content, list):
            return await self._transform_list(content, model)
        
        # If content is not a supported type, return as is
        return content
    
    async def _transform_text(self, text: str, model: Optional[str] = None) -> str:
        """
        Transform text content to the target language.
        
        Args:
            text: Text content to transform
            model: Model to use for transformation
            
        Returns:
            Transformed text
        """
        # Check if already in target language
        if await self._is_in_target_language(text):
            return text
        
        # Create a translation agent
        translation_agent = Agent(
            name="Language Transformer",
            instructions=f"Translate the following text to {self.target_language}. Preserve all formatting, code blocks, URLs, and other special content. Do not add explanations or notes.",
        )
        
        # Run the translation
        result = await Runner.run(
            translation_agent,
            text,
            model=model,
        )
        
        # Store the translation in context
        translation_key = f"translation_{hash(text)}"
        self.context_manager.add_agent_context(translation_key, result.final_output)
        
        return result.final_output
    
    async def _transform_dict(self, content: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
        """
        Transform dictionary content to the target language.
        
        Args:
            content: Dictionary content to transform
            model: Model to use for transformation
            
        Returns:
            Transformed dictionary
        """
        result = {}
        
        for key, value in content.items():
            # Transform value based on type
            if isinstance(value, str):
                result[key] = await self._transform_text(value, model)
            elif isinstance(value, dict):
                result[key] = await self._transform_dict(value, model)
            elif isinstance(value, list):
                result[key] = await self._transform_list(value, model)
            else:
                result[key] = value
        
        return result
    
    async def _transform_list(self, content: List[Any], model: Optional[str] = None) -> List[Any]:
        """
        Transform list content to the target language.
        
        Args:
            content: List content to transform
            model: Model to use for transformation
            
        Returns:
            Transformed list
        """
        result = []
        
        for item in content:
            # Transform item based on type
            if isinstance(item, str):
                result.append(await self._transform_text(item, model))
            elif isinstance(item, dict):
                result.append(await self._transform_dict(item, model))
            elif isinstance(item, list):
                result.append(await self._transform_list(item, model))
            else:
                result.append(item)
        
        return result
    
    async def _is_in_target_language(self, text: str) -> bool:
        """
        Check if text is already in the target language.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is in target language, False otherwise
        """
        # Create a language detection agent
        detection_agent = Agent(
            name="Language Detector",
            instructions=f"Determine if the following text is in {self.target_language}. Respond with 'yes' or 'no'.",
        )
        
        # Run the detection
        result = await Runner.run(
            detection_agent,
            text,
        )
        
        # Check the result
        response = result.final_output.lower().strip()
        return response == "yes" or response == "true"


async def transform_to_language(content: Any, target_language: str, model: Optional[str] = None) -> Any:
    """
    Transform content to the target language.
    
    Args:
        content: Content to transform
        target_language: Target language for transformation
        model: Model to use for transformation
        
    Returns:
        Transformed content
    """
    transformer = LanguageTransformer(target_language)
    return await transformer.transform(content, model)


class SupportedLanguages:
    """Constants for commonly supported languages."""
    ENGLISH = "English"
    SPANISH = "Spanish"
    FRENCH = "French"
    GERMAN = "German"
    ITALIAN = "Italian"
    PORTUGUESE = "Portuguese"
    CHINESE = "Chinese"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    RUSSIAN = "Russian"
    ARABIC = "Arabic"
    HINDI = "Hindi"
