"""
Utility functions for validation in reliability processing.

This module provides helper functions for validating different types of content
in agent responses, with enhanced content analysis capabilities.
"""

import re
import string
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import unicodedata

# ---------------------------------------------------------------------------
# URL validation functions
# ---------------------------------------------------------------------------

def find_urls_in_text(text: str) -> List[str]:
    """
    Find all URLs in the given text using regex pattern matching.
    
    Args:
        text: The text to search for URLs
        
    Returns:
        A list of URLs found in the text
    """
    # This pattern matches URLs starting with http://, https://, ftp://, or www.
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Alternative pattern that also matches www. URLs
    www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Find URLs
    http_urls = re.findall(url_pattern, text)
    www_urls = re.findall(www_pattern, text)
    
    # Filter www URLs that are already part of http URLs
    filtered_www_urls = []
    for www_url in www_urls:
        if not any(www_url in http_url for http_url in http_urls):
            filtered_www_urls.append(www_url)
    
    # Combine HTTP URLs and filtered WWW URLs
    return http_urls + filtered_www_urls


def contains_urls(texts: List[Union[str, Any]]) -> bool:
    """
    Check if any of the provided texts contain URLs.
    
    Args:
        texts: A list of text strings or objects to check
        
    Returns:
        True if any of the texts contain URLs, False otherwise
    """
    for text in texts:
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                continue
        
        if find_urls_in_text(text):
            return True
    return False


def validate_url(url: str) -> Dict[str, Any]:
    """
    Validate a URL for basic formatting and structure.
    
    Args:
        url: The URL to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        "is_valid": False,
        "issues": [],
        "confidence": 0.0,
    }
    
    # Check if URL starts with http:// or https://
    if not (url.startswith("http://") or url.startswith("https://")):
        validation_result["issues"].append("URL does not start with http:// or https://")
    
    # Check if URL contains a domain
    domain_pattern = r'https?://([a-zA-Z0-9][-a-zA-Z0-9]*(\.[a-zA-Z0-9][-a-zA-Z0-9]*)+)'
    domain_match = re.search(domain_pattern, url)
    if not domain_match:
        validation_result["issues"].append("URL does not contain a valid domain")
    
    # Calculate confidence based on issues
    if not validation_result["issues"]:
        validation_result["is_valid"] = True
        validation_result["confidence"] = 1.0
    else:
        validation_result["confidence"] = max(0.0, 1.0 - 0.2 * len(validation_result["issues"]))
    
    return validation_result


def analyze_url_content(text: str) -> Dict[str, Any]:
    """
    Analyze text for URL content and provide metrics.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with analysis results
    """
    urls = find_urls_in_text(text)
    
    analysis = {
        "url_count": len(urls),
        "urls": urls,
        "has_urls": len(urls) > 0,
        "url_density": 0.0,
        "validated_urls": [],
        "invalid_urls": [],
    }
    
    # Calculate URL density (URLs per 100 words)
    words = re.findall(r'\b\w+\b', text)
    if words:
        analysis["url_density"] = (len(urls) / len(words)) * 100
    
    # Validate each URL
    for url in urls:
        validation = validate_url(url)
        if validation["is_valid"]:
            analysis["validated_urls"].append(url)
        else:
            analysis["invalid_urls"].append({
                "url": url,
                "issues": validation["issues"],
            })
    
    return analysis


# ---------------------------------------------------------------------------
# Number validation functions
# ---------------------------------------------------------------------------

def extract_numbers_from_text(text: str) -> List[str]:
    """
    Extract numbers from the given text.
    
    Args:
        text: The text to search for numbers
        
    Returns:
        A list of number strings found in the text
    """
    # This pattern matches integers and floating point numbers
    number_pattern = r'\b\d+(?:\.\d+)?\b'
    
    # Extract numbers
    numbers = re.findall(number_pattern, text)
    
    # Also extract numbers with commas or currency symbols
    currency_number_pattern = r'\b(?:\$|€|£|¥)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\b'
    currency_numbers = re.findall(currency_number_pattern, text)
    
    # Combine and remove duplicates
    all_numbers = list(set(numbers + currency_numbers))
    
    return all_numbers


def contains_numbers(texts: List[Union[str, Any]]) -> bool:
    """
    Check if any of the provided texts contain numbers.
    
    Args:
        texts: A list of text strings or objects to check
        
    Returns:
        True if any of the texts contain numbers, False otherwise
    """
    for text in texts:
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                continue
        
        if extract_numbers_from_text(text):
            return True
    return False


def analyze_number_content(text: str) -> Dict[str, Any]:
    """
    Analyze text for numerical content and provide metrics.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with analysis results
    """
    numbers = extract_numbers_from_text(text)
    
    analysis = {
        "number_count": len(numbers),
        "numbers": numbers,
        "has_numbers": len(numbers) > 0,
        "number_density": 0.0,
        "percentage_numbers": [],
        "currency_numbers": [],
        "large_numbers": [],
    }
    
    # Calculate number density (numbers per 100 words)
    words = re.findall(r'\b\w+\b', text)
    if words:
        analysis["number_density"] = (len(numbers) / len(words)) * 100
    
    # Categorize numbers
    for number_str in numbers:
        # Clean the number string
        clean_number = number_str.replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
        
        try:
            value = float(clean_number)
            
            # Check if it's a percentage
            if "%" in number_str or (0 <= value <= 100 and "percent" in text.lower()):
                analysis["percentage_numbers"].append(number_str)
            
            # Check if it's a currency
            if any(symbol in number_str for symbol in ["$", "€", "£", "¥"]):
                analysis["currency_numbers"].append(number_str)
            
            # Check if it's a large number
            if value >= 1000:
                analysis["large_numbers"].append(number_str)
        except ValueError:
            # Skip non-number values
            pass
    
    return analysis


# ---------------------------------------------------------------------------
# Code validation functions
# ---------------------------------------------------------------------------

def extract_code_blocks(text: str) -> List[str]:
    """
    Extract code blocks from the given text.
    
    Args:
        text: The text to search for code blocks
        
    Returns:
        A list of code blocks found in the text
    """
    # This pattern matches code blocks in markdown format (```code```)
    code_block_pattern = r'```(?:[a-zA-Z]+)?\n([\s\S]*?)```'
    
    # Extract code blocks
    code_blocks = re.findall(code_block_pattern, text)
    
    return code_blocks


def contains_code(texts: List[Union[str, Any]]) -> bool:
    """
    Check if any of the provided texts contain code blocks.
    
    Args:
        texts: A list of text strings or objects to check
        
    Returns:
        True if any of the texts contain code blocks, False otherwise
    """
    for text in texts:
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                continue
        
        if extract_code_blocks(text):
            return True
        
        # Also check for inline code (typically in `backticks`)
        inline_code_pattern = r'`[^`]+`'
        if re.search(inline_code_pattern, text):
            return True
    
    return False


def analyze_code_content(text: str) -> Dict[str, Any]:
    """
    Analyze text for code content and provide metrics.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with analysis results
    """
    code_blocks = extract_code_blocks(text)
    
    # Check for inline code
    inline_code_pattern = r'`([^`]+)`'
    inline_code = re.findall(inline_code_pattern, text)
    
    analysis = {
        "code_block_count": len(code_blocks),
        "inline_code_count": len(inline_code),
        "has_code": len(code_blocks) > 0 or len(inline_code) > 0,
        "code_languages": [],
        "code_length": 0,
        "code_density": 0.0,
    }
    
    # Extract code languages
    language_pattern = r'```([a-zA-Z]+)'
    languages = re.findall(language_pattern, text)
    analysis["code_languages"] = languages
    
    # Calculate total code length
    for block in code_blocks:
        analysis["code_length"] += len(block)
    
    for code in inline_code:
        analysis["code_length"] += len(code)
    
    # Calculate code density (code characters per total characters)
    if len(text) > 0:
        analysis["code_density"] = (analysis["code_length"] / len(text)) * 100
    
    return analysis


# ---------------------------------------------------------------------------
# Information validation functions
# ---------------------------------------------------------------------------

def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from the given text.
    
    Args:
        text: The text to extract phrases from
        max_phrases: Maximum number of phrases to extract
        
    Returns:
        A list of key phrases
    """
    # This is a simple implementation that extracts noun phrases
    # For a real implementation, consider using a library like NLTK or spaCy
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    
    phrases = []
    for sentence in sentences:
        # Look for noun phrases (simplistic approach)
        noun_phrase_pattern = r'\b(?:the|a|an)\s+(?:\w+\s+){0,3}(?:thing|person|place|idea|concept|theory|approach|method|technique|system|process|product|service)\b'
        noun_phrases = re.findall(noun_phrase_pattern, sentence, re.IGNORECASE)
        
        # Add to phrases
        phrases.extend(noun_phrases)
        
        # Stop if we have enough phrases
        if len(phrases) >= max_phrases:
            break
    
    # Deduplicate phrases
    phrases = list(set(phrases))
    
    # Limit to max_phrases
    return phrases[:max_phrases]


def contains_information(texts: List[Union[str, Any]]) -> bool:
    """
    Check if any of the provided texts contain informational content.
    
    Args:
        texts: A list of text strings or objects to check
        
    Returns:
        True if any of the texts contain informational content, False otherwise
    """
    for text in texts:
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                continue
        
        # Check if text has substantial content
        if len(text.strip()) > 50:
            return True
    
    return False


def analyze_information_content(text: str) -> Dict[str, Any]:
    """
    Analyze text for informational content and provide metrics.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Count words
    words = re.findall(r'\b\w+\b', text)
    
    # Extract key phrases
    key_phrases = extract_key_phrases(text)
    
    analysis = {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "has_substantial_content": len(words) > 50,
        "average_sentence_length": 0,
        "key_phrases": key_phrases,
        "factual_statements": [],
    }
    
    # Calculate average sentence length
    if sentences:
        words_per_sentence = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        analysis["average_sentence_length"] = sum(words_per_sentence) / len(sentences)
    
    # Identify potential factual statements
    factual_pattern = r'\b(?:is|are|was|were|has|have|had)\b'
    for sentence in sentences:
        if re.search(factual_pattern, sentence):
            analysis["factual_statements"].append(sentence)
    
    return analysis


# ---------------------------------------------------------------------------
# Combined content analysis
# ---------------------------------------------------------------------------

def analyze_content(text: str) -> Dict[str, Any]:
    """
    Analyze text for various content types and provide comprehensive metrics.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "url_analysis": analyze_url_content(text),
        "number_analysis": analyze_number_content(text),
        "code_analysis": analyze_code_content(text),
        "information_analysis": analyze_information_content(text),
        "validation_priorities": {},
    }
    
    # Calculate validation priorities based on content analysis
    priorities = {}
    
    # URL validation priority
    if analysis["url_analysis"]["has_urls"]:
        url_priority = min(1.0, analysis["url_analysis"]["url_density"] / 10)
        priorities["url_validation"] = max(0.5, url_priority)
    else:
        priorities["url_validation"] = 0.0
    
    # Number validation priority
    if analysis["number_analysis"]["has_numbers"]:
        number_priority = min(1.0, analysis["number_analysis"]["number_density"] / 20)
        priorities["number_validation"] = max(0.5, number_priority)
    else:
        priorities["number_validation"] = 0.0
    
    # Code validation priority
    if analysis["code_analysis"]["has_code"]:
        code_priority = min(1.0, analysis["code_analysis"]["code_density"] / 30)
        priorities["code_validation"] = max(0.5, code_priority)
    else:
        priorities["code_validation"] = 0.0
    
    # Information validation priority (always at least 0.5)
    information_priority = min(1.0, analysis["information_analysis"]["word_count"] / 300)
    priorities["information_validation"] = max(0.5, information_priority)
    
    analysis["validation_priorities"] = priorities
    
    return analysis


def should_validate(content_type: str, text: str, threshold: float = 0.3) -> bool:
    """
    Determine if content should be validated based on content analysis.
    
    Args:
        content_type: Type of content to validate (url, number, code, information)
        text: Text to analyze
        threshold: Validation priority threshold
        
    Returns:
        True if content should be validated, False otherwise
    """
    analysis = analyze_content(text)
    priority_key = f"{content_type}_validation"
    
    if priority_key in analysis["validation_priorities"]:
        return analysis["validation_priorities"][priority_key] >= threshold
    
    # Default to True for unrecognized content types
    return True
