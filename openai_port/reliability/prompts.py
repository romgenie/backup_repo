"""
Validation prompts for reliability processing.

This module defines the prompts used by validator agents to check different
aspects of agent responses for hallucinations.
"""

# URL validation prompt for checking if URLs in the response are valid
URL_VALIDATION_PROMPT = """
Focus on basic URL source validation:

Source Verification:
- Check if the source is come from the content. But dont make assumption just check the context and try to find exact things. If not flag it.
- If you can see the things in the context everything okay (Trusted Source).

IMPORTANT: If the URL source cannot be verified, flag it as suspicious.
"""

# Number validation prompt for checking if numbers in the response are accurate
NUMBER_VALIDATION_PROMPT = """
Focus on basic numerical validation:

Number Verification:
- Check if the source is come from the content. But dont make assumption just check the context and try to find exact things. If not flag it.
- If you can see the things in the context everything okay (Trusted Source).

IMPORTANT: If the numbers cannot be verified, flag them as suspicious.
"""

# Code validation prompt for checking if code in the response is valid
CODE_VALIDATION_PROMPT = """
Focus on basic code validation:

Code Verification:
- Check if the source is come from the content. But dont make assumption just check the context and try to find exact things. If not flag it.
- If you can see the things in the context everything okay (Trusted Source).

IMPORTANT: If the code cannot be verified or appears suspicious, flag it as suspicious.
"""

# Information validation prompt for checking if general information in the response is valid
INFORMATION_VALIDATION_PROMPT = """
Focus on basic information validation:

Information Verification:
- Check if the source is come from the content. But dont make assumption just check the context and try to find exact things. If not flag it.
- If you can see the things in the context everything okay (Trusted Source).

IMPORTANT: If the information cannot be verified, flag it as suspicious.
"""

# Editor task prompt for cleaning up suspicious content
EDITOR_TASK_PROMPT = """
Clean and validate the output by handling suspicious content:

Processing Rules:
1. For ANY suspicious content identified in validation:
- Replace the suspicious value with None
- Do not suggest alternatives
- Do not provide explanations
- Do not modify other parts of the content

2. For non-suspicious content:
- Keep the original value unchanged
- Do not enhance or modify
- Do not add additional information

Processing Steps:
- Set suspicious fields to None
- Keep other fields as is
- Remove any suspicious content entirely
- Maintain original structure

Validation Issues Found:
{validation_feedback}

IMPORTANT:
- Set ALL suspicious values to None
- Keep verified values unchanged
- No explanations or suggestions
- No partial validations
- Maintain response format
"""
