"""
Account information tools for the voice assistant.
"""
from typing import Dict
from agents import function_tool

@function_tool
def get_account_info(user_id: str) -> Dict[str, str]:
    """
    Return account information for a given user.
    
    Args:
        user_id: The ID of the user to get account information for
        
    Returns:
        A dictionary containing account information
    """
    # This is a dummy implementation - in a real application, 
    # this would query a database or API
    return {
        "user_id": user_id,
        "name": "Bugs Bunny",
        "account_balance": "£72.50",
        "membership_status": "Gold Executive"
    }