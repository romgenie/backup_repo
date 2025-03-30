import asyncio
import functools
from typing import Callable, Any, TypeVar

T = TypeVar('T', bound=Callable[..., Any])

def async_to_sync(async_func: Callable) -> Callable:
    """
    Decorator to run an async function in a synchronous context.
    Primarily useful for Streamlit which doesn't support async functions directly.
    
    Args:
        async_func: The asynchronous function to wrap
        
    Returns:
        A synchronous function that runs the async function in a new event loop
        
    Example:
        @async_to_sync
        async def fetch_data():
            # async code here
            return data
            
        # Now fetch_data can be called synchronously
        result = fetch_data()
    """
    @functools.wraps(async_func)
    def wrapper(*args, **kwargs):
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the async function in the new event loop
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            # Close the event loop
            loop.close()
    
    return wrapper