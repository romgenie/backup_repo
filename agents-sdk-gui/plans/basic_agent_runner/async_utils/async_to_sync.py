import asyncio
import functools
from typing import Callable, Any, TypeVar
import threading

T = TypeVar('T', bound=Callable[..., Any])

# Store event loops by thread to prevent closing loops that might be reused
_thread_local = threading.local()

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
        # Get current thread ID
        thread_id = threading.get_ident()
        
        # Check if we already have a loop for this thread
        if not hasattr(_thread_local, 'event_loops'):
            _thread_local.event_loops = {}
            
        if thread_id not in _thread_local.event_loops or _thread_local.event_loops[thread_id].is_closed():
            # Create a new event loop if we don't have one or it's closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _thread_local.event_loops[thread_id] = loop
        else:
            # Use the existing loop
            loop = _thread_local.event_loops[thread_id]
            asyncio.set_event_loop(loop)
            
        try:
            # Run the async function in the event loop
            return loop.run_until_complete(async_func(*args, **kwargs))
        except Exception as e:
            print(f"Error in async execution: {str(e)}")
            raise
        
        # Don't close the loop - it might be used again
        # The loop will be cleaned up when the thread exits
    
    return wrapper