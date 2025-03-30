import asyncio
from typing import Callable, Awaitable, Any, TypeVar, cast

T = TypeVar('T')

def create_new_event_loop() -> asyncio.AbstractEventLoop:
    """
    Creates and sets a new event loop.
    
    Returns:
        The new event loop
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop

def run_in_loop(coro: Awaitable[T]) -> T:
    """
    Runs a coroutine in the current event loop or creates a new one if none exists.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
        
    Example:
        async def fetch_data():
            # async code here
            return data
            
        result = run_in_loop(fetch_data())
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If the loop is already running (like in Jupyter), create a future
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return cast(T, future.result())
        else:
            # Use the existing loop
            return cast(T, loop.run_until_complete(coro))
    except RuntimeError:
        # If there's no event loop, create a new one
        return cast(T, asyncio.run(coro))