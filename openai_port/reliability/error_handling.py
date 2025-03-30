"""
Error handling for reliability processing.

This module provides error handling capabilities for reliability processing,
including standardized error types, retry logic, and recovery strategies.
"""

import asyncio
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union


class ReliabilityError(Exception):
    """Base class for reliability-related errors."""
    pass


class ValidationError(ReliabilityError):
    """Error raised when validation fails."""
    
    def __init__(self, message: str, validation_type: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize a ValidationError.
        
        Args:
            message: Error message
            validation_type: Type of validation that failed
            details: Additional error details
        """
        super().__init__(message)
        self.validation_type = validation_type
        self.details = details or {}


class ProcessingError(ReliabilityError):
    """Error raised when processing fails."""
    
    def __init__(self, message: str, processing_step: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize a ProcessingError.
        
        Args:
            message: Error message
            processing_step: Processing step that failed
            details: Additional error details
        """
        super().__init__(message)
        self.processing_step = processing_step
        self.details = details or {}


class TransformationError(ReliabilityError):
    """Error raised when transformation fails."""
    
    def __init__(self, message: str, transformation_type: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize a TransformationError.
        
        Args:
            message: Error message
            transformation_type: Type of transformation that failed
            details: Additional error details
        """
        super().__init__(message)
        self.transformation_type = transformation_type
        self.details = details or {}


class RetryConfiguration:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        retry_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        """
        Initialize a RetryConfiguration.
        
        Args:
            max_retries: Maximum number of retries
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Exponential backoff factor
            retry_exceptions: List of exception types to retry on
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.retry_exceptions = retry_exceptions or [
            ValidationError,
            ProcessingError,
            TransformationError,
        ]


async def retry_async(
    func: Callable,
    args: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[RetryConfiguration] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Any:
    """
    Retry an asynchronous function with exponential backoff.
    
    Args:
        func: Function to retry
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        config: Retry configuration
        on_retry: Callback to call on retry
        
    Returns:
        Result of the function
        
    Raises:
        Exception: If all retries fail
    """
    args = args or []
    kwargs = kwargs or {}
    config = config or RetryConfiguration()
    
    retries = 0
    last_exception = None
    
    while retries <= config.max_retries:
        try:
            if retries > 0:
                # Calculate retry delay with exponential backoff
                delay = config.retry_delay * (config.backoff_factor ** (retries - 1))
                
                # Call on_retry callback if provided
                if on_retry:
                    on_retry(last_exception, retries)
                
                # Wait before retrying
                await asyncio.sleep(delay)
            
            # Call the function
            return await func(*args, **kwargs)
        
        except tuple(config.retry_exceptions) as e:
            # Store exception for potential re-raise
            last_exception = e
            retries += 1
        
        except Exception as e:
            # Immediately re-raise exceptions not in retry_exceptions
            raise
    
    # If we've exhausted all retries, re-raise the last exception
    if last_exception:
        raise last_exception


def retry_sync(
    func: Callable,
    args: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[RetryConfiguration] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Any:
    """
    Retry a synchronous function with exponential backoff.
    
    Args:
        func: Function to retry
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        config: Retry configuration
        on_retry: Callback to call on retry
        
    Returns:
        Result of the function
        
    Raises:
        Exception: If all retries fail
    """
    args = args or []
    kwargs = kwargs or {}
    config = config or RetryConfiguration()
    
    retries = 0
    last_exception = None
    
    while retries <= config.max_retries:
        try:
            if retries > 0:
                # Calculate retry delay with exponential backoff
                delay = config.retry_delay * (config.backoff_factor ** (retries - 1))
                
                # Call on_retry callback if provided
                if on_retry:
                    on_retry(last_exception, retries)
                
                # Wait before retrying
                time.sleep(delay)
            
            # Call the function
            return func(*args, **kwargs)
        
        except tuple(config.retry_exceptions) as e:
            # Store exception for potential re-raise
            last_exception = e
            retries += 1
        
        except Exception as e:
            # Immediately re-raise exceptions not in retry_exceptions
            raise
    
    # If we've exhausted all retries, re-raise the last exception
    if last_exception:
        raise last_exception


class ErrorLogger:
    """Logger for reliability-related errors."""
    
    def __init__(self):
        """Initialize the ErrorLogger."""
        self.errors = []
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an error.
        
        Args:
            error: The error to log
            context: Additional context information
        """
        error_info = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
        }
        
        self.errors.append(error_info)
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """
        Get all logged errors.
        
        Returns:
            List of error information dictionaries
        """
        return self.errors
    
    def get_recent_errors(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent errors.
        
        Args:
            count: Maximum number of errors to return
            
        Returns:
            List of recent error information dictionaries
        """
        return self.errors[-count:]
    
    def clear_errors(self):
        """Clear all logged errors."""
        self.errors = []


# Create a global error logger instance
error_logger = ErrorLogger()


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Log an error to the global error logger.
    
    Args:
        error: The error to log
        context: Additional context information
    """
    error_logger.log_error(error, context)


def get_errors() -> List[Dict[str, Any]]:
    """
    Get all logged errors from the global error logger.
    
    Returns:
        List of error information dictionaries
    """
    return error_logger.get_errors()


def get_recent_errors(count: int = 5) -> List[Dict[str, Any]]:
    """
    Get the most recent errors from the global error logger.
    
    Args:
        count: Maximum number of errors to return
        
    Returns:
        List of recent error information dictionaries
    """
    return error_logger.get_recent_errors(count)


def clear_errors():
    """Clear all logged errors from the global error logger."""
    error_logger.clear_errors()
