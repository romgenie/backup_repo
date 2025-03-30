"""
Security functions for MCP parameter validation and logging.
"""

import os
import re
import json
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

def validate_timezone(timezone_str: str) -> Tuple[bool, str]:
    """
    Validates that a timezone parameter is safe and known.
    
    Args:
        timezone_str: The timezone string to validate
        
    Returns:
        Tuple of (is_valid, result_or_error)
    """
    # Return early for empty/non-string values
    if not timezone_str or not isinstance(timezone_str, str):
        return False, "Invalid timezone format"
    
    # Only allow safe characters (alphanumeric, slash, underscore, dash, space)
    if not re.match(r'^[A-Za-z0-9_/\- ]+$', timezone_str):
        # Log suspicious input for security review
        log_security_event("SUSPICIOUS_TIMEZONE_CHARS", timezone_str, "WARN")
        return False, "Invalid timezone format"
    
    # Remove any suspicious patterns that might be used for command injection
    suspect_patterns = [";", "&&", "||", "`", "$", "(", ")", 
                       "|", ">", "<", "{", "}", "[", "]", "\\"]
    for pattern in suspect_patterns:
        if pattern in timezone_str:
            log_security_event("COMMAND_INJECTION_ATTEMPT", timezone_str, "HIGH")
            return False, "Invalid timezone format"
    
    # Validate against known timezone database
    try:
        import pytz
        valid_timezones = pytz.all_timezones
        
        # Direct match
        if timezone_str in valid_timezones:
            return True, timezone_str
        
        # Case-insensitive match
        for tz in valid_timezones:
            if tz.lower() == timezone_str.lower():
                return True, tz
        
        # Try finding close match
        close_matches = [tz for tz in valid_timezones 
                         if timezone_str.replace(" ", "_").lower() in tz.lower()]
        if close_matches:
            return True, close_matches[0]
            
        # Try common city names mapping
        city_to_timezone = {
            "new york": "America/New_York",
            "london": "Europe/London",
            "paris": "Europe/Paris",
            "tokyo": "Asia/Tokyo",
            "sydney": "Australia/Sydney",
            "los angeles": "America/Los_Angeles",
            "chicago": "America/Chicago"
        }
        
        for city, tz in city_to_timezone.items():
            if city.lower() in timezone_str.lower():
                return True, tz
                
        # If we get here, the timezone is not recognized
        log_security_event("UNKNOWN_TIMEZONE", timezone_str, "INFO")
        return False, "UTC"  # Default to UTC for unrecognized timezones
        
    except ImportError:
        # If pytz is not available, use a safe default
        return False, "UTC"

def validate_time_string(time_str: str) -> str:
    """
    Validates and sanitizes a time string parameter.
    
    Args:
        time_str: The time string to validate
        
    Returns:
        Sanitized time string or current time if invalid
    """
    if not time_str or not isinstance(time_str, str):
        return datetime.datetime.now().isoformat()
        
    # Remove potentially dangerous characters
    safe_time = re.sub(r'[^A-Za-z0-9_/\-:. ]', '', time_str)
    
    if safe_time != time_str:
        log_security_event("SUSPICIOUS_TIME_STRING", time_str, "WARN")
        
    # Try to parse the time
    try:
        # First attempt: Try as ISO format
        datetime.datetime.fromisoformat(safe_time.replace('Z', '+00:00'))
        return safe_time
    except ValueError:
        try:
            # Second attempt: Try as natural language with dateutil
            from dateutil import parser
            parsed_time = parser.parse(safe_time)
            return parsed_time.isoformat()
        except:
            # If all parsing fails, return current time
            log_security_event("INVALID_TIME_FORMAT", time_str, "INFO")
            return datetime.datetime.now().isoformat()

def sanitize_mcp_parameters(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Framework for sanitizing all MCP tool parameters.
    
    Args:
        tool_name: Name of the MCP tool being called
        arguments: Original arguments dictionary
        
    Returns:
        Sanitized parameters dictionary
    """
    sanitized = {}
    
    # Ensure we have a dict to work with
    if not arguments or not isinstance(arguments, dict):
        return {}
    
    # Copy non-sensitive parameters
    for key, value in arguments.items():
        if key not in ["timezone", "from_timezone", "to_timezone", "time"]:
            sanitized[key] = value
    
    # Tool-specific parameter handling
    if tool_name == "get_current_time":
        # Timezone validation
        if "timezone" in arguments:
            is_valid, result = validate_timezone(arguments["timezone"])
            sanitized["timezone"] = result
    
    elif tool_name == "convert_time":
        # Handle all parameters for this tool
        for param_name in ["from_timezone", "to_timezone"]:
            if param_name in arguments:
                is_valid, result = validate_timezone(arguments[param_name])
                sanitized[param_name] = result
        
        # Time string validation
        if "time" in arguments:
            sanitized["time"] = validate_time_string(arguments["time"])
    
    # Log any differences for security monitoring
    if sanitized != arguments:
        log_security_event("PARAMETERS_SANITIZED", 
                          {"tool": tool_name, 
                           "original": arguments, 
                           "sanitized": sanitized}, 
                          "INFO")
    
    return sanitized

def log_security_event(event_type: str, data: Any, severity: str = "INFO") -> None:
    """
    Log security events for analysis.
    
    Args:
        event_type: Type of security event
        data: Data related to the event
        severity: Severity level (INFO, WARN, HIGH)
    """
    # Define log levels
    levels = {
        "INFO": "INFO",
        "WARN": "WARNING",
        "HIGH": "CRITICAL" 
    }
    
    level = levels.get(severity, "INFO")
    
    # Format the event data
    event = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": event_type,
        "severity": severity,
        "data": data if isinstance(data, (str, int, float, bool, type(None))) else str(data)
    }
    
    # Print to console
    print(f"SECURITY {level}: {event_type} - {json.dumps(event)}")
    
    # In production, also log to file and/or security monitoring system
    try:
        log_dir = "security_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, "security_events.log")
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"Error logging security event: {str(e)}")