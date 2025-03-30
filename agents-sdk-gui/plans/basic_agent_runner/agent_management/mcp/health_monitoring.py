"""
Health monitoring functions for MCP servers.
"""

import time
import traceback
from typing import Dict, Any, List, Optional

from .server_lifecycle import create_server_from_config
from .tool_management import get_tool_details

def update_health_status(manager, server_id: str, status_update: Dict[str, Any]) -> None:
    """
    Update the health status for a server
    
    Args:
        manager: The McpManager instance
        server_id: The ID of the server
        status_update: Dictionary with status updates
    """
    # Initialize if not exists
    if server_id not in manager.server_health:
        manager.server_health[server_id] = {}
    
    # Update with new values
    manager.server_health[server_id].update(status_update)

def get_health_status(manager, server_id: str) -> Dict[str, Any]:
    """
    Get the health status for a server
    
    Args:
        manager: The McpManager instance
        server_id: The ID of the server
        
    Returns:
        Health status dictionary or empty dict if not found
    """
    return manager.server_health.get(server_id, {})

async def test_server_async(manager, server_id: str) -> Dict[str, Any]:
    """
    Test connection to an MCP server and list available tools (async version)
    
    Args:
        manager: The McpManager instance
        server_id: The ID of the server to test
        
    Returns:
        Dictionary with test results
    """
    # Create the server if needed - Get the raw server without wrapper
    if server_id not in manager.active_servers:
        # Get the configuration
        if server_id not in manager.server_configs:
            return {"success": False, "error": "Server configuration not found"}
            
        config = manager.server_configs[server_id]
        
        # Create the server using our refactored helper method
        try:
            server = await create_server_from_config(config, server_id)
            if not server:
                return {"success": False, "error": "Failed to create server instance"}
        except Exception as e:
            error_trace = traceback.format_exc()
            return {"success": False, "error": str(e), "traceback": error_trace}
    else:
        server = manager.active_servers[server_id]
    
    try:
        # Connect to the server first
        if hasattr(server, 'connect'):
            await server.connect()
            
        # List the tools to test connection
        tools = await server.list_tools()
        
        # Process tools information
        tool_details = get_tool_details(tools)
        tool_names = [t.name for t in tools]
        
        # Store tool details in cache
        manager.server_tools_cache[server_id] = tool_details
        
        # Update server health status
        update_health_status(manager, server_id, {
            "last_check": time.time(),
            "status": "healthy",
            "tool_count": len(tool_names)
        })
        
        # Close the temporary server if it's not in active_servers
        if server_id not in manager.active_servers and hasattr(server, 'close'):
            try:
                await server.close()
            except Exception as e:
                print(f"Warning: Error closing server: {str(e)}")
                # Continue anyway
        
        return {
            "success": True,
            "tools": tool_names,
            "count": len(tool_names),
            "details": tool_details
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        
        # Update server health status
        update_health_status(manager, server_id, {
            "last_check": time.time(),
            "status": "error",
            "error": str(e)
        })
        
        # Close the temporary server if it's not in active_servers
        if server_id not in manager.active_servers and hasattr(server, 'close'):
            try:
                await server.close()
            except Exception as close_err:
                print(f"Warning: Error closing server: {str(close_err)}")
                # Continue anyway
                
        return {
            "success": False,
            "error": str(e),
            "traceback": error_trace
        }