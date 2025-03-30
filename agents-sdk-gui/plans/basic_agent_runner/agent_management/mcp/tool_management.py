"""
Tool management functions for MCP servers.
"""

from typing import Dict, Any, List, Optional

def get_tool_details(tools: List[Any], server_id: str = None) -> List[Dict[str, Any]]:
    """
    Extract tool details from a list of tools
    
    Args:
        tools: List of tools from an MCP server
        server_id: Optional server ID to include in the details
        
    Returns:
        List of tool details dictionaries
    """
    tool_details = []
    for tool in tools:
        tool_info = {
            "name": tool.name,
            "description": getattr(tool, 'description', ''),
            "parameters": getattr(tool, 'parameters', []),
            "return_type": getattr(tool, 'return_type', None),
        }
        
        # Add server_id if provided
        if server_id:
            tool_info["server_id"] = server_id
            
        tool_details.append(tool_info)
    
    return tool_details

async def invalidate_tool_cache_async(manager, server_id: str) -> bool:
    """
    Invalidate the tools cache for a server (async version)
    
    Args:
        manager: The McpManager instance
        server_id: The ID of the server
        
    Returns:
        Success status
    """
    if server_id in manager.active_servers:
        server = manager.active_servers[server_id]
        
        try:
            server.invalidate_tools_cache()
            # Also remove from our local cache
            if server_id in manager.server_tools_cache:
                del manager.server_tools_cache[server_id]
            return True
        except Exception as e:
            print(f"Error invalidating tools cache: {str(e)}")
            return False
    
    return False

def get_cached_tools(manager, server_id: str) -> List[Dict[str, Any]]:
    """
    Get cached tools for a server
    
    Args:
        manager: The McpManager instance
        server_id: The ID of the server
        
    Returns:
        List of tool details or empty list if no cache
    """
    return manager.server_tools_cache.get(server_id, [])