"""
Server lifecycle management functions for MCP servers.
"""

import asyncio
import traceback
import json
from typing import Dict, Any, List, Optional

# For MCP server implementations
from agents.mcp import MCPServer as McpServer, MCPServerStdio as McpServerStdio, MCPServerSse as McpServerSse

async def create_server_from_config(config: Dict[str, Any], server_id: str = None) -> Optional[McpServer]:
    """
    Create an MCP server instance from configuration
    
    Args:
        config: Server configuration dictionary
        server_id: Optional server ID for logging
        
    Returns:
        The created server instance or None if creation failed
    """
    server_type = config.get("type")
    server_name = config.get("name", "MCP Server")
    
    id_log = f" ({server_id})" if server_id else ""
    print(f"Creating server type: {server_type} with name: {server_name}{id_log}")
    
    try:            
        # Create the appropriate server type
        if server_type == "stdio":
            # Handle args as string or list
            args = config.get("args", "")
            if isinstance(args, str):
                args = args.split()
            
            print(f"Creating stdio server with command: {config.get('command', 'npx')} and args: {args}")
            
            server = McpServerStdio(
                name=server_name,
                params={
                    "command": config.get("command", "npx"),
                    "args": args,
                },
                cache_tools_list=config.get("cache_tools", True)
            )
        elif server_type == "sse":
            # Parse headers if they're provided as a string
            headers = config.get("headers", "{}")
            if isinstance(headers, str):
                try:
                    headers = json.loads(headers)
                except json.JSONDecodeError:
                    headers = {}
            
            print(f"Creating SSE server with URL: {config.get('url', '')}")
            
            server = McpServerSse(
                name=server_name,
                url=config.get("url", ""),
                headers=headers,
                cache_tools_list=config.get("cache_tools", True)
            )
        else:
            print(f"Unknown server type: {server_type}")
            return None
        
        print(f"Successfully created server: {server_name}")
        return server
        
    except Exception as e:
        print(f"Error creating MCP server: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

async def create_server_async(manager, server_id: str) -> Optional[McpServer]:
    """
    Create an MCP server instance from configuration (async version)
    
    Args:
        manager: The McpManager instance
        server_id: The ID of the server to create
        
    Returns:
        The created server instance or None if creation failed
    """
    if server_id not in manager.server_configs:
        print(f"Server ID {server_id} not found in server configs. Available IDs: {list(manager.server_configs.keys())}")
        return None
    
    # Get the configuration
    config = manager.server_configs[server_id]
    
    # Create the server using the helper method
    server = await create_server_from_config(config, server_id)
    
    # Store the server in our active servers if successful
    if server:
        manager.active_servers[server_id] = server
        
    return server

async def close_server_async(manager, server_id: str) -> bool:
    """
    Close an active MCP server (async version)
    
    Args:
        manager: The McpManager instance
        server_id: The ID of the server to close
        
    Returns:
        Success status
    """
    if server_id in manager.active_servers:
        server = manager.active_servers[server_id]
        
        try:
            # Try to disconnect first if the server supports it
            if hasattr(server, 'disconnect'):
                try:
                    print(f"Disconnecting MCP server {server_id}")
                    await server.disconnect()
                except Exception as disconnect_err:
                    print(f"Warning: Error disconnecting server: {str(disconnect_err)}")
                    # Continue with close even if disconnect fails
            
            # Then try to close if the server supports it
            if hasattr(server, 'close'):
                print(f"Closing MCP server {server_id}")
                await server.close()
                
            # Remove from active servers regardless of close() method
            del manager.active_servers[server_id]
            print(f"Successfully removed server {server_id} from active servers")
            return True
        except Exception as e:
            print(f"Error closing MCP server {server_id}: {str(e)}")
            print(f"Server close traceback: {traceback.format_exc()}")
            
            # Still remove it from active servers
            try:
                del manager.active_servers[server_id]
                print(f"Removed server {server_id} from active servers despite error")
            except Exception as del_err:
                print(f"Error removing server from active_servers: {str(del_err)}")
                
            return False
    
    print(f"Server {server_id} not found in active servers")
    return False

async def close_all_servers_async(manager) -> None:
    """
    Close all active MCP servers (async version)
    
    Args:
        manager: The McpManager instance
    """
    for server_id in list(manager.active_servers.keys()):
        await close_server_async(manager, server_id)