"""
MCP Manager - Manages MCP server lifecycle and interactions.
"""

from typing import Dict, Any, List, Optional, Set
import asyncio
import json
import time
import os
from pathlib import Path

# Import the components
from .server_lifecycle import (
    create_server_from_config, 
    create_server_async, 
    close_server_async, 
    close_all_servers_async
)
from .tool_management import (
    get_tool_details,
    invalidate_tool_cache_async
)
from .configuration import (
    save_configurations_to_file,
    load_configurations_from_file
)
from .health_monitoring import (
    update_health_status,
    test_server_async
)

# Using async_to_sync to handle async operations in synchronous Streamlit
from async_utils import async_to_sync

# For MCP server implementations
from agents.mcp import MCPServer as McpServer, MCPServerStdio as McpServerStdio, MCPServerSse as McpServerSse

class McpManager:
    """
    Manages MCP server lifecycle and interactions
    """
    def __init__(self):
        """Initialize the MCP manager"""
        self.active_servers: Dict[str, McpServer] = {}
        self.server_configs: Dict[str, Dict[str, Any]] = {}
        self.server_health: Dict[str, Dict[str, Any]] = {}
        self.server_tools_cache: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_server_config(self, server_id: str, config: Dict[str, Any]) -> str:
        """
        Add a server configuration
        
        Args:
            server_id: Unique identifier for the server
            config: Server configuration dictionary
            
        Returns:
            The server ID
        """
        print(f"Adding server config with ID: {server_id}, config: {config}")
        self.server_configs[server_id] = config
        
        # Print current configs after addition
        config_ids = list(self.server_configs.keys())
        print(f"Current server configs after addition: {config_ids}")
        
        return server_id
    
    def remove_server_config(self, server_id: str) -> bool:
        """
        Remove a server configuration
        
        Args:
            server_id: Unique identifier for the server
            
        Returns:
            Success status
        """
        if server_id in self.server_configs:
            # Close the server if it's active
            self.close_server(server_id)
            
            # Remove the configuration
            del self.server_configs[server_id]
            return True
        return False
    
    async def _create_server_async(self, server_id: str) -> Optional[McpServer]:
        """
        Create an MCP server instance from configuration (async version)
        
        Args:
            server_id: The ID of the server to create
            
        Returns:
            The created server instance or None if creation failed
        """
        return await create_server_async(self, server_id)
        
    async def _close_server_async(self, server_id: str) -> bool:
        """
        Close an active MCP server (async version)
        
        Args:
            server_id: The ID of the server to close
            
        Returns:
            Success status
        """
        return await close_server_async(self, server_id)
        
    @async_to_sync
    async def close_server(self, server_id: str) -> bool:
        """
        Close an active MCP server (sync wrapper)
        
        Args:
            server_id: The ID of the server to close
            
        Returns:
            Success status
        """
        return await self._close_server_async(server_id)
    
    async def _test_server_async(self, server_id: str) -> Dict[str, Any]:
        """
        Test connection to an MCP server and list available tools (async version)
        
        Args:
            server_id: The ID of the server to test
            
        Returns:
            Dictionary with test results
        """
        return await test_server_async(self, server_id)
    
    @async_to_sync
    async def test_server(self, server_id: str) -> Dict[str, Any]:
        """
        Test connection to an MCP server and list available tools
        
        Args:
            server_id: The ID of the server to test
            
        Returns:
            Dictionary with test results
        """
        return await self._test_server_async(server_id)
    
    # Add a new method for creating a server with sync wrapper
    @async_to_sync
    async def create_server(self, server_id: str) -> Optional[McpServer]:
        """
        Create an MCP server instance from configuration (sync wrapper)
        
        Args:
            server_id: The ID of the server to create
            
        Returns:
            The created server instance or None if creation failed
        """
        return await self._create_server_async(server_id)
        
    async def _get_servers_for_agent_async(self, server_ids: List[str]) -> List[McpServer]:
        """
        Get a list of MCP servers for an agent (async version)
        
        Args:
            server_ids: List of server IDs to retrieve (can be str list or object)
            
        Returns:
            List of active MCP server instances
        """
        # Ensure server_ids is a list
        if not isinstance(server_ids, list):
            print(f"Warning: server_ids is not a list, converting. Type was: {type(server_ids)}")
            if hasattr(server_ids, 'keys') and callable(getattr(server_ids, 'keys')):
                server_ids = list(server_ids.keys())
            else:
                server_ids = [server_ids] if server_ids else []
        servers = []
        
        # Print debug info
        print(f"Getting {len(server_ids)} MCP servers for agent")
        
        for server_id in server_ids:
            # Create the server if it doesn't exist
            if server_id not in self.active_servers:
                print(f"Creating new MCP server for {server_id}")
                # Check if we have config for this server ID
                if server_id not in self.server_configs:
                    print(f"ERROR: Missing configuration for server {server_id}")
                    print(f"Available configs: {list(self.server_configs.keys())}")
                    continue
                
                server = await self._create_server_async(server_id)
                if server:
                    # Initialize the server by connecting it
                    if hasattr(server, 'connect'):
                        try:
                            print(f"Connecting to server {server_id}")
                            await server.connect()
                            
                            # Update health status
                            update_health_status(self, server_id, {
                                "status": "connected",
                                "last_check": time.time()
                            })
                            
                            # Try to list tools to confirm connectivity
                            try:
                                tools = await server.list_tools()
                                tool_names = [t.name for t in tools]
                                print(f"Server {server_id} has {len(tool_names)} tools: {tool_names}")
                                
                                # Update health status with tools info
                                self.server_health[server_id]["tool_count"] = len(tool_names)
                                
                                # Cache tool details
                                tool_details = get_tool_details(tools, server_id)
                                self.server_tools_cache[server_id] = tool_details
                                
                            except Exception as te:
                                print(f"Error listing tools for server {server_id}: {str(te)}")
                                self.server_health[server_id]["tool_error"] = str(te)
                            
                        except Exception as e:
                            print(f"Error connecting to server {server_id}: {str(e)}")
                            # Update health status
                            update_health_status(self, server_id, {
                                "status": "connection_error",
                                "error": str(e),
                                "last_check": time.time()
                            })
                            # Continue anyway - maybe the server doesn't need explicit connection
                            
                    servers.append(server)
                    print(f"Successfully created server {server_id}")
                else:
                    print(f"Failed to create server {server_id}")
                    # Update health status
                    update_health_status(self, server_id, {
                        "status": "creation_failed",
                        "last_check": time.time()
                    })
            else:
                print(f"Using existing MCP server for {server_id}")
                servers.append(self.active_servers[server_id])
                
                # Check if existing server is healthy (last check > 5 minutes ago)
                last_check = self.server_health.get(server_id, {}).get("last_check", 0)
                if time.time() - last_check > 300:  # 5 minutes in seconds
                    print(f"Health check for existing server {server_id} (last check was {time.time() - last_check} seconds ago)")
                    try:
                        # Simple connectivity check
                        if hasattr(servers[-1], 'list_tools'):
                            tools = await servers[-1].list_tools()
                            update_health_status(self, server_id, {
                                "status": "healthy",
                                "tool_count": len(tools),
                                "last_check": time.time()
                            })
                    except Exception as e:
                        print(f"Error in health check for server {server_id}: {str(e)}")
                        update_health_status(self, server_id, {
                            "status": "error",
                            "error": str(e),
                            "last_check": time.time()
                        })
        
        print(f"Returning {len(servers)} MCP servers for agent")
        return servers
    
    @async_to_sync
    async def get_servers_for_agent(self, server_ids: List[str]) -> List[McpServer]:
        """
        Get a list of MCP servers for an agent (sync wrapper)
        
        Args:
            server_ids: List of server IDs to retrieve
            
        Returns:
            List of active MCP server instances
        """
        return await self._get_servers_for_agent_async(server_ids)
    
    async def _close_all_servers_async(self) -> None:
        """Close all active MCP servers (async version)"""
        await close_all_servers_async(self)
            
    @async_to_sync
    async def close_all_servers(self) -> None:
        """Close all active MCP servers (sync wrapper)"""
        await self._close_all_servers_async()
    
    def get_server_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all server configurations
        
        Returns:
            Dictionary of server configurations
        """
        return self.server_configs.copy()
        
    def save_configurations(self, file_path: str) -> bool:
        """
        Save server configurations to a JSON file
        
        Args:
            file_path: Path to save the configurations
            
        Returns:
            Success status
        """
        return save_configurations_to_file(self, file_path)
            
    def load_configurations(self, file_path: str) -> bool:
        """
        Load server configurations from a JSON file
        
        Args:
            file_path: Path to load the configurations from
            
        Returns:
            Success status
        """
        return load_configurations_from_file(self, file_path)
    
    async def _invalidate_tool_cache_async(self, server_id: str) -> bool:
        """
        Invalidate the tools cache for a server (async version)
        
        Args:
            server_id: The ID of the server
            
        Returns:
            Success status
        """
        return await invalidate_tool_cache_async(self, server_id)
        
    @async_to_sync
    async def invalidate_tool_cache(self, server_id: str) -> bool:
        """
        Invalidate the tools cache for a server (sync wrapper)
        
        Args:
            server_id: The ID of the server
            
        Returns:
            Success status
        """
        return await self._invalidate_tool_cache_async(server_id)