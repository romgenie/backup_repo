import asyncio
import json
import time
import os
from typing import Dict, Any, List, Optional, Set
import uuid
from pathlib import Path

# Using async_to_sync to handle async operations in synchronous Streamlit
from async_utils import async_to_sync

# For MCP server implementations
from agents.mcp import MCPServer, MCPServerStdio, MCPServerSse

class MCPManager:
    """
    Manages MCP server lifecycle and interactions
    """
    def __init__(self):
        """Initialize the MCP manager"""
        self.active_servers: Dict[str, MCPServer] = {}
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
    
    async def _create_server_from_config(self, config: Dict[str, Any], server_id: str = None) -> Optional[MCPServer]:
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
                
                server = MCPServerStdio(
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
                
                server = MCPServerSse(
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
            import traceback
            print(f"Error creating MCP server: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
            
    async def _create_server_async(self, server_id: str) -> Optional[MCPServer]:
        """
        Create an MCP server instance from configuration (async version)
        
        Args:
            server_id: The ID of the server to create
            
        Returns:
            The created server instance or None if creation failed
        """
        if server_id not in self.server_configs:
            print(f"Server ID {server_id} not found in server configs. Available IDs: {list(self.server_configs.keys())}")
            return None
        
        # Get the configuration
        config = self.server_configs[server_id]
        
        # Create the server using the helper method
        server = await self._create_server_from_config(config, server_id)
        
        # Store the server in our active servers if successful
        if server:
            self.active_servers[server_id] = server
            
        return server
    
    async def _close_server_async(self, server_id: str) -> bool:
        """
        Close an active MCP server (async version)
        
        Args:
            server_id: The ID of the server to close
            
        Returns:
            Success status
        """
        if server_id in self.active_servers:
            server = self.active_servers[server_id]
            
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
                del self.active_servers[server_id]
                print(f"Successfully removed server {server_id} from active servers")
                return True
            except Exception as e:
                print(f"Error closing MCP server {server_id}: {str(e)}")
                import traceback
                print(f"Server close traceback: {traceback.format_exc()}")
                
                # Still remove it from active servers
                try:
                    del self.active_servers[server_id]
                    print(f"Removed server {server_id} from active servers despite error")
                except Exception as del_err:
                    print(f"Error removing server from active_servers: {str(del_err)}")
                    
                return False
        
        print(f"Server {server_id} not found in active servers")
        return False
        
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
        # Create the server if needed - Get the raw server without wrapper
        if server_id not in self.active_servers:
            # Get the configuration
            if server_id not in self.server_configs:
                return {"success": False, "error": "Server configuration not found"}
                
            config = self.server_configs[server_id]
            
            # Create the server using our refactored helper method
            try:
                server = await self._create_server_from_config(config, server_id)
                if not server:
                    return {"success": False, "error": "Failed to create server instance"}
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                return {"success": False, "error": str(e), "traceback": error_trace}
        else:
            server = self.active_servers[server_id]
        
        try:
            # Connect to the server first
            if hasattr(server, 'connect'):
                await server.connect()
                
            # List the tools to test connection
            tools = await server.list_tools()
            
            # Process tools information
            tool_details = []
            tool_names = []
            for tool in tools:
                tool_names.append(tool.name)
                
                # Extract tool details for caching
                tool_info = {
                    "name": tool.name,
                    "description": getattr(tool, 'description', ''),
                    "parameters": getattr(tool, 'parameters', []),
                    "return_type": getattr(tool, 'return_type', None)
                }
                tool_details.append(tool_info)
            
            # Store tool details in cache
            self.server_tools_cache[server_id] = tool_details
            
            # Update server health status
            self.server_health[server_id] = {
                "last_check": time.time(),
                "status": "healthy",
                "tool_count": len(tool_names)
            }
            
            # Close the temporary server if it's not in active_servers
            if server_id not in self.active_servers and hasattr(server, 'close'):
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
            import traceback
            error_trace = traceback.format_exc()
            
            # Update server health status
            self.server_health[server_id] = {
                "last_check": time.time(),
                "status": "error",
                "error": str(e)
            }
            
            # Close the temporary server if it's not in active_servers
            if server_id not in self.active_servers and hasattr(server, 'close'):
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
    
    @async_to_sync
    async def test_server(self, server_id: str) -> Dict[str, Any]:
        """
        Test connection to an MCP server and list available tools
        
        Args:
            server_id: The ID of the server to test
            
        Returns:
            Dictionary with test results
        """
        # Call the async implementation directly
        return await self._test_server_async(server_id)
    
    # Add a new method for creating a server with sync wrapper
    @async_to_sync
    async def create_server(self, server_id: str) -> Optional[MCPServer]:
        """
        Create an MCP server instance from configuration (sync wrapper)
        
        Args:
            server_id: The ID of the server to create
            
        Returns:
            The created server instance or None if creation failed
        """
        return await self._create_server_async(server_id)
        
    async def _get_servers_for_agent_async(self, server_ids: List[str]) -> List[MCPServer]:
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
                            self.server_health[server_id] = {
                                "last_check": time.time(),
                                "status": "connected"
                            }
                            
                            # Try to list tools to confirm connectivity
                            try:
                                tools = await server.list_tools()
                                tool_names = [t.name for t in tools]
                                print(f"Server {server_id} has {len(tool_names)} tools: {tool_names}")
                                
                                # Update health status with tools info
                                self.server_health[server_id]["tool_count"] = len(tool_names)
                                
                                # Cache tool details
                                tool_details = []
                                for tool in tools:
                                    tool_info = {
                                        "name": tool.name,
                                        "description": getattr(tool, 'description', ''),
                                        "parameters": getattr(tool, 'parameters', []),
                                        "return_type": getattr(tool, 'return_type', None),
                                        "server_id": server_id
                                    }
                                    tool_details.append(tool_info)
                                self.server_tools_cache[server_id] = tool_details
                                
                            except Exception as te:
                                print(f"Error listing tools for server {server_id}: {str(te)}")
                                self.server_health[server_id]["tool_error"] = str(te)
                            
                        except Exception as e:
                            print(f"Error connecting to server {server_id}: {str(e)}")
                            # Update health status
                            self.server_health[server_id] = {
                                "last_check": time.time(),
                                "status": "connection_error",
                                "error": str(e)
                            }
                            # Continue anyway - maybe the server doesn't need explicit connection
                            
                    servers.append(server)
                    print(f"Successfully created server {server_id}")
                else:
                    print(f"Failed to create server {server_id}")
                    # Update health status
                    self.server_health[server_id] = {
                        "last_check": time.time(),
                        "status": "creation_failed"
                    }
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
                            self.server_health[server_id] = {
                                "last_check": time.time(),
                                "status": "healthy",
                                "tool_count": len(tools)
                            }
                    except Exception as e:
                        print(f"Error in health check for server {server_id}: {str(e)}")
                        self.server_health[server_id] = {
                            "last_check": time.time(),
                            "status": "error",
                            "error": str(e)
                        }
        
        print(f"Returning {len(servers)} MCP servers for agent")
        return servers
    
    @async_to_sync
    async def get_servers_for_agent(self, server_ids: List[str]) -> List[MCPServer]:
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
        for server_id in list(self.active_servers.keys()):
            await self._close_server_async(server_id)
            
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
        try:
            with open(file_path, 'w') as f:
                json.dump(self.server_configs, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving MCP configurations: {str(e)}")
            return False
            
    def load_configurations(self, file_path: str) -> bool:
        """
        Load server configurations from a JSON file
        
        Args:
            file_path: Path to load the configurations from
            
        Returns:
            Success status
        """
        try:
            with open(file_path, 'r') as f:
                configs = json.load(f)
                
            # Close any existing servers
            self.close_all_servers()
                
            # Update configurations
            self.server_configs = configs
            return True
        except Exception as e:
            print(f"Error loading MCP configurations: {str(e)}")
            return False
    
    async def _invalidate_tool_cache_async(self, server_id: str) -> bool:
        """
        Invalidate the tools cache for a server (async version)
        
        Args:
            server_id: The ID of the server
            
        Returns:
            Success status
        """
        if server_id in self.active_servers:
            server = self.active_servers[server_id]
            
            try:
                server.invalidate_tools_cache()
                return True
            except Exception as e:
                print(f"Error invalidating tools cache: {str(e)}")
                return False
        
        return False
        
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