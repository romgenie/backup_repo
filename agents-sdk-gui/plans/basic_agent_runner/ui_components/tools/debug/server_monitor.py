"""
MCP server health monitoring components.
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

def render_server_health_status(mcp_manager) -> None:
    """
    Render the MCP server health status section
    
    Args:
        mcp_manager: MCP manager instance
    """
    st.subheader("Server Health Status")
    
    # Get server health data
    server_health = mcp_manager.server_health
    
    if server_health:
        # Prepare data
        health_data = []
        for server_id, health_info in server_health.items():
            # Get server config
            server_config = mcp_manager.server_configs.get(server_id, {})
            server_name = server_config.get("name", "Unknown")
            
            # Get health status
            status = health_info.get("status", "unknown")
            last_check = health_info.get("last_check", 0)
            tool_count = health_info.get("tool_count", 0)
            error = health_info.get("error", "")
            
            # Format timestamp
            if last_check:
                last_check_time = datetime.fromtimestamp(last_check).strftime("%Y-%m-%d %H:%M:%S")
            else:
                last_check_time = "Never"
                
            # Create status emoji
            if status == "healthy" or status == "connected":
                status_emoji = "✅"
            elif status == "error" or status == "connection_error" or status == "creation_failed":
                status_emoji = "❌"
            else:
                status_emoji = "⚠️"
                
            # Add to data
            health_data.append({
                "Server ID": server_id,
                "name": server_name,
                "Status": f"{status_emoji} {status}",
                "Tools": tool_count,
                "Last Check": last_check_time,
                "Error": error
            })
            
        # Create dataframe and display
        if health_data:
            df = pd.DataFrame(health_data)
            st.dataframe(df, use_container_width=True)
            
            # Add refresh button
            if st.button("Refresh Health Status"):
                # This will force a rerun
                st.rerun()
        else:
            st.info("No server health data available yet")
    else:
        st.info("No MCP servers have been used yet")
        
def render_cache_management(mcp_manager) -> None:
    """
    Render the MCP tool cache management section
    
    Args:
        mcp_manager: MCP manager instance
    """
    # Create tabs for cache management
    st.subheader("Cache Management")
    cache_tabs = st.tabs(["Cache Status", "Clear Cache"])
    
    # Get tool cache data
    tool_cache = mcp_manager.server_tools_cache
    
    # Cache status tab
    with cache_tabs[0]:
        if tool_cache:
            # Create a summary
            cache_info = []
            total_tools = 0
            
            for server_id, tools in tool_cache.items():
                # Get server config
                server_config = mcp_manager.server_configs.get(server_id, {})
                server_name = server_config.get("name", "Unknown")
                
                # Count tools
                tool_count = len(tools)
                total_tools += tool_count
                
                # Add to cache info
                cache_info.append({
                    "Server ID": server_id,
                    "name": server_name,
                    "Cached Tools": tool_count
                })
                
            # Display info
            if cache_info:
                # Show total tools cached
                st.metric("Total Cached Tools", total_tools)
                
                # Show cache size estimation (rough estimate)
                import sys
                import json
                
                # Convert to JSON to estimate size
                cache_json = json.dumps(tool_cache)
                cache_size_bytes = len(cache_json.encode('utf-8'))
                cache_size_kb = cache_size_bytes / 1024
                
                # Display cache size
                st.metric("Cache Size", f"{cache_size_kb:.2f} KB")
                
                # Also show categorization cache stats if available
                try:
                    from ui_components.tools.mcp.categorization import _tool_category_cache, MCP_TOOL_CACHE_SIZE
                    cat_cache_size = len(_tool_category_cache.cache)
                    cat_cache_percent = (cat_cache_size / MCP_TOOL_CACHE_SIZE) * 100
                    
                    st.metric("Tool Categorization Cache", f"{cat_cache_size} / {MCP_TOOL_CACHE_SIZE} entries ({cat_cache_percent:.1f}%)")
                except ImportError:
                    pass
                
                # Show detail table
                st.markdown("**Cache Contents:**")
                df = pd.DataFrame(cache_info)
                st.dataframe(df, use_container_width=True)
                
                # Show tools for each server
                for server_id, tools in tool_cache.items():
                    with st.expander(f"Tools for {server_id}"):
                        # Show tool names
                        for i, tool in enumerate(tools):
                            st.markdown(f"{i+1}. **{tool.get('name', 'Unknown')}**")
                            
                            # Show tool details
                            if "description" in tool:
                                st.markdown(f"   Description: {tool['description']}")
                                
                            # Show parameter count
                            params = tool.get("parameters", [])
                            st.markdown(f"   Parameters: {len(params)}")
                            
                            # Show divider
                            if i < len(tools) - 1:
                                st.divider()
            else:
                st.info("No tool cache data available")
        else:
            st.info("No tools have been cached yet")
            
    # Clear cache tab
    with cache_tabs[1]:
        if not tool_cache:
            st.info("No cached tools to clear")
        else:
            st.warning("Clearing the tool cache will force tools to be re-fetched from MCP servers.")
            
            # Option to clear all caches
            st.markdown("#### Clear All Caches")
            
            # Add a confirmation checkbox
            confirm_all = st.checkbox("I understand this will clear all tool caches", key="confirm_clear_all")
            
            clear_all_button = st.button("Clear All Tool Caches", disabled=not confirm_all)
            if clear_all_button and confirm_all:
                # Clear the tool cache
                for server_id in list(tool_cache.keys()):
                    mcp_manager.invalidate_tool_cache(server_id)
                
                # Clear the cache dictionary
                mcp_manager.server_tools_cache.clear()
                
                # Also clear tool categorization cache if available
                try:
                    from ui_components.tools.mcp.categorization import _tool_category_cache
                    _tool_category_cache.cache.clear()
                    _tool_category_cache.usage.clear()
                    st.success("All caches cleared successfully (tool cache and categorization cache)")
                except ImportError:
                    st.success("Tool cache cleared successfully")
                
                st.rerun()
        
            # Selective clearing tab
            with st.expander("Clear Specific Server Caches"):
                # Select servers to clear
                selected_servers = st.multiselect(
                    "Select Servers to Clear Cache", 
                    options=list(tool_cache.keys()),
                    format_func=lambda x: x
                )
                
                if selected_servers:
                    confirm_selected = st.checkbox("I understand this will clear the selected caches", key="confirm_clear_selected")
                    
                    clear_selected = st.button("Clear Selected Caches", disabled=not confirm_selected)
                    if clear_selected and confirm_selected:
                        for server_id in selected_servers:
                            # Invalidate the cache
                            mcp_manager.invalidate_tool_cache(server_id)
                            
                            # Also remove from the cache dictionary
                            if server_id in mcp_manager.server_tools_cache:
                                del mcp_manager.server_tools_cache[server_id]
                                
                        st.success(f"Cleared caches for {len(selected_servers)} servers")
                        st.rerun()