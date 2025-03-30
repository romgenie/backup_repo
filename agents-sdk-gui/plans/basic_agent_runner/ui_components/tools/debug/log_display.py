"""
MCP log collection and display components.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

def render_server_logs(mcp_manager) -> None:
    """
    Render the MCP server logs section
    
    Args:
        mcp_manager: MCP manager instance
    """
    st.subheader("MCP Server Logs")
    
    # Get server logs if available
    server_logs = getattr(mcp_manager, "server_logs", {})
    
    if not server_logs:
        st.info("No server logs are available. Logs will appear here as MCP servers are used.")
        return
    
    # Create tabs for each server
    server_ids = list(server_logs.keys())
    
    if not server_ids:
        st.info("No logs recorded for any MCP server yet")
        return
    
    # If we have multiple servers, create tabs
    if len(server_ids) > 1:
        # Add an "All Logs" option
        tabs = st.tabs(["All Logs"] + [get_server_name(mcp_manager, server_id) for server_id in server_ids])
        
        # All logs tab
        with tabs[0]:
            all_logs = []
            for server_id, logs in server_logs.items():
                server_name = get_server_name(mcp_manager, server_id)
                for log in logs:
                    log_copy = log.copy()
                    log_copy["Server"] = server_name
                    all_logs.append(log_copy)
            
            display_logs(all_logs)
        
        # Individual server tabs
        for i, server_id in enumerate(server_ids):
            with tabs[i+1]:
                display_logs(server_logs[server_id])
    else:
        # Just one server, no need for tabs
        display_logs(server_logs[server_ids[0]])

def display_logs(logs: List[Dict[str, Any]]) -> None:
    """
    Display a list of log entries
    
    Args:
        logs: List of log entries
    """
    if not logs:
        st.info("No logs recorded")
        return
    
    # Sort logs by timestamp (newest first)
    sorted_logs = sorted(logs, key=lambda x: x.get("timestamp", 0), reverse=True)
    
    # Convert to dataframe for display
    log_data = []
    for log in sorted_logs:
        # Format timestamp
        timestamp = log.get("timestamp", 0)
        if timestamp:
            time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_str = "Unknown"
        
        # Get level and message
        level = log.get("level", "INFO").upper()
        message = log.get("message", "")
        
        # Create entry
        entry = {
            "Time": time_str,
            "Level": level,
            "Message": message[:100] + ("..." if len(message) > 100 else "")
        }
        
        # Add server if available
        if "Server" in log:
            entry["Server"] = log["Server"]
        
        log_data.append(entry)
    
    # Create dataframe
    if log_data:
        df = pd.DataFrame(log_data)
        
        # Add styling based on log level
        def style_log_level(val):
            color_map = {
                "ERROR": "red",
                "WARNING": "orange",
                "INFO": "blue",
                "DEBUG": "gray"
            }
            
            color = color_map.get(val, "black")
            return f'color: {color}; font-weight: bold;'
        
        # Create the styled dataframe
        st.dataframe(
            df.style.applymap(style_log_level, subset=["Level"]),
            use_container_width=True
        )
        
        # Add log filter controls
        with st.expander("Log Filters"):
            # Filter by level
            levels = ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"]
            selected_level = st.selectbox("Filter by Level", levels)
            
            # Filter by message
            filter_text = st.text_input("Filter by Text")
            
            if selected_level != "ALL" or filter_text:
                # Filter data
                filtered_data = log_data.copy()
                
                if selected_level != "ALL":
                    filtered_data = [log for log in filtered_data if log["Level"] == selected_level]
                
                if filter_text:
                    filtered_data = [log for log in filtered_data if filter_text.lower() in log["Message"].lower()]
                
                # Show filtered data
                if filtered_data:
                    st.markdown(f"**Filtered Logs:** {len(filtered_data)} entries")
                    filtered_df = pd.DataFrame(filtered_data)
                    st.dataframe(
                        filtered_df.style.applymap(style_log_level, subset=["Level"]),
                        use_container_width=True
                    )
                else:
                    st.info("No logs match the current filters")
    else:
        st.info("No logs recorded")

def get_server_name(mcp_manager, server_id: str) -> str:
    """
    Get a friendly name for a server ID
    
    Args:
        mcp_manager: MCP manager instance
        server_id: Server ID to get name for
        
    Returns:
        Friendly server name
    """
    # Get server config
    server_config = mcp_manager.server_configs.get(server_id, {})
    return server_config.get("name", server_id)