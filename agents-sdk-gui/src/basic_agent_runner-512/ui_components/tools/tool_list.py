import streamlit as st
from typing import Dict, Any, List, Optional, Callable
from agents import Tool, WebSearchTool, FileSearchTool

def create_actual_tool(tool_config: Dict[str, Any]) -> Optional[Tool]:
    """
    Creates an actual tool instance from a tool configuration
    
    Args:
        tool_config: The tool configuration dictionary
        
    Returns:
        A Tool instance or None if creation failed
    """
    try:
        if tool_config["type"] == "WebSearchTool":
            return WebSearchTool(
                user_location=tool_config.get("user_location"),
                search_context_size=tool_config.get("search_context_size", "medium")
            )
        elif tool_config["type"] == "FileSearchTool":
            return FileSearchTool(
                vector_store_ids=tool_config["vector_store_ids"],
                max_num_results=tool_config.get("max_num_results", 3),
                include_search_results=tool_config.get("include_search_results", True)
            )
        elif tool_config["type"] == "ExampleTool":
            # Return the tool instance directly
            return tool_config["instance"]
        # Add more tool types as needed
    except Exception as e:
        st.error(f"Failed to create tool: {str(e)}")
        return None

def display_tools_list(on_tool_select: Callable[[List[Tool]], None]):
    """
    Displays the list of configured tools
    
    Args:
        on_tool_select: Callback when tools are selected
        
    Returns:
        Boolean indicating if tools were applied to the agent
    """
    # Initialize tool lists if not present
    if "custom_tools" not in st.session_state:
        st.session_state.custom_tools = {}
    
    if "predefined_tools" not in st.session_state:
        st.session_state.predefined_tools = {}
    
    # Display available tools
    st.subheader("Available Tools")
    
    tools_applied = False
    
    # Display predefined tools
    if st.session_state.predefined_tools:
        st.write("Predefined Tools:")
        for tool_id, tool_config in st.session_state.predefined_tools.items():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                selected = st.checkbox("", key=f"select_{tool_id}", value=True)
            
            with col2:
                st.markdown(f"**{tool_config['type']}**")
                if tool_config["type"] == "WebSearchTool":
                    if tool_config.get("user_location"):
                        st.markdown(f"Location: {tool_config['user_location'].get('city', 'None')}")
                    st.markdown(f"Context: {tool_config.get('search_context_size', 'medium')}")
                elif tool_config["type"] == "FileSearchTool":
                    st.markdown(f"Vector Stores: {', '.join(tool_config['vector_store_ids'])}")
                    st.markdown(f"Max Results: {tool_config.get('max_num_results', 3)}")
                elif tool_config["type"] == "ExampleTool":
                    st.markdown(f"Function: {tool_config['name']}")
            
            with col3:
                if st.button("Remove", key=f"remove_{tool_id}"):
                    del st.session_state.predefined_tools[tool_id]
                    st.rerun()
    
    # Display custom tools
    if st.session_state.custom_tools:
        st.write("Custom Tools:")
        for tool_name, tool_config in st.session_state.custom_tools.items():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                selected = st.checkbox("", key=f"select_custom_{tool_name}", value=True)
            
            with col2:
                st.markdown(f"**{tool_name}**")
                st.markdown(f"*{tool_config['description']}*")
                
                # Show params in a collapsible section
                with st.expander("Parameters"):
                    for param_name, param_info in tool_config['schema']['properties'].items():
                        st.markdown(f"**{param_name}** ({param_info['type']}): {param_info['description']}")
            
            with col3:
                if st.button("Remove", key=f"remove_custom_{tool_name}"):
                    del st.session_state.custom_tools[tool_name]
                    st.rerun()
    
    # If tools were configured, show a button to apply them
    if st.session_state.predefined_tools or st.session_state.custom_tools:
        if st.button("Apply Selected Tools to Agent"):
            # Collect selected tools
            selected_tools = []
            
            # Add selected predefined tools
            for tool_id, tool_config in st.session_state.predefined_tools.items():
                if st.session_state.get(f"select_{tool_id}", False):
                    tool = create_actual_tool(tool_config)
                    if tool:
                        selected_tools.append(tool)
            
            # Add selected custom tools
            for tool_name, tool_config in st.session_state.custom_tools.items():
                if st.session_state.get(f"select_custom_{tool_name}", False):
                    # We'd need to evaluate the function code to create actual tools
                    # This is a simplified version for demo purposes
                    st.info(f"Custom tool '{tool_name}' would be included (actual function evaluation not implemented)")
            
            # Call the callback with selected tools
            on_tool_select(selected_tools)
            st.success("Tools applied to agent configuration!")
            tools_applied = True
            
    return tools_applied