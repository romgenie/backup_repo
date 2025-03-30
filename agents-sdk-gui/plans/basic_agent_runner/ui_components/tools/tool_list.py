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
    
    # Display available tools section header
    st.markdown("""
    <div class="form-section">
        <h3>Available Tools</h3>
        <div class="text-sm text-neutral-600 mb-xs">Select tools to include in your agent configuration</div>
    </div>
    """, unsafe_allow_html=True)
    
    tools_applied = False
    
    # Display predefined tools
    if st.session_state.predefined_tools:
        st.markdown('<div class="form-section mt-md">', unsafe_allow_html=True)
        st.markdown('<h4 class="text-primary font-semibold">Predefined Tools</h4>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create a copy of the dictionary to avoid modification during iteration
        predefined_tools_copy = dict(st.session_state.predefined_tools)
        
        for tool_id, tool_config in predefined_tools_copy.items():
            st.markdown(f'<div class="tool-item p-sm">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                selected = st.checkbox("", key=f"select_{tool_id}", value=True)
            
            with col2:
                st.markdown(f'<div class="tool-name">{tool_config["type"]}</div>', unsafe_allow_html=True)
                tool_details = []
                
                if tool_config["type"] == "WebSearchTool":
                    if tool_config.get("user_location"):
                        tool_details.append(f"Location: {tool_config['user_location'].get('city', 'None')}")
                    tool_details.append(f"Context: {tool_config.get('search_context_size', 'medium')}")
                elif tool_config["type"] == "FileSearchTool":
                    tool_details.append(f"Vector Stores: {', '.join(tool_config['vector_store_ids'])}")
                    tool_details.append(f"Max Results: {tool_config.get('max_num_results', 3)}")
                elif tool_config["type"] == "ExampleTool":
                    tool_details.append(f"Function: {tool_config['name']}")
                
                for detail in tool_details:
                    st.markdown(f'<div class="tool-description">{detail}</div>', unsafe_allow_html=True)
            
            with col3:
                if st.button("Remove", key=f"remove_{tool_id}", help="Remove this tool"):
                    del st.session_state.predefined_tools[tool_id]
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display custom tools
    if st.session_state.custom_tools:
        st.markdown('<div class="form-section mt-md">', unsafe_allow_html=True)
        st.markdown('<h4 class="text-primary font-semibold">Custom Tools</h4>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create a copy of the dictionary to avoid modification during iteration
        custom_tools_copy = dict(st.session_state.custom_tools)
        
        for tool_name, tool_config in custom_tools_copy.items():
            st.markdown(f'<div class="tool-item p-sm">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                selected = st.checkbox("", key=f"select_custom_{tool_name}", value=True)
            
            with col2:
                st.markdown(f'<div class="tool-name">{tool_name}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="tool-description">{tool_config["description"]}</div>', unsafe_allow_html=True)
                
                # Show params in a collapsible section
                with st.expander("Parameters"):
                    for param_name, param_info in tool_config['schema']['properties'].items():
                        st.markdown(f"""
                        <div class="mb-xs">
                            <span class="font-semibold">{param_name}</span> 
                            <span class="text-sm text-neutral-600">({param_info['type']})</span>: 
                            {param_info['description']}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col3:
                if st.button("Remove", key=f"remove_custom_{tool_name}", help="Remove this tool"):
                    del st.session_state.custom_tools[tool_name]
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # If no tools configured, show a message
    if not st.session_state.predefined_tools and not st.session_state.custom_tools:
        st.markdown("""
        <div class="neutral-message">
            No tools have been configured yet. Add predefined tools or create custom tools above.
        </div>
        """, unsafe_allow_html=True)
    
    # If tools were configured, show a button to apply them
    if st.session_state.predefined_tools or st.session_state.custom_tools:
        st.markdown('<div class="form-buttons mt-md">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Apply Selected Tools to Agent", help="Update the agent with the selected tools"):
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
                        st.markdown(f"""
                        <div class="info-message">
                            Custom tool '{tool_name}' would be included (actual function evaluation not implemented)
                        </div>
                        """, unsafe_allow_html=True)
                
                # Call the callback with selected tools
                on_tool_select(selected_tools)
                st.markdown("""
                <div class="success-message">
                    Tools successfully applied to agent configuration!
                </div>
                """, unsafe_allow_html=True)
                tools_applied = True
        st.markdown('</div>', unsafe_allow_html=True)
            
    return tools_applied