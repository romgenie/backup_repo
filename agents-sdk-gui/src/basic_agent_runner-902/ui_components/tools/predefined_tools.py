import streamlit as st
from typing import Dict, Any

def render_predefined_tool_form():
    """
    Renders a form for adding predefined tools
    
    Returns:
        Boolean indicating if a tool was added
    """
    with st.expander("Add Predefined Tool", expanded=False):
        tool_added = False
        
        with st.form("add_predefined_tool_form"):
            tool_type = st.selectbox(
                "Tool Type", 
                options=["WebSearchTool", "FileSearchTool"],
                index=0,
                help="Type of predefined tool to add"
            )
            
            # Fields based on tool type
            if tool_type == "WebSearchTool":
                location_type = st.selectbox(
                    "Location Type",
                    options=["approximate", "none"],
                    index=0
                )
                
                if location_type == "approximate":
                    city = st.text_input("City", "New York")
                    
                search_context = st.select_slider(
                    "Search Context Size",
                    options=["low", "medium", "high"],
                    value="medium"
                )
                
                # Create the tool configuration
                tool_config = {
                    "type": "WebSearchTool",
                    "user_location": {"type": location_type, "city": city} if location_type == "approximate" else None,
                    "search_context_size": search_context
                }
                
            elif tool_type == "FileSearchTool":
                vector_store_ids = st.text_input(
                    "Vector Store IDs (comma-separated)",
                    ""
                )
                
                max_results = st.number_input(
                    "Max Results",
                    min_value=1,
                    max_value=50,
                    value=3
                )
                
                include_search_results = st.checkbox(
                    "Include Search Results",
                    value=True
                )
                
                # Create the tool configuration
                tool_config = {
                    "type": "FileSearchTool",
                    "vector_store_ids": [id.strip() for id in vector_store_ids.split(",") if id.strip()],
                    "max_num_results": max_results,
                    "include_search_results": include_search_results
                }
            
            # Submit button
            submitted = st.form_submit_button("Add Tool")
            
            if submitted:
                # Validate inputs
                if tool_type == "FileSearchTool" and not tool_config["vector_store_ids"]:
                    st.error("Please provide at least one Vector Store ID")
                    return False
                
                # Store the tool in session state
                if "predefined_tools" not in st.session_state:
                    st.session_state.predefined_tools = {}
                
                tool_id = f"{tool_type}_{len(st.session_state.predefined_tools)}"
                st.session_state.predefined_tools[tool_id] = tool_config
                
                st.success(f"{tool_type} added successfully!")
                tool_added = True
        
        return tool_added