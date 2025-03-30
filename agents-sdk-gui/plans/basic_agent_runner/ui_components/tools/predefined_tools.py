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
        
        st.markdown("""
        <div class="form-section">
            <h3>Add Standard Tool</h3>
            <div class="text-sm text-neutral-600 mb-xs">Configure and add predefined tools to your agent</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("add_predefined_tool_form"):
            # Tool type selection
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            tool_type = st.selectbox(
                "Tool Type", 
                options=["WebSearchTool", "FileSearchTool"],
                index=0,
                help="Type of predefined tool to add"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Divider for tool specific settings
            st.markdown("""
            <div class="form-section mt-md">
                <h4 class="text-primary font-semibold">Tool Configuration</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Fields based on tool type
            if tool_type == "WebSearchTool":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="form-group">', unsafe_allow_html=True)
                    location_type = st.selectbox(
                        "Location Type",
                        options=["approximate", "none"],
                        index=0,
                        help="Whether to include user location in searches"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if location_type == "approximate":
                    with col2:
                        st.markdown('<div class="form-group">', unsafe_allow_html=True)
                        city = st.text_input(
                            "City", 
                            "New York",
                            help="City to use for location context"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="form-group mt-sm">', unsafe_allow_html=True)
                search_context = st.select_slider(
                    "Search Context Size",
                    options=["low", "medium", "high"],
                    value="medium",
                    help="Amount of context to include with search results"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Create the tool configuration
                tool_config = {
                    "type": "WebSearchTool",
                    "user_location": {"type": location_type, "city": city} if location_type == "approximate" else None,
                    "search_context_size": search_context
                }
                
                # Show explanation
                st.markdown("""
                <div class="neutral-message mt-sm">
                    <strong>About WebSearchTool</strong>: Enables the agent to search the web for information. 
                    Provides search results based on queries generated from the conversation.
                </div>
                """, unsafe_allow_html=True)
                
            elif tool_type == "FileSearchTool":
                st.markdown('<div class="form-group">', unsafe_allow_html=True)
                vector_store_ids = st.text_input(
                    "Vector Store IDs (comma-separated)",
                    "",
                    help="IDs of vector stores to search in (required)"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="form-group">', unsafe_allow_html=True)
                    max_results = st.number_input(
                        "Max Results",
                        min_value=1,
                        max_value=50,
                        value=3,
                        help="Maximum number of results to return"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="form-group">', unsafe_allow_html=True)
                    include_search_results = st.checkbox(
                        "Include Search Results",
                        value=True,
                        help="Include the full text of search results"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Create the tool configuration
                tool_config = {
                    "type": "FileSearchTool",
                    "vector_store_ids": [id.strip() for id in vector_store_ids.split(",") if id.strip()],
                    "max_num_results": max_results,
                    "include_search_results": include_search_results
                }
                
                # Show explanation
                st.markdown("""
                <div class="neutral-message mt-sm">
                    <strong>About FileSearchTool</strong>: Enables the agent to search in vector stores for relevant files.
                    Useful for retrieving information from documents, code, or other text files.
                </div>
                """, unsafe_allow_html=True)
            
            # Submit button with better styling
            st.markdown('<div class="form-buttons mt-md">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submitted = st.form_submit_button("Add Tool")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if submitted:
                # Validate inputs
                if tool_type == "FileSearchTool" and not tool_config["vector_store_ids"]:
                    st.markdown("""
                    <div class="error-message">
                        Please provide at least one Vector Store ID
                    </div>
                    """, unsafe_allow_html=True)
                    return False
                
                # Store the tool in session state
                if "predefined_tools" not in st.session_state:
                    st.session_state.predefined_tools = {}
                
                tool_id = f"{tool_type}_{len(st.session_state.predefined_tools)}"
                st.session_state.predefined_tools[tool_id] = tool_config
                
                st.markdown(f"""
                <div class="success-message">
                    {tool_type} added successfully!
                </div>
                """, unsafe_allow_html=True)
                tool_added = True
        
        return tool_added