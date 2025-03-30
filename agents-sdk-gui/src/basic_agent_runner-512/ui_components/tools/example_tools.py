import streamlit as st
from typing import Dict, Any

def render_example_tools_form():
    """
    Renders a form for adding example tools
    
    Returns:
        Boolean indicating if any tools were added
    """
    with st.expander("Add Example Tools", expanded=False):
        st.write("These are fully functional example tools you can add to your agent:")
        
        tools_added = False
        
        # Weather tool
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Weather Tool**")
            st.markdown("*Get current weather information for a location*")
        with col2:
            if st.button("Add Weather Tool"):
                if "predefined_tools" not in st.session_state:
                    st.session_state.predefined_tools = {}
                
                from agent_management.agent_runner import get_weather
                
                tool_id = f"WeatherTool_{len(st.session_state.predefined_tools)}"
                st.session_state.predefined_tools[tool_id] = {
                    "type": "ExampleTool",
                    "name": "get_weather",
                    "instance": get_weather
                }
                st.success("Weather Tool added!")
                tools_added = True
        
        st.divider()
        
        # News search tool
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**News Search Tool**")
            st.markdown("*Search for news articles on a given topic*")
        with col2:
            if st.button("Add News Tool"):
                if "predefined_tools" not in st.session_state:
                    st.session_state.predefined_tools = {}
                
                from agent_management.agent_runner import search_news
                
                tool_id = f"NewsSearchTool_{len(st.session_state.predefined_tools)}"
                st.session_state.predefined_tools[tool_id] = {
                    "type": "ExampleTool",
                    "name": "search_news",
                    "instance": search_news
                }
                st.success("News Search Tool added!")
                tools_added = True
        
        return tools_added