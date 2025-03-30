import streamlit as st
import json
from typing import Dict, Any, List, Optional, Callable
from agents import Tool, FunctionTool, WebSearchTool, FileSearchTool, function_tool
from typing_extensions import Literal

# Import tools from the tools module
from agent_management.tools import get_weather, search_news, add_calendar_event, get_all_tools

def render_custom_tool_form():
    """
    Renders a form for creating a custom tool
    """
    with st.expander("Create Custom Tool", expanded=False):
        # Initialize parameters list in session state if not present
        if "tool_params" not in st.session_state:
            st.session_state.tool_params = [{"name": "", "type": "string", "description": "", "required": True}]
        
        # Tool parameters are managed outside the form
        st.subheader("Tool Parameters")
        
        params_list = st.session_state.tool_params
        
        # Show parameter list with remove buttons (outside the form)
        for i, param in enumerate(params_list):
            col1, col2, col3, col4, col5 = st.columns([2, 2, 4, 1, 1])
            
            with col1:
                param["name"] = st.text_input(
                    "Name", 
                    value=param["name"], 
                    key=f"param_name_{i}"
                )
            
            with col2:
                param["type"] = st.selectbox(
                    "Type", 
                    options=["string", "number", "integer", "boolean", "array", "object"],
                    index=["string", "number", "integer", "boolean", "array", "object"].index(param["type"]), 
                    key=f"param_type_{i}"
                )
            
            with col3:
                param["description"] = st.text_input(
                    "Description", 
                    value=param["description"], 
                    key=f"param_desc_{i}"
                )
            
            with col4:
                param["required"] = st.checkbox(
                    "Required", 
                    value=param["required"], 
                    key=f"param_req_{i}"
                )
            
            with col5:
                if i > 0 and st.button("✕", key=f"remove_param_{i}"):
                    params_list.pop(i)
                    st.session_state.tool_params = params_list
                    st.rerun()
        
        # Add parameter button (outside the form)
        if st.button("+ Add Parameter"):
            params_list.append({"name": "", "type": "string", "description": "", "required": True})
            st.session_state.tool_params = params_list
            st.rerun()
        
        # Save tool parameters to session state
        st.session_state.tool_params = params_list
        
        # Form for the tool metadata and function code
        with st.form("create_custom_tool_form"):
            tool_name = st.text_input("Tool Name", key="tool_name", 
                              help="Name of the tool (without spaces)")
            tool_description = st.text_area("Tool Description", key="tool_description",
                                   help="Detailed description of what the tool does")
            
            # Function implementation section
            st.subheader("Function Implementation")
            function_code = st.text_area(
                "Python Code", 
                value='''def tool_function(parameter1, parameter2):
    """This function implements the tool.
    
    Args:
        parameter1: Description of parameter1
        parameter2: Description of parameter2
        
    Returns:
        The result of the tool execution
    """
    # Your code here
    return f"Processed {parameter1} and {parameter2}"
''',
                height=200,
                key="function_code", 
                help="Python code that implements the tool function"
            )
            
            # Submit button for the form
            submitted = st.form_submit_button("Create Tool")
            
            if submitted and tool_name and tool_description:
                # Create a proper JSON schema for the tool parameters
                properties = {}
                required_params = []
                
                for param in params_list:
                    if param["name"]:  # Only process params with names
                        properties[param["name"]] = {
                            "type": param["type"],
                            "description": param["description"] or f"Parameter {param['name']}"
                        }
                        
                        if param["required"]:
                            required_params.append(param["name"])
                
                schema = {
                    "type": "object",
                    "properties": properties,
                }
                
                if required_params:
                    schema["required"] = required_params
                
                # Store the tool definition in session state
                if "custom_tools" not in st.session_state:
                    st.session_state.custom_tools = {}
                
                st.session_state.custom_tools[tool_name] = {
                    "name": tool_name,
                    "description": tool_description,
                    "schema": schema,
                    "function_code": function_code
                }
                
                st.success(f"Tool '{tool_name}' created successfully!")
                st.rerun()

def render_example_tools_form():
    """
    Renders a form for adding example tools
    """
    with st.expander("Add Example Tools", expanded=True):
        st.write("These are fully functional example tools you can add to your agent:")
        
        # Add All Tools button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**📦 All Tools Package**")
            st.markdown("*Add all available tools to your agent at once*")
        with col2:
            if st.button("Add All Tools", type="primary"):
                if "predefined_tools" not in st.session_state:
                    st.session_state.predefined_tools = {}
                
                # Initialize selected tools if needed
                if "selected_tools" not in st.session_state:
                    st.session_state.selected_tools = []
                
                # Get all available tools and add them
                all_tools = get_all_tools()
                for tool in all_tools:
                    tool_name = getattr(tool, 'name', str(tool))
                    tool_id = f"{tool_name}_{len(st.session_state.predefined_tools)}"
                    st.session_state.predefined_tools[tool_id] = {
                        "type": "ExampleTool",
                        "name": tool_name,
                        "instance": tool,
                        "description": tool.__doc__.split('\n')[0] if tool.__doc__ else "No description"
                    }
                    
                    # Add to selected tools if not already there
                    if tool not in st.session_state.selected_tools:
                        st.session_state.selected_tools.append(tool)
                
                st.success(f"Added all {len(all_tools)} tools to your agent!")
                st.rerun()
        
        st.divider()
        
        # Weather tool
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**☁️ Weather Tool**")
            st.markdown("*Get current weather information for a location*")
        with col2:
            if st.button("Add Weather Tool"):
                if "predefined_tools" not in st.session_state:
                    st.session_state.predefined_tools = {}
                
                tool_id = f"WeatherTool_{len(st.session_state.predefined_tools)}"
                st.session_state.predefined_tools[tool_id] = {
                    "type": "ExampleTool",
                    "name": "get_weather",
                    "instance": get_weather,
                    "description": "Get current weather information for a location"
                }
                
                # Automatically select the tool for the agent
                if "selected_tools" not in st.session_state:
                    st.session_state.selected_tools = []
                if get_weather not in st.session_state.selected_tools:
                    st.session_state.selected_tools.append(get_weather)
                
                st.success("Weather Tool added and selected for the agent!")
                st.rerun()
        
        st.divider()
        
        # News search tool
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**📰 News Search Tool**")
            st.markdown("*Search for news articles on a given topic*")
        with col2:
            if st.button("Add News Tool"):
                if "predefined_tools" not in st.session_state:
                    st.session_state.predefined_tools = {}
                
                tool_id = f"NewsSearchTool_{len(st.session_state.predefined_tools)}"
                st.session_state.predefined_tools[tool_id] = {
                    "type": "ExampleTool",
                    "name": "search_news",
                    "instance": search_news,
                    "description": "Search for news articles on a given topic"
                }
                
                # Automatically select the tool for the agent
                if "selected_tools" not in st.session_state:
                    st.session_state.selected_tools = []
                if search_news not in st.session_state.selected_tools:
                    st.session_state.selected_tools.append(search_news)
                
                st.success("News Search Tool added and selected for the agent!")
                st.rerun()
        
        st.divider()
        
        # Calendar tool
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**📅 Calendar Tool**")
            st.markdown("*Add events to your calendar*")
        with col2:
            if st.button("Add Calendar Tool"):
                if "predefined_tools" not in st.session_state:
                    st.session_state.predefined_tools = {}
                
                tool_id = f"CalendarTool_{len(st.session_state.predefined_tools)}"
                st.session_state.predefined_tools[tool_id] = {
                    "type": "ExampleTool",
                    "name": "add_calendar_event",
                    "instance": add_calendar_event,
                    "description": "Add events to your calendar"
                }
                
                # Automatically select the tool for the agent
                if "selected_tools" not in st.session_state:
                    st.session_state.selected_tools = []
                if add_calendar_event not in st.session_state.selected_tools:
                    st.session_state.selected_tools.append(add_calendar_event)
                
                st.success("Calendar Tool added and selected for the agent!")
                st.rerun()

def render_predefined_tool_form():
    """
    Renders a form for adding predefined tools
    """
    with st.expander("Add Predefined Tool", expanded=False):
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
                    st.stop()
                
                # Store the tool in session state
                if "predefined_tools" not in st.session_state:
                    st.session_state.predefined_tools = {}
                
                tool_id = f"{tool_type}_{len(st.session_state.predefined_tools)}"
                st.session_state.predefined_tools[tool_id] = tool_config
                
                st.success(f"{tool_type} added successfully!")
                st.rerun()

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
            # Return the tool instance directly - it's already a FunctionTool
            return tool_config["instance"]
        # Add more tool types as needed
    except Exception as e:
        st.error(f"Failed to create tool: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        return None

def display_tools_list(on_tool_select: Callable[[List[Tool]], None]):
    """
    Displays the list of configured tools
    
    Args:
        on_tool_select: Callback when tools are selected
    """
    # Initialize tool lists if not present
    if "custom_tools" not in st.session_state:
        st.session_state.custom_tools = {}
    
    if "predefined_tools" not in st.session_state:
        st.session_state.predefined_tools = {}
    
    # Display available tools
    st.subheader("Available Tools")
    
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

def render_tools_tab() -> None:
    """
    Render the tools tab of the application
    """
    st.header("Tools Configuration")
    
    # Callback for tool selection
    def on_tool_select(tools: List[Tool]):
        # Store selected tools in session state for the agent
        st.session_state.selected_tools = tools
    
    # Render tools forms and list
    render_example_tools_form()
    st.divider()
    render_predefined_tool_form()
    render_custom_tool_form()
    st.divider()
    display_tools_list(on_tool_select)