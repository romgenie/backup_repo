import streamlit as st
from typing import Dict, Any, List

def render_custom_tool_form():
    """
    Renders a form for creating a custom tool
    
    Returns:
        Boolean indicating if a tool was created
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
        
        tool_created = False
        
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
                tool_created = True
        
        return tool_created