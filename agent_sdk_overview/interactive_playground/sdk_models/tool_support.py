"""
Advanced tool support for the SDK Playground.
This module provides components for defining, testing, and visualizing tools.
"""
import streamlit as st
import json
from typing import Dict, Any, Optional, List, Union, Callable
from uuid import uuid4


class ToolDefinition:
    """Tool definition with parameters and function implementation."""
    
    TEMPLATE_TOOLS = {
        "weather": {
            "name": "get_weather",
            "description": "Get the current weather for a specified location",
            "schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature to use"
                    }
                },
                "required": ["location"]
            },
            "mock_responses": [
                {
                    "input": {"location": "San Francisco, CA", "unit": "celsius"},
                    "output": {
                        "temperature": 18,
                        "unit": "celsius",
                        "description": "Partly cloudy",
                        "humidity": 65
                    }
                },
                {
                    "input": {"location": "New York, NY", "unit": "fahrenheit"},
                    "output": {
                        "temperature": 72,
                        "unit": "fahrenheit",
                        "description": "Sunny",
                        "humidity": 45
                    }
                }
            ]
        },
        "calculator": {
            "name": "calculator",
            "description": "Perform a mathematical calculation",
            "schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '2 + 2'"
                    }
                },
                "required": ["expression"]
            },
            "mock_responses": [
                {
                    "input": {"expression": "2 + 2"},
                    "output": {"result": 4}
                },
                {
                    "input": {"expression": "sqrt(16)"},
                    "output": {"result": 4}
                },
                {
                    "input": {"expression": "sin(0)"},
                    "output": {"result": 0}
                }
            ]
        },
        "database_query": {
            "name": "query_database",
            "description": "Query a database for information",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute"
                    },
                    "database": {
                        "type": "string",
                        "description": "The database to query"
                    }
                },
                "required": ["query", "database"]
            },
            "mock_responses": [
                {
                    "input": {"query": "SELECT * FROM users LIMIT 3", "database": "users_db"},
                    "output": {
                        "results": [
                            {"id": 1, "name": "John Doe", "email": "john@example.com"},
                            {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
                            {"id": 3, "name": "Bob Johnson", "email": "bob@example.com"}
                        ],
                        "count": 3,
                        "query_time_ms": 15
                    }
                }
            ]
        },
        "web_search": {
            "name": "search_web",
            "description": "Search the web for information",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of results to return",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            },
            "mock_responses": [
                {
                    "input": {"query": "latest AI developments", "num_results": 2},
                    "output": {
                        "results": [
                            {
                                "title": "Latest Breakthroughs in Artificial Intelligence",
                                "url": "https://example.com/ai-news",
                                "snippet": "Recent developments in AI have shown promising results in..."
                            },
                            {
                                "title": "The Future of AI: Trends to Watch",
                                "url": "https://example.com/ai-trends",
                                "snippet": "As AI continues to evolve, several key trends are emerging..."
                            }
                        ],
                        "search_time_ms": 320
                    }
                }
            ]
        },
        "custom": {
            "name": "custom_tool",
            "description": "Custom tool definition",
            "schema": {
                "type": "object",
                "properties": {},
                "required": []
            },
            "mock_responses": [
                {
                    "input": {},
                    "output": {}
                }
            ]
        }
    }
    
    def __init__(
        self,
        name: str = "custom_tool",
        description: str = "",
        schema: Dict = None,
        mock_responses: List[Dict] = None,
        implementation: Optional[Callable] = None,
        template_name: Optional[str] = None
    ):
        """
        Initialize a tool definition.
        
        Args:
            name: Tool name
            description: Tool description
            schema: JSON schema for parameters
            mock_responses: List of mock input/output pairs
            implementation: Actual function implementation
            template_name: Name of template to use
        """
        if template_name and template_name in self.TEMPLATE_TOOLS:
            template = self.TEMPLATE_TOOLS[template_name]
            self.name = template["name"]
            self.description = template["description"]
            self.schema = template["schema"]
            self.mock_responses = template["mock_responses"]
        else:
            self.name = name
            self.description = description
            self.schema = schema or {"type": "object", "properties": {}, "required": []}
            self.mock_responses = mock_responses or [{"input": {}, "output": {}}]
        
        self.implementation = implementation
        self.id = str(uuid4())
    
    def to_function_tool(self) -> Dict:
        """
        Convert to a dictionary format suitable for the OpenAI function calling API.
        
        Returns:
            A dictionary with the tool definition
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema
            }
        }
    
    def to_sdk_tool(self) -> str:
        """
        Generate Python code for defining this tool with the SDK.
        
        Returns:
            Python code as a string
        """
        schema_str = json.dumps(self.schema, indent=4)
        
        code = f"""from agents.tool import FunctionTool

# Define the {self.name} tool
{self.name}_tool = FunctionTool(
    name="{self.name}",
    description="{self.description}",
    params_json_schema={schema_str},
    func=lambda params: {{
        # Implementation goes here
        # This is a placeholder
        return {{"result": "Placeholder result"}}
    }}
)
"""
        return code
    
    def execute_mock(self, params: Dict) -> Dict:
        """
        Execute a mock response based on input parameters.
        
        Args:
            params: Input parameters
            
        Returns:
            Mock output response
        """
        # Find the closest matching mock response
        best_match = None
        best_match_score = -1
        
        for mock in self.mock_responses:
            mock_input = mock["input"]
            score = 0
            
            # Score based on parameter matches
            for key, value in params.items():
                if key in mock_input and mock_input[key] == value:
                    score += 1
            
            if score > best_match_score:
                best_match = mock
                best_match_score = score
        
        # If we found a match, return its output
        if best_match:
            return best_match["output"]
        
        # Fallback to the first mock response
        if self.mock_responses:
            return self.mock_responses[0]["output"]
        
        # Last resort
        return {"result": "No mock response available"}


def tool_builder_ui() -> Optional[ToolDefinition]:
    """
    UI for building and editing tool definitions.
    
    Returns:
        A ToolDefinition if saved, or None
    """
    st.subheader("Tool Builder")
    
    # Template selection
    template_options = list(ToolDefinition.TEMPLATE_TOOLS.keys())
    template_names = [
        "Weather API",
        "Calculator",
        "Database Query",
        "Web Search",
        "Custom Tool"
    ]
    template_display = {k: v for k, v in zip(template_options, template_names)}
    
    selected_template = st.selectbox(
        "Select a tool template:",
        options=template_options,
        format_func=lambda x: template_display[x]
    )
    
    # Initialize tool from template
    if "current_tool" not in st.session_state or st.session_state.get("tool_template") != selected_template:
        tool = ToolDefinition(template_name=selected_template)
        st.session_state.current_tool = tool
        st.session_state.tool_template = selected_template
    else:
        tool = st.session_state.current_tool
    
    # Tool information
    col1, col2 = st.columns(2)
    with col1:
        tool_name = st.text_input("Tool Name:", value=tool.name)
    with col2:
        tool_description = st.text_input("Description:", value=tool.description)
    
    # Parameter schema editor
    st.subheader("Parameter Schema")
    schema_str = st.text_area(
        "Edit parameter schema (JSON):",
        value=json.dumps(tool.schema, indent=2),
        height=200
    )
    
    # Try to parse the schema
    try:
        schema = json.loads(schema_str)
        st.session_state.current_tool.schema = schema
        schema_valid = True
    except json.JSONDecodeError:
        st.error("Invalid JSON. Please check your syntax.")
        schema_valid = False
    
    # Mock responses
    st.subheader("Mock Responses")
    
    # Show existing responses
    for i, mock in enumerate(tool.mock_responses):
        with st.expander(f"Mock Response {i+1}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input:**")
                input_str = json.dumps(mock["input"], indent=2)
                st.code(input_str, language="json")
            with col2:
                st.markdown("**Output:**")
                output_str = json.dumps(mock["output"], indent=2)
                st.code(output_str, language="json")
    
    # Add new mock response
    with st.expander("Add New Mock Response"):
        col1, col2 = st.columns(2)
        with col1:
            new_input_str = st.text_area(
                "Input Parameters (JSON):",
                value=json.dumps({"param": "value"}, indent=2),
                height=150,
                key="new_mock_input"
            )
        with col2:
            new_output_str = st.text_area(
                "Output Response (JSON):",
                value=json.dumps({"result": "value"}, indent=2),
                height=150,
                key="new_mock_output"
            )
        
        if st.button("Add Mock Response"):
            try:
                new_input = json.loads(new_input_str)
                new_output = json.loads(new_output_str)
                
                st.session_state.current_tool.mock_responses.append({
                    "input": new_input,
                    "output": new_output
                })
                
                st.success("Mock response added!")
                st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON. Please check your syntax.")
    
    # Update the tool with edited values
    if schema_valid:
        tool.name = tool_name
        tool.description = tool_description
        
        # Generate SDK code
        st.subheader("SDK Tool Definition")
        st.code(tool.to_sdk_tool(), language="python")
        
        # Option to save this tool
        if st.button("Save Tool"):
            # Save to session state
            if "tools" not in st.session_state:
                st.session_state.tools = []
            
            # Check if this tool already exists
            existing_idx = None
            for i, existing_tool in enumerate(st.session_state.tools):
                if existing_tool.id == tool.id:
                    existing_idx = i
                    break
            
            if existing_idx is not None:
                st.session_state.tools[existing_idx] = tool
            else:
                st.session_state.tools.append(tool)
            
            st.success(f"Tool '{tool.name}' saved!")
            return tool
    
    return None


def tool_testing_ui(tool: ToolDefinition):
    """
    UI for testing a tool with mock responses.
    
    Args:
        tool: The tool to test
    """
    st.subheader(f"Test Tool: {tool.name}")
    st.markdown(tool.description)
    
    # Display parameter schema
    with st.expander("Parameter Schema"):
        st.code(json.dumps(tool.schema, indent=2), language="json")
    
    # Parameter input based on schema
    st.markdown("### Parameters")
    
    # Create input fields based on schema properties
    params = {}
    properties = tool.schema.get("properties", {})
    required = tool.schema.get("required", [])
    
    for param_name, param_schema in properties.items():
        param_type = param_schema.get("type", "string")
        description = param_schema.get("description", "")
        required_text = " (required)" if param_name in required else ""
        
        # Create different input types based on parameter type
        if param_type == "string":
            if "enum" in param_schema:
                # Dropdown for enum types
                options = param_schema["enum"]
                params[param_name] = st.selectbox(
                    f"{param_name}{required_text}: {description}",
                    options=options,
                    key=f"param_{tool.id}_{param_name}"
                )
            else:
                # Text input for strings
                params[param_name] = st.text_input(
                    f"{param_name}{required_text}: {description}",
                    key=f"param_{tool.id}_{param_name}"
                )
        elif param_type == "integer" or param_type == "number":
            min_val = param_schema.get("minimum", 0)
            max_val = param_schema.get("maximum", 100)
            params[param_name] = st.number_input(
                f"{param_name}{required_text}: {description}",
                min_value=float(min_val),
                max_value=float(max_val),
                key=f"param_{tool.id}_{param_name}"
            )
        elif param_type == "boolean":
            params[param_name] = st.checkbox(
                f"{param_name}{required_text}: {description}",
                key=f"param_{tool.id}_{param_name}"
            )
    
    # Execute button
    if st.button("Execute Tool"):
        # Validate required parameters
        missing_required = [param for param in required if not params.get(param)]
        
        if missing_required:
            st.error(f"Missing required parameters: {', '.join(missing_required)}")
        else:
            # Execute mock response
            response = tool.execute_mock(params)
            
            # Display the response
            st.markdown("### Response")
            st.code(json.dumps(response, indent=2), language="json")


def tool_execution_visualizer(tools: List[ToolDefinition]):
    """
    Visualize tool execution flow.
    
    Args:
        tools: List of tools to visualize
    """
    if not tools:
        st.info("No tools defined yet. Create tools in the Tool Builder tab.")
        return
    
    st.subheader("Tool Execution Flow")
    
    # Sample conversation with tool calls
    conversation = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What's the weather like in San Francisco and New York?"}
    ]
    
    # Add assistant response with tool calls
    tool_calls = []
    for idx, tool in enumerate(tools[:2]):  # Limit to first 2 tools for simplicity
        if tool.name == "get_weather":
            # Find weather tool and create sample tool calls
            locations = ["San Francisco, CA", "New York, NY"]
            for i, location in enumerate(locations):
                tool_calls.append({
                    "id": f"call_{idx}_{i}",
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "arguments": json.dumps({"location": location, "unit": "celsius"})
                    }
                })
    
    # If we don't have weather tools, use whatever tools we have
    if not tool_calls and tools:
        tool = tools[0]
        if tool.mock_responses:
            sample_input = tool.mock_responses[0]["input"]
            tool_calls.append({
                "id": "call_0_0",
                "type": "function",
                "function": {
                    "name": tool.name,
                    "arguments": json.dumps(sample_input)
                }
            })
    
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls
    }
    conversation.append(assistant_message)
    
    # Add tool responses
    for idx, tool_call in enumerate(tool_calls):
        tool_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        
        # Find the tool
        matching_tool = next((t for t in tools if t.name == tool_name), None)
        
        if matching_tool:
            response = matching_tool.execute_mock(args)
            conversation.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(response)
            })
    
    # Add final assistant response
    if tool_calls:
        if "get_weather" in [call["function"]["name"] for call in tool_calls]:
            final_response = "Based on the weather data, it's 18°C and partly cloudy in San Francisco, while it's 72°F and sunny in New York."
        else:
            final_response = "I've processed your request using the available tools."
        
        conversation.append({
            "role": "assistant",
            "content": final_response
        })
    
    # Display conversation with tool execution
    for message in conversation:
        role = message["role"]
        
        if role == "system":
            st.markdown(f"**System:**")
            st.markdown(f"> {message['content']}")
        
        elif role == "user":
            st.markdown(f"**User:**")
            st.markdown(f"> {message['content']}")
        
        elif role == "assistant":
            st.markdown(f"**Assistant:**")
            
            if message.get("tool_calls"):
                st.markdown("*I'll need to use tools to answer this question...*")
                
                for tool_call in message["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"])
                    
                    with st.expander(f"Tool Call: {func_name}"):
                        st.code(json.dumps(args, indent=2), language="json")
            
            elif message.get("content"):
                st.markdown(f"> {message['content']}")
        
        elif role == "tool":
            # Tool responses
            tool_call_id = message.get("tool_call_id", "")
            content = message.get("content", "{}")
            
            try:
                response_data = json.loads(content)
                # Find the original tool call to get the name
                original_call = next((tc for tc in assistant_message.get("tool_calls", []) 
                                     if tc["id"] == tool_call_id), None)
                
                if original_call:
                    func_name = original_call["function"]["name"]
                    st.markdown(f"**Tool Response** ({func_name}):")
                    st.code(json.dumps(response_data, indent=2), language="json")
            except json.JSONDecodeError:
                st.markdown(f"**Tool Response:**")
                st.markdown(f"> {content}")
    
    # Mermaid diagram for visualization
    st.subheader("Execution Diagram")
    
    mermaid_code = """
    sequenceDiagram
        participant User
        participant Assistant
        participant ToolSystem
    """
    
    # Add the flow based on conversation
    if len(conversation) >= 3:  # We have at least one tool call
        mermaid_code += """
        User->>Assistant: Ask question
        Assistant->>ToolSystem: Call tools
        """
        
        for idx, tool_call in enumerate(tool_calls):
            func_name = tool_call["function"]["name"]
            mermaid_code += f"""
        ToolSystem-->>ToolSystem: Execute {func_name}
            """
        
        mermaid_code += """
        ToolSystem-->>Assistant: Return tool results
        Assistant->>User: Provide answer with tool data
        """
    
    st.code(mermaid_code, language="mermaid")


def tool_support_ui():
    """Main UI component for tool support."""
    st.header("Advanced Tool Support")
    
    tab1, tab2, tab3 = st.tabs(["Tool Builder", "Tool Testing", "Execution Flow"])
    
    with tab1:
        tool = tool_builder_ui()
        
    with tab2:
        # Allow testing of all saved tools
        if "tools" in st.session_state and st.session_state.tools:
            tool_options = {t.name: i for i, t in enumerate(st.session_state.tools)}
            selected_tool_name = st.selectbox(
                "Select a tool to test:",
                options=list(tool_options.keys())
            )
            
            # Get the selected tool
            selected_tool_idx = tool_options[selected_tool_name]
            selected_tool = st.session_state.tools[selected_tool_idx]
            
            # Show the testing UI
            tool_testing_ui(selected_tool)
        else:
            st.info("No tools available for testing. Create tools in the Tool Builder tab.")
    
    with tab3:
        # Show tool execution visualization
        if "tools" in st.session_state and st.session_state.tools:
            tool_execution_visualizer(st.session_state.tools)
        else:
            st.info("No tools available for visualization. Create tools in the Tool Builder tab.")