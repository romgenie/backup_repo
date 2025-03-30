import streamlit as st
import json
import pandas as pd
import inspect
import os
import sys
from enum import Enum
import dataclasses
import streamlit_mermaid as stmd

# Import the refactored playground modules
from interactive_playground.playground import display_playground
from interactive_playground.sdk_models.showcase_wrapper import run_showcase

# Add path to the OpenAI agents SDK
openai_agents_path = "/Users/timgregg/mcp/Github/openai/openai-agents-python/src"
if openai_agents_path not in sys.path:
    sys.path.append(openai_agents_path)

try:
    # Import the modules with corrected names based on the file listing
    from agents.models.interface import Model, ModelProvider, ModelTracing
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from agents.models.openai_provider import OpenAIProvider
    from agents.models.openai_responses import OpenAIResponsesModel
    from agents.models._openai_shared import get_default_openai_client  # Changed to correct function name
    from agents.model_settings import ModelSettings
    from agents.items import ModelResponse
    from agents.tool import Tool, FunctionTool
    
    import_success = True
except ImportError as e:
    import_success = False
    import_error = str(e)

# Set page config
st.set_page_config(
    page_title="OpenAI Agents SDK Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title
st.title("OpenAI Agents SDK Explorer")
st.markdown("### Module Focus: `models`")
st.info("This interactive tool will help you understand the OpenAI Agents SDK structure and functionality, focusing on the models module.")

if not import_success:
    st.error(f"Failed to import SDK modules: {import_error}")
    st.markdown("""
    ### Troubleshooting SDK Import
    
    The application failed to import the required SDK modules. This could be due to:
    
    1. The SDK path is incorrect
    2. The SDK version you have installed has different class or module names
    3. Missing dependencies
    
    Let's check what modules and classes are actually available:
    """)
    
    # Try to explore what's available in the SDK
    try:
        if openai_agents_path not in sys.path:
            sys.path.append(openai_agents_path)
            
        import agents
        
        # Check models module structure
        st.markdown("### SDK Structure Check")
        
        # Check if models module exists
        if hasattr(agents, 'models'):
            st.success("✅ `agents.models` module found")
            
            # List available modules in models
            models_dir = os.path.join(openai_agents_path, 'agents', 'models')
            if os.path.exists(models_dir):
                modules = [f for f in os.listdir(models_dir) if f.endswith('.py')]
                st.write("Available modules in models directory:", modules)
                
                # Try to import specific modules to see exact import names
                for module_name in modules:
                    try:
                        module_path = f"agents.models.{os.path.splitext(module_name)[0]}"
                        module = __import__(module_path, fromlist=['*'])
                        st.success(f"✅ Successfully imported `{module_path}`")
                        
                        # List classes and functions in the module
                        classes = [name for name, obj in inspect.getmembers(module) if inspect.isclass(obj)]
                        functions = [name for name, obj in inspect.getmembers(module) if inspect.isfunction(obj)]
                        
                        st.write(f"Classes in {os.path.splitext(module_name)[0]}:", classes)
                        st.write(f"Functions in {os.path.splitext(module_name)[0]}:", functions)
                    except ImportError as e:
                        st.error(f"❌ Failed to import {module_name}: {e}")
            
        else:
            st.error("❌ `agents.models` module not found")
            
    except ImportError as e:
        st.error(f"❌ Failed to import base 'agents' module: {e}")
        
    st.markdown("""
    ### Next Steps
    
    Based on the information above:
    
    1. Update the import statements in the code to match the actual module and class names
    2. Make sure the SDK is correctly installed and accessible
    3. Check if you need to install additional dependencies
    
    You can manually browse the SDK code at:
    """)
    st.code(openai_agents_path)
    
    # Exit early if imports failed
    st.stop()

# Sidebar for module selection
st.sidebar.markdown("## Navigation")
section = st.sidebar.radio(
    "Select a section to explore:",
    [
        "SDK Overview",
        "Core Interfaces",
        "Model Implementation",
        "Model Settings",
        "Tool Integration",
        "Response Handling",
        "Playground",
        "Advanced Playground"
    ]
)

# Function to display class details
def display_class_details(cls, show_methods=True):
    st.markdown(f"### Class: `{cls.__name__}`")
    
    # Get the docstring
    if cls.__doc__:
        st.markdown(f"**Description:**  \n{cls.__doc__}")
    
    # Display inheritance
    if cls.__bases__ and cls.__bases__[0] != object:
        st.markdown(f"**Inherits from:** {', '.join([base.__name__ for base in cls.__bases__])}")
    
    # Display attributes and methods
    if show_methods:
        methods = []
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith('_') or name == '__init__':
                methods.append((name, method))
        
        if methods:
            st.markdown("#### Methods")
            for name, method in methods:
                signature = str(inspect.signature(method))
                doc = method.__doc__ or ""
                st.markdown(f"**`{name}{signature}`**  \n{doc}")

# Function to display dataclass fields
def display_dataclass_fields(cls):
    st.markdown("#### Fields")
    for field in dataclasses.fields(cls):
        field_type = field.type
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            options = [f"`{e.name}` - {e.value}" for e in field_type]
            st.markdown(f"**`{field.name}`**: {field_type.__name__}  \n" + 
                       f"Options: {', '.join(options)}")
        else:
            doc = field.__doc__ or ""
            st.markdown(f"**`{field.name}`**: {field.type}  \n{doc}")

# Function to visualize the relationship between classes
def generate_model_relationship_diagram():
    st.markdown("### SDK Model Architecture")
    
    # Mermaid diagram showing the relationship between classes
    mermaid_code = """
    graph TD
        A[ModelProvider] --> B[Model]
        C[OpenAIProvider] -.-> A
        D[OpenAIChatCompletionsModel] -.-> B
        E[OpenAIResponsesModel] -.-> B
        C --> D
        C --> E
        F[ModelSettings] --> B
        G[Tool] --> B
        H[ModelResponse] <-- Output --- B
        I[TResponseInputItem] --> B
        J[AgentOutputSchema] --> B
        K[ModelTracing] --> B
        subgraph Core Interfaces
            A
            B
        end
        subgraph Implementations
            C
            D
            E
        end
        subgraph Settings & Config
            F
            K
        end
        subgraph Tools
            G
        end
        subgraph Data Flow
            I --> B
            B --> H
        end
    """
    
    # Display using both methods
    
    # 1. Standard markdown for Mermaid diagrams
    st.markdown("#### Mermaid via Markdown:")
    st.markdown(f"```mermaid\n{mermaid_code}\n```")
    
    # 2. Using the streamlit-mermaid component
    st.markdown("#### Mermaid via Component:")
    stmd.st_mermaid(mermaid_code)

if section == "SDK Overview":
    st.header("SDK Overview")
    
    st.markdown("""
    ### OpenAI Agents Python SDK
    
    The OpenAI Agents Python SDK is a framework for building applications with LLM agents. It provides abstractions to:
    
    1. **Define agent capabilities** - Through tools, model configurations, and output schemas
    2. **Execute agent runs** - Calling LLMs with appropriate context, handling input and output processing
    3. **Streamline workflows** - Managing complex agent interactions and tool usage
    
    ### Module Focus: `models`
    
    In this explorer, we'll focus on the `models` module which provides abstractions for:
    
    * Interfacing with OpenAI models (and potentially other LLM providers)
    * Constructing and sending appropriate API requests
    * Processing and transforming responses
    * Handling streaming and chunked responses
    * Managing model settings and configurations
    
    ### Key Files in `models`
    """)
    
    files_df = pd.DataFrame({
        "File": [
            "interface.py", 
            "openai_chatcompletions.py", 
            "openai_provider.py",
            "openai_responses.py", 
            "_openai_shared.py",  # Fixed: underscore prefix
            "../model_settings.py"
        ],
        "Description": [
            "Defines core interfaces (`Model`, `ModelProvider`, etc.) that all implementations follow",
            "Implements OpenAI's chat completions API interface",
            "Provides factory methods for creating OpenAI model instances",
            "Implements OpenAI's Responses API interface",
            "Shared utilities for working with OpenAI",
            "Defines model configuration parameters like temperature, top_p, etc."
        ],
        "Core Classes": [
            "Model, ModelProvider, ModelTracing",
            "OpenAIChatCompletionsModel",
            "OpenAIProvider", 
            "OpenAIResponsesModel",
            "get_default_openai_client",  # Fixed function name
            "ModelSettings"
        ]
    })
    
    st.dataframe(files_df, use_container_width=True)
    
    # Visualize the relationship between classes
    generate_model_relationship_diagram()

elif section == "Core Interfaces":
    st.header("Core Interfaces")
    
    st.markdown("""
    The core interfaces define the abstractions that all model implementations must follow.
    These are defined in `interface.py` and provide a consistent way to interact with different LLM providers.
    """)
    
    # Create tabs for each interface
    interface_tabs = st.tabs(["ModelTracing", "Model", "ModelProvider"])
    
    with interface_tabs[0]:
        st.markdown("### `ModelTracing` (Enum)")
        st.markdown("""
        This enum defines the tracing capabilities of models, allowing for recording model inputs/outputs for debugging.
        """)
        
        for name, member in ModelTracing.__members__.items():
            st.markdown(f"**`{name}`** ({member.value}): {member.__doc__}")
        
        st.markdown("#### Methods")
        st.markdown("""
        - **`is_disabled()`**: Returns True if tracing is disabled
        - **`include_data()`**: Returns True if trace should include input/output data
        """)
    
    with interface_tabs[1]:
        display_class_details(Model)
    
    with interface_tabs[2]:
        display_class_details(ModelProvider)

elif section == "Model Implementation":
    st.header("Model Implementations")
    
    st.markdown("""
    The SDK provides specific implementations of the core interfaces for different OpenAI API endpoints.
    Let's explore the main implementations and how they integrate with OpenAI.
    """)
    
    impl_tabs = st.tabs(["OpenAIChatCompletionsModel", "OpenAIResponsesModel", "OpenAIProvider"])
    
    with impl_tabs[0]:
        st.markdown("### `OpenAIChatCompletionsModel`")
        st.markdown("""
        This model implementation uses the OpenAI Chat Completions API, which is the API endpoint for models like GPT-3.5 and GPT-4.
        It handles formatting inputs, processing responses, and dealing with streaming.
        """)
        
        # Show key methods
        st.markdown("#### Key Methods")
        methods = [
            ("__init__", "Initialize with model name and OpenAI client"),
            ("get_response", "Synchronously get a complete model response"),
            ("stream_response", "Stream response events asynchronously"),
            ("_fetch_response", "Internal method that makes the actual API call")
        ]
        
        for method, desc in methods:
            st.markdown(f"**`{method}`**: {desc}")
        
        # Data conversion flow
        st.markdown("#### Data Conversion Flow")
        
        st.markdown("""
        ```
        Input (str | list[TResponseInputItem]) 
          ↓ _Converter.items_to_messages
        OpenAI ChatCompletionMessageParam
          ↓ OpenAI API call
        ChatCompletion or ChatCompletionChunk (streaming)
          ↓ _Converter.message_to_output_items
        Output (list[TResponseOutputItem])
        ```
        """)
        
        # Helper classes
        st.markdown("#### Helper Classes")
        st.markdown("""
        - **`_Converter`**: Handles conversions between SDK types and OpenAI API types
        - **`_StreamingState`**: Tracks state during streaming responses
        - **`ToolConverter`**: Converts SDK Tool objects to OpenAI API tool format
        """)
    
    with impl_tabs[1]:
        st.markdown("### `OpenAIResponsesModel`")
        st.markdown("""
        This implementation uses OpenAI's Responses API. The Responses API is a newer API that provides more structured output formats and better support for tool use.
        """)
        
        # Show key methods
        st.markdown("#### Key Methods")
        methods = [
            ("__init__", "Initialize with model name and OpenAI client"),
            ("get_response", "Synchronously get a complete model response"),
            ("stream_response", "Stream response events asynchronously"),
            ("_create_response", "Internal method that makes the actual API call")
        ]
        
        for method, desc in methods:
            st.markdown(f"**`{method}`**: {desc}")
        
        st.markdown("#### Differences from Chat Completions")
        st.markdown("""
        - More native support for structured inputs and outputs
        - Better handling of tool calls and responses
        - Cleaner streaming interface with specific event types
        - Improved support for multi-modal input (text, images, etc.)
        """)
    
    with impl_tabs[2]:
        # Now display OpenAIProvider instead of OpenAIModelProvider
        display_class_details(OpenAIProvider)
        
        st.markdown("#### Model Provider Flow")
        st.markdown("""
        ```
        1. Initialize OpenAIProvider with client (or default)
        2. Call get_model("gpt-4") or other model name
        3. Provider determines appropriate implementation:
           - OpenAIChatCompletionsModel for chat models
           - OpenAIResponsesModel for Responses API models
        4. Returns appropriate model implementation
        ```
        """)

elif section == "Model Settings":
    st.header("Model Settings")
    
    st.markdown("""
    `ModelSettings` is a dataclass that holds all the configuration options for a model call. 
    These settings are used to control the behavior of the model, like temperature, top_p, etc.
    """)
    
    # Display ModelSettings as a table
    settings_df = pd.DataFrame({
        "Parameter": [
            "temperature", 
            "top_p", 
            "frequency_penalty",
            "presence_penalty",
            "tool_choice",
            "parallel_tool_calls",
            "truncation",
            "max_tokens",
            "store"
        ],
        "Type": [
            "float | None",
            "float | None",
            "float | None",
            "float | None",
            "Literal['auto', 'required', 'none'] | str | None",
            "bool | None",
            "Literal['auto', 'disabled'] | None",
            "int | None",
            "bool | None"
        ],
        "Description": [
            "Controls randomness (higher = more random)",
            "Alternative to temperature, samples from top probability tokens",
            "Reduces repetition of tokens based on frequency",
            "Reduces repetition of tokens regardless of frequency",
            "Controls if/how tools are called ('auto', 'required', 'none', or specific tool)",
            "Whether to allow parallel tool calls",
            "How to handle context window limitations",
            "Maximum number of tokens to generate",
            "Whether to store the response for later retrieval"
        ],
        "Default": [
            "None (provider default)",
            "None (provider default)",
            "None (provider default)",
            "None (provider default)",
            "None (typically 'auto')",
            "None (typically False)",
            "None (provider default)",
            "None (provider default)",
            "None (typically True)"
        ]
    })
    
    st.dataframe(settings_df, use_container_width=True)
    
    # Example of updating model settings
    st.markdown("### Creating and Updating ModelSettings")
    st.code("""
    # Create default settings
    default_settings = ModelSettings()
    
    # Create settings with specific values
    creative_settings = ModelSettings(
        temperature=0.9,
        top_p=0.95,
        frequency_penalty=0.2
    )
    
    # Create hybrid settings by overlaying values
    custom_settings = default_settings.resolve(creative_settings)
    
    # Only override specific settings
    focused_settings = ModelSettings(temperature=0.2, top_p=0.8)
    """, language="python")

elif section == "Tool Integration":
    st.header("Tool Integration")
    
    st.markdown("""
    The models module integrates with the Tools system to allow models to use tools (functions) during their execution.
    This is how agents can take actions, access external knowledge, or perform computations.
    """)
    
    st.markdown("### Tool Conversion Flow")
    
    # Diagram showing tool conversion
    tool_flow = """
    graph TD
        A[SDK Tool Object] --> B["ToolConverter.to_openai()"]
        B --> C[OpenAI Tool Format]
        C --> D[Include in API Request]
        D --> E[Model Uses Tool]
        E --> F[Tool Call in Response]
        F --> G["_Converter.message_to_output_items()"]
        G --> H[SDK Response Format]
    """
    
    # Display using both methods
    
    # 1. Standard markdown for Mermaid diagrams
    st.markdown("#### Mermaid via Markdown:")
    st.markdown(f"```mermaid\n{tool_flow}\n```")
    
    # 2. Using the streamlit-mermaid component
    st.markdown("#### Mermaid via Component:")
    stmd.st_mermaid(tool_flow)
    
    st.markdown("### Example Tool Definition")
    
    st.code("""
    from agents.tool import FunctionTool
    
    # Define a simple calculator tool
    calculator_tool = FunctionTool(
        name="calculator",
        description="Perform mathematical calculations",
        params_json_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        },
        func=lambda params: {"result": eval(params["expression"])}
    )
    
    # Use in model call
    response = await model.get_response(
        system_instructions="You are a helpful assistant.",
        input="What is 123 * 456?",
        model_settings=ModelSettings(),
        tools=[calculator_tool],  # Pass the tool here
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED
    )
    """, language="python")
    
    st.markdown("### ToolConverter Implementation")
    
    # Show the tool converter code
    st.code("""
class ToolConverter:
    @classmethod
    def to_openai(cls, tool: Tool) -> ChatCompletionToolParam:
        if isinstance(tool, FunctionTool):
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.params_json_schema,
                },
            }

        raise UserError(
            f"Hosted tools are not supported with the ChatCompletions API."
            f"Got tool type: {type(tool)}, tool: {tool}"
        )

    @classmethod
    def convert_handoff_tool(cls, handoff: Handoff[Any]) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": handoff.tool_name,
                "description": handoff.tool_description,
                "parameters": handoff.input_json_schema,
            },
        }
    """, language="python")

elif section == "Response Handling":
    st.header("Response Handling")
    
    st.markdown("""
    The SDK provides abstractions for handling responses from LLMs, including both complete responses
    and streaming responses. Let's look at response formats and how they're processed.
    """)
    
    # Tabs for different aspects of response handling
    response_tabs = st.tabs(["ModelResponse", "Streaming Events", "Response Conversion"])
    
    with response_tabs[0]:
        st.markdown("### `ModelResponse`")
        
        st.markdown("""
        `ModelResponse` is the container for model outputs. It includes:
        - `output`: A list of output items (text, function calls, etc.)
        - `usage`: Token usage statistics
        - `referenceable_id`: Optional ID for future references
        """)
        
        st.code("""
        # Example ModelResponse structure
        ModelResponse(
            output=[
                ResponseOutputMessage(
                    id="...",
                    content=[
                        ResponseOutputText(text="Hello, how can I help?", type="output_text", annotations=[])
                    ],
                    role="assistant",
                    type="message",
                    status="completed"
                )
            ],
            usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
            referenceable_id=None
        )
        """, language="python")
    
    with response_tabs[1]:
        st.markdown("### Streaming Response Events")
        
        st.markdown("""
        When streaming, the model yields a series of events that describe parts of the response as they become available.
        Event types include:
        """)
        
        streaming_events = pd.DataFrame({
            "Event Type": [
                "ResponseCreatedEvent",
                "ResponseOutputItemAddedEvent",
                "ResponseContentPartAddedEvent",
                "ResponseTextDeltaEvent",
                "ResponseFunctionCallArgumentsDeltaEvent",
                "ResponseContentPartDoneEvent",
                "ResponseOutputItemDoneEvent",
                "ResponseCompletedEvent"
            ],
            "Description": [
                "Initial event when response creation begins",
                "A new output item (message, function call) is added",
                "A new content part within a message is added",
                "New text is added to the current content part",
                "Arguments for a function call are added/updated",
                "A content part has been completed",
                "An output item has been completed",
                "The entire response is complete"
            ]
        })
        
        st.dataframe(streaming_events, use_container_width=True)
        
        st.markdown("### Example Streaming Loop")
        
        st.code("""
        async def stream_example():
            model = provider.get_model("gpt-4")
            stream = model.stream_response(
                system_instructions="You are a helpful assistant.",
                input="Tell me about neural networks",
                model_settings=ModelSettings(),
                tools=[],
                output_schema=None,
                handoffs=[],
                tracing=ModelTracing.DISABLED
            )
            
            text_so_far = ""
            
            async for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    text_so_far += event.delta
                    print(f"New text: {event.delta}")
                elif isinstance(event, ResponseCompletedEvent):
                    print("Response complete!")
                    
            return text_so_far
        """, language="python")
    
    with response_tabs[2]:
        st.markdown("### Response Conversion")
        
        st.markdown("""
        The SDK converts between OpenAI's response format and its own abstractions using the `_Converter` class.
        Key conversions include:
        """)
        
        conversions = pd.DataFrame({
            "From": [
                "ChatCompletionMessage",
                "ResponseOutputMessage",
                "TResponseInputItem",
                "Tool",
                "ResponseOutputText",
                "ResponseFunctionToolCall"
            ],
            "To": [
                "list[TResponseOutputItem]",
                "ChatCompletionAssistantMessageParam",
                "list[ChatCompletionMessageParam]",
                "ChatCompletionToolParam",
                "ChatCompletionContentPartTextParam",
                "ChatCompletionMessageToolCallParam"
            ],
            "Method": [
                "message_to_output_items",
                "items_to_messages",
                "items_to_messages",
                "to_openai",
                "extract_text_content",
                "maybe_function_tool_call"
            ]
        })
        
        st.dataframe(conversions, use_container_width=True)
        
        st.markdown("""
        The conversion process handles:
        - Structured messages with multiple content parts
        - Tool calls and tool results
        - Streaming deltas and events
        - Various input formats (strings, structured objects)
        """)

elif section == "Playground":
    # Use the refactored playground module
    display_playground()
    
elif section == "Advanced Playground":
    # Use the enhanced playground showcase through the wrapper
    run_showcase()