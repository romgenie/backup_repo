import streamlit as st
import json
import os

# Import the modules
from basic_config_module import basic_config_module
from env_variables_module import env_variables_module
from api_config_module import api_config_module
from model_settings_module import model_settings_module
from models_module import models_module
from streaming_config_module import streaming_config_module
from tracing_config_module import tracing_config_module
from logging_config_module import logging_config_module

def main():
    st.set_page_config(
        page_title="Agent Configuration",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Agent Configuration")
    
    # Initialize session state for storing configurations
    if 'config' not in st.session_state:
        st.session_state.config = {
            "basic": {},
            "env": {},
            "api": {},
            "model_settings": {},
            "models": {},
            "streaming": {},
            "tracing": {},
            "logging": {}
        }
    
    # Create tabs for each configuration module
    tabs = st.tabs([
        "Basic Configuration", 
        "Environment Variables", 
        "API Configuration",
        "Model Settings",
        "Models",
        "Streaming Configuration",
        "Tracing Configuration", 
        "Logging Configuration"
    ])
    
    # Populate each tab with its module
    with tabs[0]:
        st.session_state.config["basic"] = basic_config_module()
    
    with tabs[1]:
        st.session_state.config["env"] = env_variables_module()
    
    with tabs[2]:
        st.session_state.config["api"] = api_config_module()
    
    with tabs[3]:
        st.session_state.config["model_settings"] = model_settings_module()
    
    with tabs[4]:
        st.session_state.config["models"] = models_module()
    
    with tabs[5]:
        st.session_state.config["streaming"] = streaming_config_module()
    
    with tabs[6]:
        st.session_state.config["tracing"] = tracing_config_module()
    
    with tabs[7]:
        st.session_state.config["logging"] = logging_config_module()
    
    # Generate complete configuration code
    st.header("Complete Configuration")
    
    # Generate code from all modules
    generate_complete_code()
    
    # Export configuration
    if st.button("Export Configuration as JSON", key="main_export_config_button"):
        export_configuration()

def generate_complete_code():
    """Generate complete Python code from all configurations"""
    config = st.session_state.config
    
    # Build imports
    imports = ["from agents import Agent"]
    
    # Basic configuration (no imports needed)
    
    # Environment Variables
    if config["env"].get("env_vars", {}).get("OPENAI_API_KEY"):
        imports.append("from agents import set_default_openai_key")
    
    if config["env"].get("use_custom_client"):
        imports.append("from openai import AsyncOpenAI")
        imports.append("from agents import set_default_openai_client")
    
    # API Configuration
    if config["api"].get("openai_api"):
        imports.append("from agents import set_default_openai_api")
    
    # Streaming Configuration
    if "streaming" in config and config["streaming"].get("use_streaming", False):
        imports.append("from agents import Runner")
    else:
        imports.append("from agents import Runner")
    
    # Tracing Configuration
    if config["tracing"].get("tracing_api_key"):
        imports.append("from agents import set_tracing_export_api_key")
    
    if config["tracing"].get("tracing_disabled"):
        imports.append("from agents import set_tracing_disabled")
        
    if config["tracing"].get("trace_metadata"):
        imports.append("from agents import set_trace_attributes")
    
    # Logging Configuration
    if config["logging"].get("logging_level") == "verbose":
        imports.append("from agents import enable_verbose_stdout_logging")
    elif config["logging"].get("logging_level") != "warning" or config["logging"].get("file_logging", {}).get("enabled"):
        imports.append("import logging")
        
        if config["logging"].get("file_logging", {}).get("enabled"):
            imports.append("from logging.handlers import RotatingFileHandler")
            imports.append("import os")
    
    # Remove duplicates and join imports
    imports = list(dict.fromkeys(imports))
    imports_code = "\n".join(imports)
    
    # Build configuration code
    code_sections = []
    
    # API Key setup
    if config["env"].get("env_vars", {}).get("OPENAI_API_KEY"):
        api_key = config["env"]["env_vars"]["OPENAI_API_KEY"]
        code_sections.append(f"# Set OpenAI API Key\nset_default_openai_key(\"{api_key}\")\n")
    
    # Custom client setup
    if config["env"].get("use_custom_client"):
        client_code = "# Configure custom OpenAI client\ncustom_client = AsyncOpenAI("
        
        if config["env"].get("base_url"):
            client_code += f"base_url=\"{config['env']['base_url']}\", "
        
        if config["env"].get("env_vars", {}).get("OPENAI_API_KEY"):
            client_code += f"api_key=\"{config['env']['env_vars']['OPENAI_API_KEY']}\""
        else:
            client_code += "# You need to provide an API key"
        
        client_code += ")\nset_default_openai_client(custom_client)\n"
        code_sections.append(client_code)
    
    # API Type setup
    if config["api"].get("openai_api"):
        code_sections.append(f"# Set OpenAI API type\nset_default_openai_api(\"{config['api']['openai_api']}\")\n")
    
    # Streaming setup
    if "streaming" in config:
        use_streaming = config["streaming"].get("use_streaming", False)
        streaming_code = "# Streaming Configuration\n"
        
        if use_streaming:
            streaming_code += """# Example of running with streaming
async def run_agent_streamed(input_text):
    result = Runner.run_streamed(agent, input=input_text)
    
    # Process streaming events
    async for event in result.stream_events():
        if event.type == "raw_response_event" and hasattr(event.data, "delta"):
            # Display the streamed tokens
            print(event.data.delta, end="", flush=True)
            
    return result.final_output
"""
        else:
            streaming_code += """# Example of running without streaming
def run_agent(input_text):
    # Synchronous method
    result = Runner.run_sync(agent, input=input_text)
    return result.final_output

# Alternative async method without streaming
async def run_agent_async(input_text):
    result = await Runner.run(agent, input=input_text)
    return result.final_output
"""
        code_sections.append(streaming_code)
    
    # Tracing setup
    if config["tracing"].get("tracing_api_key") or config["tracing"].get("tracing_disabled") or config["tracing"].get("trace_metadata"):
        tracing_code = "# Tracing Configuration\n"
        
        if config["tracing"].get("tracing_api_key"):
            tracing_code += f"set_tracing_export_api_key(\"{config['tracing']['tracing_api_key']}\")\n"
        
        if config["tracing"].get("tracing_disabled"):
            tracing_code += "set_tracing_disabled(True)\n"
            
        if config["tracing"].get("trace_metadata"):
            import json
            metadata_str = json.dumps(config["tracing"]["trace_metadata"], indent=2).replace('"', "'")
            tracing_code += f"set_trace_attributes({metadata_str})\n"
        
        code_sections.append(tracing_code)
    
    # Logging setup
    if config["logging"].get("logging_level") != "warning" or config["logging"].get("file_logging", {}).get("enabled"):
        logging_code = "# Logging Configuration\n"
        
        if config["logging"].get("logging_level") == "verbose":
            logging_code += "enable_verbose_stdout_logging()\n"
        else:
            level = config["logging"].get("logging_level", "").upper()
            logging_code += f"logger = logging.getLogger(\"openai.agents\")\n"
            logging_code += f"logger.setLevel(logging.{level})\n"
            
            # Console handler
            logging_code += "# Console logging\n"
            logging_code += f"console_handler = logging.StreamHandler()\n"
            logging_code += f"console_handler.setLevel(logging.{level})\n"
            logging_code += "formatter = logging.Formatter(\"%(asctime)s - %(name)s - %(levelname)s - %(message)s\")\n"
            logging_code += "console_handler.setFormatter(formatter)\n"
            logging_code += "logger.addHandler(console_handler)\n"
            
            # File handler if enabled
            if config["logging"].get("file_logging", {}).get("enabled"):
                file_path = config["logging"]["file_logging"]["file_path"]
                rotation_size = config["logging"]["file_logging"]["rotation_size_mb"]
                backup_count = config["logging"]["file_logging"]["backup_count"]
                
                logging_code += "\n# File logging\n"
                
                # Directory creation if needed
                import os
                log_dir = os.path.dirname(file_path)
                if log_dir:
                    logging_code += "# Ensure log directory exists\n"
                    logging_code += f"os.makedirs(\"{log_dir}\", exist_ok=True)\n\n"
                
                logging_code += f"file_handler = RotatingFileHandler(\n"
                logging_code += f"    \"{file_path}\",\n"
                logging_code += f"    maxBytes={rotation_size * 1024 * 1024},  # {rotation_size} MB\n"
                logging_code += f"    backupCount={backup_count}\n"
                logging_code += f")\n"
                logging_code += f"file_handler.setLevel(logging.{level})\n"
                logging_code += "file_handler.setFormatter(formatter)\n"
                logging_code += "logger.addHandler(file_handler)\n"
        
        code_sections.append(logging_code)
    
    # Sensitive data logging configuration
    log_model = config["logging"].get("log_model_data", True)
    log_tool = config["logging"].get("log_tool_data", True)
    
    if not log_model or not log_tool:
        sensitive_code = "# Sensitive Data Logging Configuration\n"
        sensitive_code += "# Add these to your environment before starting your application:\n"
        
        if not log_model:
            sensitive_code += "# export OPENAI_AGENTS_DONT_LOG_MODEL_DATA=1\n"
        
        if not log_tool:
            sensitive_code += "# export OPENAI_AGENTS_DONT_LOG_TOOL_DATA=1\n"
        
        code_sections.append(sensitive_code)
    
    # Model settings
    model_settings_code = []
    model_settings_dict = {}
    
    # Response format
    response_format_type = config["model_settings"].get("response_format", {}).get("type", "auto")
    if response_format_type != "auto":
        if response_format_type == "json":
            model_settings_code.append("# Response format configuration")
            model_settings_code.append("response_format = {\"type\": \"json_object\"}")
        elif response_format_type == "custom":
            custom_format = config["model_settings"].get("response_format", {}).get("custom")
            if custom_format:
                model_settings_code.append("# Custom response format configuration")
                model_settings_code.append(f"response_format = {json.dumps(custom_format, indent=2)}")
    
    # Reliability settings
    reliability = config["model_settings"].get("reliability", {})
    if reliability:
        reliability_settings = {}
        
        if reliability.get("max_retries", 0) > 0:
            reliability_settings["max_retries"] = reliability["max_retries"]
        
        if reliability.get("timeout", 300.0) != 300.0:
            reliability_settings["timeout"] = reliability["timeout"]
        
        if reliability.get("seed") is not None:
            reliability_settings["seed"] = reliability["seed"]
        
        if reliability.get("parallel_tool_calls") is False:  # Only if disabled
            reliability_settings["parallel_tool_calls"] = False
        
        if reliability_settings:
            model_settings_dict.update(reliability_settings)
            model_settings_code.append("\n# Reliability settings")
            model_settings_code.append("model_settings = {")
            for key, value in reliability_settings.items():
                model_settings_code.append(f"    \"{key}\": {value},")
            model_settings_code.append("}")
    
    # User identifier
    user_id = config["model_settings"].get("user_identifier")
    if user_id:
        model_settings_code.append("\n# User identifier")
        model_settings_code.append(f"user_id = \"{user_id}\"")
    
    if model_settings_code:
        code_sections.append("\n".join(model_settings_code))
    
    # Agent creation
    if config["basic"].get("agent_name"):
        agent_code = "# Create Agent\n"
        agent_code += "agent = Agent(\n"
        agent_code += f"    name=\"{config['basic'].get('agent_name')}\",\n"
        
        if config["basic"].get("instructions"):
            agent_code += f"    instructions=\"{config['basic'].get('instructions')}\",\n"
        
        if config["basic"].get("model"):
            agent_code += f"    model=\"{config['basic'].get('model')}\",\n"
        
        # Add model settings to agent creation
        if response_format_type != "auto":
            agent_code += "    response_format=response_format,\n"
        
        if model_settings_dict:
            for key, value in model_settings_dict.items():
                agent_code += f"    {key}={value},\n"
        
        # Add max_turns from basic config if available
        if config["basic"].get("model_settings", {}).get("max_turns"):
            max_turns = config["basic"]["model_settings"]["max_turns"]
            agent_code += f"    max_turns={max_turns},\n"
        
        if user_id:
            agent_code += f"    user=\"{user_id}\",\n"
        
        # Remove the last comma
        agent_code = agent_code.rstrip(",\n") + "\n"
        agent_code += ")"
        
        # Add a comment about which run method to use based on streaming setting
        if "streaming" in config and config["streaming"].get("use_streaming", False):
            agent_code += "\n\n# Use Runner.run_streamed() to stream responses"
        else:
            agent_code += "\n\n# Use Runner.run_sync() for synchronous execution or Runner.run() for async"
        
        code_sections.append(agent_code)
    
    # Combine all code
    full_code = imports_code + "\n\n" + "\n".join(code_sections)
    
    # Display the full code
    st.code(full_code, language="python")
    
    # Add a download button for the code
    st.download_button(
        label="Download Configuration Code",
        data=full_code,
        file_name="agent_config.py",
        mime="text/plain",
        key="main_download_code_button"
    )

def export_configuration():
    """Export the current configuration as a JSON file"""
    config_json = json.dumps(st.session_state.config, indent=2)
    
    st.download_button(
        label="Download JSON Configuration",
        data=config_json,
        file_name="agent_config.json",
        mime="application/json",
        key="main_download_json_button"
    )
    
    st.success("Configuration exported successfully!")

if __name__ == "__main__":
    main()