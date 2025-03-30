import streamlit as st
import os

def logging_config_module():
    st.header("Logging Configuration")
    
    # Initialize session state
    if 'logging_level' not in st.session_state:
        st.session_state.logging_level = "warning"
    
    if 'log_model_data' not in st.session_state:
        st.session_state.log_model_data = True
    
    if 'log_tool_data' not in st.session_state:
        st.session_state.log_tool_data = True
    
    if 'enable_file_logging' not in st.session_state:
        st.session_state.enable_file_logging = False
    
    if 'log_file_path' not in st.session_state:
        st.session_state.log_file_path = "logs/agents.log"
    
    if 'log_rotation_size' not in st.session_state:
        st.session_state.log_rotation_size = 1
    
    if 'log_backup_count' not in st.session_state:
        st.session_state.log_backup_count = 5
    
    # Create form
    with st.form(key="logging_config_form"):
        logging_level = st.selectbox(
            "Logging Level",
            options=["warning", "debug", "info", "error", "verbose"],
            index=["warning", "debug", "info", "error", "verbose"].index(st.session_state.logging_level),
            format_func=lambda x: f"{x.title()}" + (" (default)" if x == "warning" else "") + 
                               (" (all logs)" if x == "debug" else "") + 
                               (" (enable verbose stdout logging)" if x == "verbose" else "")
        )
        
        log_model_data = st.checkbox(
            "Log LLM Inputs/Outputs",
            value=st.session_state.log_model_data
        )
        
        log_tool_data = st.checkbox(
            "Log Tool Inputs/Outputs",
            value=st.session_state.log_tool_data
        )
        
        # Log Output Destinations
        st.subheader("Log Output Destinations")
        
        enable_file_logging = st.checkbox(
            "Enable File Logging",
            key="enable_file_logging_checkbox",
            value=st.session_state.enable_file_logging,
            help="Log to a file in addition to console output"
        )
        
        if enable_file_logging:
            col1, col2 = st.columns(2)
            
            with col1:
                log_file_path = st.text_input(
                    "Log File Path",
                    key="log_file_path_input",
                    value=st.session_state.log_file_path,
                    help="Path to the log file"
                )
            
            with col2:
                log_rotation_size = st.number_input(
                    "Rotation Size (MB)",
                    key="log_rotation_size_input",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.log_rotation_size,
                    help="Size in MB before rotating log file"
                )
            
            log_backup_count = st.slider(
                "Backup Count",
                key="log_backup_count_slider",
                min_value=1,
                max_value=20,
                value=st.session_state.log_backup_count,
                help="Number of backup log files to keep"
            )
        else:
            log_file_path = st.session_state.log_file_path
            log_rotation_size = st.session_state.log_rotation_size
            log_backup_count = st.session_state.log_backup_count
        
        # Submit button
        submitted = st.form_submit_button("Save Logging Configuration")
        
        if submitted:
            st.session_state.logging_level = logging_level
            st.session_state.log_model_data = log_model_data
            st.session_state.log_tool_data = log_tool_data
            st.session_state.enable_file_logging = enable_file_logging
            st.session_state.log_file_path = log_file_path
            st.session_state.log_rotation_size = log_rotation_size
            st.session_state.log_backup_count = log_backup_count
            st.success("Logging configuration saved!")
    
    # Show generated code
    st.subheader("Logging Configuration Code")
    
    code_lines = []
    
    # Imports
    imports = []
    if st.session_state.logging_level == "verbose":
        imports.append("from agents import enable_verbose_stdout_logging")
    else:
        imports.append("import logging")
        if st.session_state.enable_file_logging:
            imports.append("from logging.handlers import RotatingFileHandler")
            imports.append("import os")
    
    for imp in imports:
        code_lines.append(imp)
    
    code_lines.append("")  # Empty line after imports
    
    # Logging configuration
    code_lines.append("# Logging Configuration")
    
    if st.session_state.logging_level == "verbose":
        code_lines.append("enable_verbose_stdout_logging()")
    else:
        code_lines.append('logger = logging.getLogger("openai.agents")')
        code_lines.append(f'logger.setLevel(logging.{st.session_state.logging_level.upper()})')
        
        # Console handler
        code_lines.append("# Console logging")
        code_lines.append('console_handler = logging.StreamHandler()')
        code_lines.append(f'console_handler.setLevel(logging.{st.session_state.logging_level.upper()})')
        code_lines.append('formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")')
        code_lines.append('console_handler.setFormatter(formatter)')
        code_lines.append('logger.addHandler(console_handler)')
        
        # File handler (if enabled)
        if st.session_state.enable_file_logging:
            code_lines.append("")
            code_lines.append("# File logging")
            
            # Ensure log directory exists
            log_dir = os.path.dirname(st.session_state.log_file_path)
            if log_dir:
                code_lines.append(f'# Ensure log directory exists')
                code_lines.append(f'os.makedirs("{log_dir}", exist_ok=True)')
            
            # Create rotating file handler
            code_lines.append(f'file_handler = RotatingFileHandler(')
            code_lines.append(f'    "{st.session_state.log_file_path}",')
            code_lines.append(f'    maxBytes={st.session_state.log_rotation_size * 1024 * 1024},  # {st.session_state.log_rotation_size} MB')
            code_lines.append(f'    backupCount={st.session_state.log_backup_count}')
            code_lines.append(f')')
            code_lines.append(f'file_handler.setLevel(logging.{st.session_state.logging_level.upper()})')
            code_lines.append('file_handler.setFormatter(formatter)')
            code_lines.append('logger.addHandler(file_handler)')
    
    code_lines.append("")
    
    # Sensitive data logging configuration
    if not st.session_state.log_model_data or not st.session_state.log_tool_data:
        code_lines.append("# Sensitive Data Logging Configuration")
        code_lines.append("# Add these to your environment before starting your application:")
        if not st.session_state.log_model_data:
            code_lines.append("# export OPENAI_AGENTS_DONT_LOG_MODEL_DATA=1")
        if not st.session_state.log_tool_data:
            code_lines.append("# export OPENAI_AGENTS_DONT_LOG_TOOL_DATA=1")
    
    # If no configuration, add a comment
    if not code_lines or (len(code_lines) == 1 and code_lines[0] == ""):
        code_lines = ["# Default logging configuration (warning level)"]
    
    full_code = "\n".join(code_lines)
    
    st.code(full_code, language="python")
    
    # Return the current configuration
    return {
        "logging_level": st.session_state.logging_level,
        "log_model_data": st.session_state.log_model_data,
        "log_tool_data": st.session_state.log_tool_data,
        "file_logging": {
            "enabled": st.session_state.enable_file_logging,
            "file_path": st.session_state.log_file_path if st.session_state.enable_file_logging else None,
            "rotation_size_mb": st.session_state.log_rotation_size if st.session_state.enable_file_logging else None,
            "backup_count": st.session_state.log_backup_count if st.session_state.enable_file_logging else None
        }
    }

# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Logging Configuration", layout="wide")
    logging_config = logging_config_module()
    
    # Show the current configuration
    with st.expander("Current Configuration"):
        st.json(logging_config)