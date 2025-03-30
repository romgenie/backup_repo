"""
Setup Wizard Component for Agent Runner

This module provides a guided setup experience for first-time users
that can be integrated into the existing application with minimal changes.
"""

import streamlit as st
from typing import Callable, Dict, Any, Optional
import json

class SetupWizard:
    """
    Setup wizard component that guides users through initial configuration.
    This component is designed to be embedded in the existing UI structure.
    """
    
    def __init__(self, 
                 on_api_key_save: Callable[[str, bool, str], str] = None,
                 on_api_key_load: Callable[[str], tuple[str, bool]] = None,
                 on_agent_create: Callable[[Any, Any], None] = None):
        """
        Initialize the setup wizard with callbacks for integration.
        
        Args:
            on_api_key_save: Callback for saving API key (receives key, save_to_file flag, and file path)
            on_api_key_load: Callback for loading API key from .env file (receives file path)
            on_agent_create: Callback for creating an agent
        """
        self.on_api_key_save = on_api_key_save
        self.on_api_key_load = on_api_key_load
        self.on_agent_create = on_agent_create
        
        # Initialize wizard state if not present
        if "wizard_step" not in st.session_state:
            st.session_state.wizard_step = 1
        if "wizard_completed" not in st.session_state:
            st.session_state.wizard_completed = False
            
    def is_completed(self) -> bool:
        """Check if the wizard has been completed."""
        return st.session_state.get("wizard_completed", False)
    
    def start_wizard(self):
        """Start or reset the wizard."""
        st.session_state.wizard_step = 1
        st.session_state.wizard_completed = False
        
    def render(self) -> None:
        """Render the setup wizard interface."""
        if self.is_completed():
            return
        
        # Get current step
        current_step = st.session_state.wizard_step
        total_steps = 4
        
        # Create container with custom styling
        st.markdown("""
        <style>
        .wizard-container {
            border: 1px solid rgba(65, 105, 225, 0.15);
            border-radius: 8px;
            padding: 24px;
            background-color: rgba(65, 105, 225, 0.03);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 24px 0;
        }
        .wizard-steps-indicator {
            margin-bottom: 16px;
            color: #626E7F;
            font-size: 14px;
        }
        .wizard-steps-indicator .current-step {
            color: #4169E1;
            font-weight: 600;
        }
        .wizard-nav-buttons {
            display: flex;
            justify-content: space-between;
            margin-top: 24px;
        }
        /* Style for progress bar */
        .stProgress > div > div {
            background-color: #4169E1;
        }
        .stProgress {
            background-color: rgba(65, 105, 225, 0.1);
        }
        </style>
        <div class="wizard-container">
        """, unsafe_allow_html=True)
        
        # Progress bar
        progress_percentage = (current_step - 1) / (total_steps - 1)
        st.progress(progress_percentage)
        
        # Step indicator
        st.markdown(f"""
        <div class="wizard-steps-indicator">
            <span class="current-step">Step {current_step}</span> of {total_steps}
        </div>
        """, unsafe_allow_html=True)
        
        # Render the appropriate step
        if current_step == 1:
            self._render_welcome_step()
        elif current_step == 2:
            self._render_api_key_step()
        elif current_step == 3:
            self._render_agent_config_step()
        elif current_step == 4:
            self._render_final_step()
        
        # Navigation buttons
        self._render_navigation_buttons(current_step, total_steps)
        
        # Close container
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _render_welcome_step(self):
        """Render step 1: Welcome and introduction."""
        st.header("Welcome to Agent Runner 👋")
        st.markdown("""
        Let's set up your first AI agent in just a few simple steps:
        
        1. Connect to OpenAI API
        2. Configure your agent
        3. Start chatting!
        
        This guided setup will help you get started quickly.
        """)
        
        # Display sample conversation to set expectations
        with st.expander("Preview Example Conversation", expanded=False):
            st.markdown("""
            **You**: What's the weather in New York?
            
            **Agent**: I'll check the weather for New York.
            
            [Tool Call: get_weather]
            
            Based on the current weather data, it's 72°F and sunny in New York City.
            """)
    
    def _render_api_key_step(self):
        """Render step 2: API key configuration."""
        st.header("Connect to OpenAI API")
        st.markdown("""
        Your agent needs access to OpenAI's API. Don't worry - your API key stays on your computer.
        """)
        
        # First check for existing API key in environment
        import os
        env_api_key = os.environ.get("OPENAI_API_KEY", "")
        env_key_found = bool(env_api_key)
        
        # Create tabs for different methods
        tab1, tab2, tab3 = st.tabs(["Environment", "Load from File", "Manual Entry"])
        
        with tab1:
            # Environment Variable Tab
            st.subheader("Environment Variables")
            
            # Show environment status
            if env_key_found:
                st.success("**API key found in environment variables!**")
                
                # Show a masked version of the key
                masked_key = env_api_key[:7] + "..." + env_api_key[-4:] if len(env_api_key) > 10 else "***"
                st.info(f"Current key: {masked_key}")
                
                # Option to use environment variable
                use_env_var = st.checkbox(
                    "Use this environment variable",
                    value=True,
                    help="Use the OPENAI_API_KEY from your environment",
                    key="use_env_var"
                )
                
                if use_env_var:
                    # Using environment variable - store it for later use
                    st.session_state.temp_api_key = env_api_key
                    st.session_state.using_env_var = True
                    # Important: Mark that we have a valid key source for validation
                    st.session_state.has_key_source = True
            else:
                # Look for key in .env file as a fallback
                from pathlib import Path
                current_dir = os.path.abspath(os.getcwd())
                default_env_path = os.path.join(current_dir, ".env")
                env_file_key = ""
                
                try:
                    env_path = Path(default_env_path)
                    if env_path.exists():
                        with open(env_path, "r") as f:
                            env_content = f.read()
                            import re
                            key_match = re.search(r'OPENAI_API_KEY=([^\n]+)', env_content)
                            if key_match:
                                env_file_key = key_match.group(1).strip()
                                
                                # Show that we found a key in .env
                                st.warning("**API key found in .env file but not loaded in environment**")
                                masked_key = env_file_key[:7] + "..." + env_file_key[-4:] if len(env_file_key) > 10 else "***"
                                st.info(f"Key in .env file: {masked_key}")
                                
                                # Option to use this key
                                use_file_key = st.checkbox(
                                    "Use this key from .env file",
                                    value=True,
                                    help="Use this API key from the .env file",
                                    key="use_file_key"
                                )
                                
                                if use_file_key:
                                    st.session_state.temp_api_key = env_file_key
                                    st.session_state.using_env_var = False
                                    st.session_state.has_key_source = True
                                
                                # Option to load into environment
                                if st.button("Load into environment", key="load_to_env"):
                                    os.environ["OPENAI_API_KEY"] = env_file_key
                                    st.success("✅ Key loaded into environment!")
                                    st.rerun()
                except Exception:
                    # Silently fail on file read errors
                    pass
                
                # If no key found anywhere, show instructions
                if not env_file_key:
                    st.warning("No API key found in environment variables.")
                    st.markdown("""
                    You can set environment variables:
                    ```bash
                    # Linux/Mac
                    export OPENAI_API_KEY=sk-your-key
                    
                    # Windows
                    set OPENAI_API_KEY=sk-your-key
                    ```
                    """)
                    st.session_state.using_env_var = False
                    st.session_state.has_key_source = False
        
        with tab2:
            # Load from File Tab
            st.subheader("Load from .env File")
            
            # File uploader for .env file
            uploaded_file = st.file_uploader("Upload .env file", type=["env", "txt"])
            
            if uploaded_file is not None:
                # Read the file
                env_contents = uploaded_file.getvalue().decode()
                
                # Parse the .env content for OPENAI_API_KEY
                import re
                api_key_match = re.search(r'OPENAI_API_KEY=([^\n]+)', env_contents)
                
                if api_key_match:
                    loaded_key = api_key_match.group(1).strip()
                    # Check if the key is valid format
                    if loaded_key.startswith("sk-"):
                        masked_key = loaded_key[:7] + "..." + loaded_key[-4:] if len(loaded_key) > 10 else "***"
                        st.success(f"Found API key: {masked_key}")
                        
                        # Option to use this key
                        use_loaded_key = st.checkbox(
                            "Use this key",
                            value=True,
                            help="Use the API key from the .env file",
                            key="use_loaded_key"
                        )
                        
                        if use_loaded_key:
                            st.session_state.temp_api_key = loaded_key
                            st.session_state.using_env_var = False
                            st.session_state.has_key_source = True
                    else:
                        st.warning("Found key doesn't match expected format (should start with 'sk-')")
                else:
                    st.error("No OPENAI_API_KEY found in the uploaded file")
            
            # Option to load from default location with file browser
            # Get the application's working directory to display as a hint
            import os
            current_dir = os.path.abspath(os.getcwd())
            
            # First show file browser button (will be improved with actual OS file browser)
            st.markdown("#### Browse for .env file")
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                parent_dir = os.path.dirname(current_dir)
                if st.button("📂 Parent Directory", key="parent_dir"):
                    st.session_state.browse_dir = parent_dir
                    st.rerun()
            
            with col2:
                if st.button("📂 Current Directory", key="current_dir"):
                    st.session_state.browse_dir = current_dir
                    st.rerun()
            
            with col3:
                if st.button("📂 Home", key="home_dir"):
                    st.session_state.browse_dir = os.path.expanduser("~")
                    st.rerun()
            
            # Get current browse directory from session state or use current directory
            browse_dir = st.session_state.get("browse_dir", current_dir)
            
            # Display the currently browsed directory
            st.info(f"Browsing: {browse_dir}")
            
            # Show directories and files
            try:
                # Get directories and files (including hidden ones)
                items = os.listdir(browse_dir)
                dirs = [d for d in items if os.path.isdir(os.path.join(browse_dir, d))]
                files = [f for f in items if os.path.isfile(os.path.join(browse_dir, f))]
                
                # Sort directories and files separately
                dirs.sort()
                files.sort()
                
                # Display directories first
                if dirs:
                    st.markdown("**Directories:**")
                    dir_cols = st.columns(3)
                    for i, directory in enumerate(dirs):
                        col_idx = i % 3
                        with dir_cols[col_idx]:
                            # Create a clickable directory button
                            if st.button(f"📁 {directory}", key=f"dir_{i}"):
                                new_path = os.path.join(browse_dir, directory)
                                st.session_state.browse_dir = new_path
                                st.rerun()
                
                # Display .env files with emphasis
                env_files = [f for f in files if f == '.env' or f.endswith('.env')]
                other_files = [f for f in files if f not in env_files]
                
                if env_files:
                    st.markdown("**Environment Files:**")
                    env_cols = st.columns(3)
                    for i, env_file in enumerate(env_files):
                        col_idx = i % 3
                        with env_cols[col_idx]:
                            file_path = os.path.join(browse_dir, env_file)
                            if st.button(f"✨ {env_file}", key=f"env_{i}"):
                                # Set the selected file path
                                st.session_state.selected_env_file = file_path
                                st.rerun()
                
                # Display a limited number of other files
                if other_files:
                    with st.expander("Other Files"):
                        file_cols = st.columns(3)
                        # Limit to first 30 files to avoid overwhelming the UI
                        for i, file in enumerate(other_files[:30]):
                            col_idx = i % 3
                            with file_cols[col_idx]:
                                file_path = os.path.join(browse_dir, file)
                                if st.button(f"📄 {file}", key=f"file_{i}"):
                                    # Set the selected file path if it might be an env file
                                    st.session_state.selected_env_file = file_path
                                    st.rerun()
                        
                        if len(other_files) > 30:
                            st.caption(f"...and {len(other_files) - 30} more files")
            
            except Exception as e:
                st.error(f"Error browsing directory: {str(e)}")
            
            # Show a separator
            st.markdown("---")
            
            # Manual path input
            st.markdown("#### Manual Path Entry")
            col1, col2 = st.columns([3, 1])
            with col1:
                # Use selected file path if available, otherwise use default
                default_path = st.session_state.get("selected_env_file", os.path.join(current_dir, ".env"))
                
                default_env_path = st.text_input(
                    "Enter path to .env file", 
                    value=default_path,
                    help="Path to your existing .env file (note: .env files are hidden by default)"
                )
                
                # Show a help text about hidden files
                st.caption(f"💡 Current working directory: {current_dir}")
                st.caption("💡 .env files are hidden - on Mac/Linux, use ls -a to see them")
                
                # Check if the specified file exists and show an indicator
                from pathlib import Path
                env_path = Path(default_env_path)
                if env_path.exists():
                    st.success("✓ File exists")
                else:
                    st.error("✗ File not found")
            with col2:
                if st.button("Load", key="load_env_button"):
                    if self.on_api_key_load:
                        try:
                            # Attempt to load through callback
                            loaded_key, success = self.on_api_key_load(default_env_path)
                            if success and loaded_key:
                                st.session_state.temp_api_key = loaded_key
                                st.session_state.has_key_source = True
                                masked_key = loaded_key[:7] + "..." + loaded_key[-4:] if len(loaded_key) > 10 else "***"
                                st.success(f"Loaded API key: {masked_key}")
                            else:
                                st.error(f"Failed to load key from {default_env_path}")
                        except Exception as e:
                            st.error(f"Error loading .env file: {str(e)}")
                    else:
                        # Mock function for demo
                        try:
                            from pathlib import Path
                            env_path = Path(default_env_path)
                            if env_path.exists():
                                with open(env_path, "r") as f:
                                    env_content = f.read()
                                    import re
                                    api_key_match = re.search(r'OPENAI_API_KEY=([^\n]+)', env_content)
                                    if api_key_match:
                                        loaded_key = api_key_match.group(1).strip()
                                        st.session_state.temp_api_key = loaded_key
                                        st.session_state.has_key_source = True
                                        masked_key = loaded_key[:7] + "..." + loaded_key[-4:] if len(loaded_key) > 10 else "***"
                                        st.success(f"Loaded API key: {masked_key}")
                                    else:
                                        st.error("No OPENAI_API_KEY found in the file")
                            else:
                                st.error(f"File not found: {default_env_path}")
                        except Exception as e:
                            st.error(f"Error loading .env file: {str(e)}")
        
        with tab3:
            # Manual Entry Tab
            st.subheader("Manual Entry")
            
            # API key input with security note
            api_key = st.text_input(
                "Enter your OpenAI API key:",
                type="password",
                help="Your API key starts with 'sk-'",
                value=st.session_state.get("temp_api_key", "") if not st.session_state.get("using_env_var", False) else ""
            )
            
            # Save temporarily
            if api_key:
                st.session_state.temp_api_key = api_key
                st.session_state.using_env_var = False
                st.session_state.has_key_source = True
            
            # Visual indicator of valid key format
            if api_key and api_key.startswith("sk-"):
                st.success("API key format looks good! 👍")
            elif api_key:
                st.warning("API keys typically start with 'sk-'")
        
        # File storage options
        st.markdown("---")
        st.subheader("Storage Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            save_to_file = st.checkbox(
                "Save API key to .env file",
                value=st.session_state.get("save_api_key_to_file", True),
                help="Store your API key in a local .env file for future use"
            )
            st.session_state.save_api_key_to_file = save_to_file
            
            if save_to_file:
                # Get the application's working directory for full path
                import os
                current_dir = os.path.abspath(os.getcwd())
                
                env_file_path = st.text_input(
                    ".env file path", 
                    value=os.path.join(current_dir, ".env"),
                    help="Path where to save your .env file"
                )
                st.session_state.env_file_path = env_file_path
                
                # Mention about hidden files and check if file exists
                st.caption("💡 .env will be saved as a hidden file in the specified location")
                
                # Check if file exists and show appropriate message
                from pathlib import Path
                env_path = Path(env_file_path)
                if env_path.exists():
                    with st.expander("File exists - click to see details"):
                        try:
                            # Read file and show masked key if found
                            with open(env_path, "r") as f:
                                content = f.read()
                                import re
                                key_match = re.search(r'OPENAI_API_KEY=([^\n]+)', content)
                                if key_match:
                                    existing_key = key_match.group(1).strip()
                                    masked = existing_key[:7] + "..." + existing_key[-4:] if len(existing_key) > 10 else "***"
                                    st.info(f"This file contains an API key: {masked}")
                                    st.warning("⚠️ The existing file will be overwritten")
                                else:
                                    st.info("File exists, but no API key was found")
                            # Show file stats
                            stats = env_path.stat()
                            import datetime
                            modified_time = datetime.datetime.fromtimestamp(stats.st_mtime)
                            st.caption(f"Last modified: {modified_time}")
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
                
                st.markdown("""
                <div style="background-color:rgba(65, 105, 225, 0.05);padding:10px;border-radius:5px;border:1px solid rgba(65, 105, 225, 0.15);">
                    <p style="margin:0;"><strong>📁 Local Storage</strong>: Your API key will be saved in the specified <code>.env</code> file.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            export_env = st.checkbox(
                "Export as downloadable .env file",
                value=st.session_state.get("export_env_file", False),
                help="Generate a downloadable .env file with your API key"
            )
            st.session_state.export_env_file = export_env
            
            if export_env and st.session_state.get("temp_api_key"):
                # Generate a downloadable .env file
                env_content = f"OPENAI_API_KEY={st.session_state.temp_api_key}\n"
                
                # Create a download button
                st.download_button(
                    label="⬇️ Download .env file",
                    data=env_content,
                    file_name=".env",
                    mime="text/plain",
                    help="Download a .env file with your API key"
                )
        
        # Help links and information
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("[Need an API key? Get one here →](https://platform.openai.com/account/api-keys)")
        with col2:
            st.markdown("[Learn about .env files →](https://www.twilio.com/en-us/blog/working-with-environment-variables-in-python)")
    
    def _render_agent_config_step(self):
        """Render step 3: Agent configuration."""
        st.header("Configure Your Agent")
        
        # Agent name field
        agent_name = st.text_input(
            "Name your agent:",
            value=st.session_state.get("temp_agent_name", "My Assistant"),
            help="Give your agent a name to identify it"
        )
        if agent_name:
            st.session_state.temp_agent_name = agent_name
        
        # Model selection
        model_options = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        default_index = 0 if "temp_model" not in st.session_state else model_options.index(st.session_state.temp_model)
        model = st.selectbox(
            "Select model:",
            options=model_options,
            index=default_index,
            help="Choose the AI model to power your agent"
        )
        st.session_state.temp_model = model
        
        # Tools selection with simple explanation
        st.subheader("Enable Tools")
        st.markdown("""
        Tools give your agent special capabilities. We recommend starting with these:
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            time_tool = st.checkbox(
                "Time Tool (get current time)",
                value=st.session_state.get("temp_time_tool", True),
                help="Let your agent know the current time"
            )
            st.session_state.temp_time_tool = time_tool
            
            weather_tool = st.checkbox(
                "Weather Tool (check weather)",
                value=st.session_state.get("temp_weather_tool", True),
                help="Let your agent check weather conditions"
            )
            st.session_state.temp_weather_tool = weather_tool
        
        with col2:
            news_tool = st.checkbox(
                "News Tool (search news)",
                value=st.session_state.get("temp_news_tool", True),
                help="Let your agent find recent news articles"
            )
            st.session_state.temp_news_tool = news_tool
        
        # Advanced options (collapsed)
        with st.expander("Advanced Options", expanded=False):
            system_message = st.text_area(
                "System message:",
                value=st.session_state.get("temp_system_message", "You are a helpful AI assistant."),
                help="Instructions that define how your agent behaves"
            )
            st.session_state.temp_system_message = system_message
            
            max_turns = st.slider(
                "Maximum conversation turns:",
                min_value=5,
                max_value=20,
                value=st.session_state.get("temp_max_turns", 10),
                help="Limit how many back-and-forth messages in a conversation"
            )
            st.session_state.temp_max_turns = max_turns
    
    def _render_final_step(self):
        """Render step 4: Final step with summary."""
        st.header("Ready to Go! 🚀")
        st.markdown("""
        Your agent is configured and ready to use. Here's a summary of your setup:
        """)
        
        # Display the summary
        summary_data = {
            "Agent Name": st.session_state.get("temp_agent_name", "My Assistant"),
            "Model": st.session_state.get("temp_model", "gpt-4"),
            "Tools Enabled": [],
            "Max Conversation Turns": st.session_state.get("temp_max_turns", 10)
        }
        
        # Add enabled tools to the summary
        if st.session_state.get("temp_time_tool", True):
            summary_data["Tools Enabled"].append("Time Tool")
        if st.session_state.get("temp_weather_tool", True):
            summary_data["Tools Enabled"].append("Weather Tool")
        if st.session_state.get("temp_news_tool", True):
            summary_data["Tools Enabled"].append("News Tool")
        
        # Display as a table
        for key, value in summary_data.items():
            if key == "Tools Enabled":
                if value:
                    st.write(f"**{key}:** {', '.join(value)}")
                else:
                    st.write(f"**{key}:** None")
            else:
                st.write(f"**{key}:** {value}")
        
        # Success message
        st.success("Your agent is configured! Click 'Finish' to start chatting.")
        
        # Information about future changes
        st.info("You can always change these settings later in the Configure tab.")
    
    def _render_navigation_buttons(self, current_step, total_steps):
        """Render navigation buttons for the wizard."""
        st.markdown('<div class="wizard-nav-buttons">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if current_step > 1:
                # Add a unique key for the previous button based on step
                if st.button("⬅️ Previous", key=f"prev_button_step_{current_step}"):
                    st.session_state.wizard_step = current_step - 1
                    st.rerun()
        
        with col2:
            if current_step < total_steps:
                # Add a unique key for the next button based on step
                next_button = st.button("Next ➡️", key=f"next_button_step_{current_step}")
                if next_button:
                    # Validate current step
                    valid = self._validate_current_step(current_step)
                    if valid:
                        st.session_state.wizard_step = current_step + 1
                        st.rerun()
            else:
                # Add a unique key for the finish button
                finish_button = st.button("Finish and Start Chatting 🚀", key="finish_button")
                if finish_button:
                    # Create agent using the configuration
                    self._create_agent_from_wizard_config()
                    st.session_state.wizard_completed = True
                    st.session_state.switch_to_chat_tab = True
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _validate_current_step(self, step):
        """Validate the current wizard step before proceeding."""
        if step == 1:
            # Welcome step - no validation needed
            return True
        elif step == 2:
            # API key validation
            api_key = st.session_state.get("temp_api_key", "")
            has_key_source = st.session_state.get("has_key_source", False)
            
            # Check if we have a key either from input or environment
            if not api_key and not has_key_source:
                # Only show error if no key and no valid source selected
                st.error("Please enter an API key or use one from your environment variables")
                return False
            elif api_key and not api_key.startswith("sk-"):
                # Show warning about unusual format but allow proceeding
                st.warning("Your API key format is unusual, but you can continue")
            
            # Get storage options
            save_to_file = st.session_state.get("save_api_key_to_file", True)
            env_file_path = st.session_state.get("env_file_path", "./.env")
            
            # Save the API key (to env var always, to file if selected)
            if self.on_api_key_save:
                # Pass the save_to_file flag and path for the callback to handle
                self.on_api_key_save(api_key, save_to_file, env_file_path)
            else:
                # If no callback, at least set the environment variable
                import os
                os.environ["OPENAI_API_KEY"] = api_key
                
                # Mock saving to file if selected (for demo)
                if save_to_file:
                    from pathlib import Path
                    env_path = Path(env_file_path)
                    try:
                        # Create directory if it doesn't exist
                        env_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Write to .env file
                        with open(env_path, "w") as f:
                            f.write(f"OPENAI_API_KEY={api_key}\n")
                        st.toast(f"Saved API key to {env_file_path}")
                    except Exception as e:
                        st.error(f"Error saving to {env_file_path}: {str(e)}")
                
            return True
        elif step == 3:
            # Agent configuration validation
            agent_name = st.session_state.get("temp_agent_name", "")
            if not agent_name:
                st.error("Please give your agent a name")
                return False
            return True
        
        return True
    
    def _create_agent_from_wizard_config(self):
        """Create the agent using the wizard configuration."""
        if not self.on_agent_create:
            st.warning("Agent creation callback not provided")
            return
        
        try:
            # Collect agent configuration
            agent_config = {
                "name": st.session_state.get("temp_agent_name", "My Assistant"),
                "model": st.session_state.get("temp_model", "gpt-4"),
                "tools": [],
                "system_message": st.session_state.get("temp_system_message", 
                                                      "You are a helpful AI assistant.")
            }
            
            # Add enabled tools
            if st.session_state.get("temp_time_tool", True):
                agent_config["tools"].append("time")
            if st.session_state.get("temp_weather_tool", True):
                agent_config["tools"].append("weather")
            if st.session_state.get("temp_news_tool", True):
                agent_config["tools"].append("news")
            
            # Create run configuration
            run_config = {
                "max_turns": st.session_state.get("temp_max_turns", 10)
            }
            
            # Call the agent creation callback
            self.on_agent_create(agent_config, run_config)
            
            # Mark the wizard as completed
            st.session_state.wizard_completed = True
            
        except Exception as e:
            st.error(f"Error creating agent: {str(e)}")
            return False
        
        return True


# Example usage:
# def render_wizard_in_chat_tab():
#     # Only show wizard if no agent is configured
#     if "agent" not in st.session_state:
#         wizard = SetupWizard(
#             on_api_key_save=lambda key: on_api_key_save(app, key),
#             on_agent_create=lambda agent_config, run_config: on_agent_create(app, agent_config, run_config)
#         )
#         wizard.render()
#         
#         # If wizard is still showing, return early
#         if not wizard.is_completed():
#             return False
#     
#     # Continue with regular chat tab rendering
#     return True