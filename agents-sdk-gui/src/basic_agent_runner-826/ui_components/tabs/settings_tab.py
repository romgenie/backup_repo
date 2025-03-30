import streamlit as st
from typing import Optional, Callable
import os
import base64

def render_settings_tab(
    api_key: str,
    env_file_path: str,
    on_api_key_save: Callable[[str], str],
    on_api_key_load: Callable[[str], tuple]
) -> None:
    """
    Render the settings tab of the application
    
    Args:
        api_key: Current API key
        env_file_path: Path to the .env file
        on_api_key_save: Callback to save API key
        on_api_key_load: Callback to load API key
    """
    st.header("Settings")
    
    # Placeholder for now - will be implemented in future
    st.info("This tab will contain application settings. Coming soon!")
    
    # API Key configuration (will be enhanced in the future)
    st.subheader("API Key Configuration")
    
    api_key_input = st.text_input("OpenAI API Key", value=api_key, type="password")
    
    col1, col2, col3 = st.columns(3)
    
    if col1.button("Save to .env"):
        if api_key_input:
            saved_path = on_api_key_save(api_key_input)
            st.success(f"API key saved to {saved_path}")
        else:
            st.warning("Please enter an API key to save")
    
    if col2.button("Load from .env"):
        loaded_api_key, success = on_api_key_load(env_file_path)
        if success:
            st.success("API key loaded from .env file")
            # Force a rerun to update the text input with the new API key
            st.rerun()
        else:
            st.warning("No .env file found")
    
    # Add download button in the third column
    download_clicked = col3.button("Download .env")
    
    # Handle download button click
    if download_clicked:
        if os.path.exists(env_file_path):
            try:
                with open(env_file_path, 'rb') as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode()
                
                # Create a download link
                download_link = f'<a href="data:application/octet-stream;base64,{b64}" download=".env" target="_blank">Click here to download</a> if the download doesn\'t start automatically.'
                
                # Create an HTML auto-downloading link
                auto_download_html = f"""
                <script>
                    // Create a link element
                    const a = document.createElement('a');
                    // Set link destination to the base64 encoded file
                    a.href = "data:application/octet-stream;base64,{b64}";
                    // Set the file name
                    a.download = ".env";
                    // Append to the body
                    document.body.appendChild(a);
                    // Trigger the download
                    a.click();
                    // Clean up
                    document.body.removeChild(a);
                </script>
                """
                
                # Show success message
                st.success("Preparing .env for download...")
                
                # Display size info and fallback link
                file_size = os.path.getsize(env_file_path)
                file_size_kb = file_size / 1024
                st.caption(f"File size: {file_size_kb:.2f} KB | Path: {env_file_path}")
                
                # Display the HTML with both auto-download script and fallback link
                st.markdown(auto_download_html + download_link, unsafe_allow_html=True)
                
                # Add a security note
                st.info("⚠️ The .env file contains sensitive information. Keep it secure!")
                
            except Exception as e:
                st.error(f"Error preparing download: {str(e)}")
        else:
            st.warning("No .env file found. Save your API key first to create one.")