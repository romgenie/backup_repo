"""
About section components for settings tab.
"""

import streamlit as st
import platform
import sys
import os
from typing import Dict, Any, Optional, Tuple

def get_system_info() -> Dict[str, str]:
    """
    Get system information
    
    Returns:
        Dictionary with system info
    """
    info = {}
    info["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    info["platform"] = platform.platform()
    info["processor"] = platform.processor()
    
    # Add streamlit version if available
    try:
        import streamlit
        info["streamlit_version"] = streamlit.__version__
    except (ImportError, AttributeError):
        info["streamlit_version"] = "Unknown"
    
    # Add openai version if available
    try:
        import openai
        info["openai_version"] = openai.__version__
    except (ImportError, AttributeError):
        info["openai_version"] = "Unknown"
    
    # Add agents version if available
    try:
        import agents
        info["agents_version"] = getattr(agents, "__version__", "Unknown")
    except ImportError:
        info["agents_version"] = "Unknown"
    
    return info

def get_app_info() -> Dict[str, str]:
    """
    Get application information
    
    Returns:
        Dictionary with application info
    """
    info = {}
    info["app_name"] = "Agent Runner"
    info["version"] = "0.1.0"
    info["build_date"] = "2023-03-28"
    
    # Get the working directory
    info["working_dir"] = os.getcwd()
    
    return info

def render_about_section() -> None:
    """Render about section"""
    st.markdown("""
    <div class="form-section">
        <h3>About Agent Runner</h3>
    </div>
    """, unsafe_allow_html=True)
    
    app_info = get_app_info()
    system_info = get_system_info()
    
    # Application information
    st.markdown(f"""
    <div class="about-section">
        <p><strong>{app_info['app_name']}</strong> version {app_info['version']}</p>
        <p>A Streamlit-based interface for the OpenAI Agents SDK</p>
        <p>Built: {app_info['build_date']}</p>
        <p>Working directory: {app_info['working_dir']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System information
    with st.expander("System Information", expanded=False):
        st.markdown("""
        <div class="form-section">
            <h4>System Information</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Python Environment**")
            st.markdown(f"- Python: {system_info['python_version']}")
            st.markdown(f"- Streamlit: {system_info['streamlit_version']}")
            st.markdown(f"- OpenAI: {system_info['openai_version']}")
            st.markdown(f"- Agents SDK: {system_info['agents_version']}")
        
        with col2:
            st.markdown("**System**")
            st.markdown(f"- Platform: {system_info['platform']}")
            st.markdown(f"- Processor: {system_info['processor']}")
    
    # Documentation section
    with st.expander("Documentation", expanded=False):
        st.markdown("""
        <div class="form-section">
            <h4>Documentation</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Getting Started
        
        Agent Runner provides a user-friendly interface for creating, configuring, and running agents
        with a variety of tools, including MCP servers.
        
        ### Key Features
        
        - Configure and run OpenAI-compatible agents
        - Connect to MCP servers for additional tools
        - Manage conversation history
        - Customize UI appearance
        
        ### External Resources
        
        - [OpenAI API Documentation](https://platform.openai.com/docs)
        - [Agents SDK Documentation](https://github.com/openai/openai-python)
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [MCP Documentation](https://modelcontextprotocol.ai/docs)
        """)
    
    # Credits section
    with st.expander("Credits", expanded=False):
        st.markdown("""
        <div class="form-section">
            <h4>Credits</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        This application uses the following open source projects:
        
        - [Streamlit](https://streamlit.io/) - For the web interface
        - [OpenAI Python SDK](https://github.com/openai/openai-python) - For API integration
        - [Agents SDK](https://github.com/openai/agents) - For agent functionality
        - [Model Context Protocol](https://modelcontextprotocol.ai) - For tool integration
        
        Special thanks to all contributors and the open source community.
        """)
    
    # License information
    with st.expander("License", expanded=False):
        st.markdown("""
        <div class="form-section">
            <h4>License</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        MIT License

        Copyright (c) 2023 Agent Runner Contributors

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """)