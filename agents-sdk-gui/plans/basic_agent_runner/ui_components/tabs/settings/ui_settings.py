"""
UI settings components for settings tab.
"""

import streamlit as st
from typing import Dict, Any, Optional

def render_ui_settings() -> None:
    """Render UI settings section"""
    st.markdown("""
    <div class="form-section">
        <h3>UI Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize UI settings if not present
    if "ui_settings" not in st.session_state:
        st.session_state.ui_settings = {
            "theme": "light",
            "chat_bubbles": True,
            "show_timestamps": True,
            "compact_view": False,
            "font_size": "medium",
        }
    
    # UI settings
    ui_settings = st.session_state.ui_settings
    
    # Theme selection
    st.markdown('<div class="form-group">', unsafe_allow_html=True)
    theme_options = ["light", "dark", "auto"]
    theme_index = theme_options.index(ui_settings.get("theme", "light")) if ui_settings.get("theme") in theme_options else 0
    ui_settings["theme"] = st.selectbox(
        "Theme",
        options=theme_options,
        index=theme_index,
        help="Choose application theme (may require refresh)"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat display options
    col1, col2 = st.columns(2)
    
    with col1:
        ui_settings["chat_bubbles"] = st.checkbox(
            "Use chat bubbles",
            value=ui_settings.get("chat_bubbles", True),
            help="Display messages in chat bubble style"
        )
        
        ui_settings["show_timestamps"] = st.checkbox(
            "Show timestamps",
            value=ui_settings.get("show_timestamps", True),
            help="Show timestamp for each message"
        )
    
    with col2:
        ui_settings["compact_view"] = st.checkbox(
            "Compact view",
            value=ui_settings.get("compact_view", False),
            help="Use more compact UI layout"
        )
    
    # Font size selection
    st.markdown('<div class="form-group">', unsafe_allow_html=True)
    font_size_options = ["small", "medium", "large"]
    font_size_index = font_size_options.index(ui_settings.get("font_size", "medium")) if ui_settings.get("font_size") in font_size_options else 1
    ui_settings["font_size"] = st.selectbox(
        "Font Size",
        options=font_size_options,
        index=font_size_index,
        help="Choose text size for the application"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Code block settings
    with st.expander("Code Block Settings", expanded=False):
        if "code_block_settings" not in st.session_state:
            st.session_state.code_block_settings = {
                "show_copy_button": True,
                "line_numbers": True,
                "syntax_highlighting": True,
                "wrap_code": False
            }
        
        code_settings = st.session_state.code_block_settings
        
        # Code block options
        code_col1, code_col2 = st.columns(2)
        
        with code_col1:
            code_settings["show_copy_button"] = st.checkbox(
                "Show copy button",
                value=code_settings.get("show_copy_button", True),
                help="Display copy button for code blocks"
            )
            
            code_settings["line_numbers"] = st.checkbox(
                "Show line numbers",
                value=code_settings.get("line_numbers", True),
                help="Display line numbers in code blocks"
            )
        
        with code_col2:
            code_settings["syntax_highlighting"] = st.checkbox(
                "Syntax highlighting",
                value=code_settings.get("syntax_highlighting", True),
                help="Highlight syntax in code blocks"
            )
            
            code_settings["wrap_code"] = st.checkbox(
                "Wrap code",
                value=code_settings.get("wrap_code", False),
                help="Wrap long lines in code blocks"
            )
    
    # Accessibility settings
    with st.expander("Accessibility", expanded=False):
        if "accessibility_settings" not in st.session_state:
            st.session_state.accessibility_settings = {
                "high_contrast": False,
                "reduced_motion": False,
                "screen_reader_support": False
            }
        
        access_settings = st.session_state.accessibility_settings
        
        access_settings["high_contrast"] = st.checkbox(
            "High contrast mode",
            value=access_settings.get("high_contrast", False),
            help="Increase contrast for better readability"
        )
        
        access_settings["reduced_motion"] = st.checkbox(
            "Reduce motion",
            value=access_settings.get("reduced_motion", False),
            help="Minimize animations and motion effects"
        )
        
        access_settings["screen_reader_support"] = st.checkbox(
            "Screen reader enhancements",
            value=access_settings.get("screen_reader_support", False),
            help="Add additional context for screen readers"
        )
    
    # Apply UI settings button
    if st.button("Apply UI Settings"):
        # Check if we need to change theme
        if ui_settings.get("theme") == "dark":
            st.markdown("""
            <script>
                document.body.classList.add('dark');
            </script>
            """, unsafe_allow_html=True)
        elif ui_settings.get("theme") == "light":
            st.markdown("""
            <script>
                document.body.classList.remove('dark');
            </script>
            """, unsafe_allow_html=True)
        
        # Set font size CSS
        font_size = ui_settings.get("font_size", "medium")
        font_size_values = {"small": "0.85rem", "medium": "1rem", "large": "1.2rem"}
        font_size_css = font_size_values.get(font_size, "1rem")
        
        st.markdown(f"""
        <style>
            .stApp {{ font-size: {font_size_css} !important; }}
        </style>
        """, unsafe_allow_html=True)
        
        st.success("UI settings applied successfully")
        # Note: Some settings may require a page refresh to take full effect