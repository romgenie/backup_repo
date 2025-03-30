"""
Chat tab component for Streamlit interface.

Note: This file has been refactored into modular components in the chat/ directory.
It now serves as a facade to maintain backward compatibility.
"""

# Import the refactored module
from ui_components.tabs.chat import render_chat_tab

# Export for backward compatibility
__all__ = ["render_chat_tab"]