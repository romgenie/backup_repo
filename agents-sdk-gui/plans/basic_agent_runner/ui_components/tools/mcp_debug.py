"""
MCP debugging utilities and UI components.

Note: This file has been refactored into modular components in the debug/ directory.
It now serves as a facade to maintain backward compatibility.
"""

# Import from refactored modules
from ui_components.tools.debug import render_server_debug_section

# Export for backward compatibility
__all__ = ["render_server_debug_section"]