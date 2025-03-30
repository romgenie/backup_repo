"""
Settings tab for the application.

This module has been refactored into smaller components:
- ui_components/tabs/settings/ - Component modules
"""

# Import from the refactored settings module to maintain backward compatibility
from .settings import render_settings_tab

# Make the render_settings_tab function available directly from this module
__all__ = ['render_settings_tab']