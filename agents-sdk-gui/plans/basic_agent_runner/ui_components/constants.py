"""
Constants used throughout the UI components.
"""

# HTML-like tags used for tool calls
TOOL_START_TAG = "<tool>"
TOOL_END_TAG = "</tool>"
ARGS_START_TAG = "<args>"
ARGS_END_TAG = "</args>"

# Tool categories with their icons
TIME_CATEGORY = "time"
TIME_ICON = "🕒"

FILESYSTEM_CATEGORY = "filesystem"
FILESYSTEM_ICON = "📁"

MEMORY_CATEGORY = "memory"
MEMORY_ICON = "🧠"

WEB_CATEGORY = "web"
WEB_ICON = "🌐"

GITHUB_CATEGORY = "github"
GITHUB_ICON = "🐙"

THINKING_CATEGORY = "thinking"
THINKING_ICON = "💭"

GENERAL_CATEGORY = "general"
GENERAL_ICON = "🔧"

# UI component strings
FORM_FIELD_NAME = "name"
FORM_FIELD_TYPE = "type" 
FORM_FIELD_DESCRIPTION = "description"
FORM_FIELD_REQUIRED = "required"

# Category to icon mapping
CATEGORY_ICONS = {
    TIME_CATEGORY: TIME_ICON,
    FILESYSTEM_CATEGORY: FILESYSTEM_ICON,
    MEMORY_CATEGORY: MEMORY_ICON,
    WEB_CATEGORY: WEB_ICON,
    GITHUB_CATEGORY: GITHUB_ICON,
    THINKING_CATEGORY: THINKING_ICON,
    GENERAL_CATEGORY: GENERAL_ICON,
}