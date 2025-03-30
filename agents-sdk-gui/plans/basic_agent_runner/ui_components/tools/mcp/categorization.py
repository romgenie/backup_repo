"""
Tool categorization utilities for MCP tools.
"""

from typing import Dict, Any, List, Optional, Tuple, NamedTuple

# Cache size for tool categorization
MCP_TOOL_CACHE_SIZE = 500

# Define a named tuple for tool categorization results
class ToolCategory(NamedTuple):
    is_mcp: bool
    category: str
    
# Implement a simple LRU cache for tool categorization
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[str, ToolCategory] = {}
        self.usage: List[str] = []
    
    def get(self, key: str) -> Optional[ToolCategory]:
        if key in self.cache:
            # Move key to the end (most recently used)
            self.usage.remove(key)
            self.usage.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: ToolCategory) -> None:
        if key in self.cache:
            # Update existing entry
            self.cache[key] = value
            self.usage.remove(key)
            self.usage.append(key)
        else:
            # Add new entry
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                lru_key = self.usage.pop(0)
                del self.cache[lru_key]
            self.cache[key] = value
            self.usage.append(key)

# Initialize the cache
_tool_category_cache = LRUCache(MCP_TOOL_CACHE_SIZE)

# Common MCP tool patterns for better identification
MCP_TOOL_PATTERNS = {
    # Time tools
    "time": ["getCurrentTime", "getTime", "formatTime", "timeZone"],
    # Filesystem tools
    "filesystem": ["readFile", "writeFile", "listFiles", "createDirectory", "fileExists"],
    # Memory tools
    "memory": ["saveMemory", "loadMemory", "listKeys", "deleteMemory"],
    # Web tools
    "web": ["fetch", "navigate", "screenshot", "getHtml", "fillForm", "click"],
    # GitHub tools
    "github": ["getRepository", "listIssues", "createPullRequest"],
    # Sequential thinking
    "thinking": ["sequentialThink", "stepByStep", "analyze"]
}

def is_mcp_tool(tool_name: str) -> Tuple[bool, str]:
    """
    Determine if a tool is an MCP tool based on its name.
    
    Args:
        tool_name: The name of the tool to check
        
    Returns:
        Tuple of (is_mcp_tool, category_name)
    """
    # Check cache first
    cached_result = _tool_category_cache.get(tool_name)
    if cached_result:
        return cached_result.is_mcp, cached_result.category
    
    # Not in cache, perform the categorization
    result = _categorize_tool(tool_name)
    
    # Store result in cache
    _tool_category_cache.put(tool_name, ToolCategory(result[0], result[1]))
    
    return result

def _categorize_tool(tool_name: str) -> Tuple[bool, str]:
    """
    Internal function to categorize a tool by name.
    
    Args:
        tool_name: The name of the tool to check
        
    Returns:
        Tuple of (is_mcp_tool, category_name)
    """
    # Common MCP tool naming patterns
    if (
        # The tool is namespaced (common in MCP tools)
        "." in tool_name or
        # Tool starts with mcp
        tool_name.startswith("mcp.") or
        # Common MCP meta-tools
        tool_name in ["listTools", "help"]
    ):
        return True, "general"
    
    # Check against known MCP tool patterns
    for category, patterns in MCP_TOOL_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in tool_name.lower():
                return True, category
    
    # Explicit checks for commonly used MCP tools
    common_mcp_tools = {
        "puppeteer": "web",
        "browser": "web",
        "getdate": "time",
        "datetime": "time",
        "calendar": "time",
        "remember": "memory",
        "recall": "memory",
        "search": "web",
        "file": "filesystem",
        "github": "github",
        "repo": "github"
    }
    
    for keyword, category in common_mcp_tools.items():
        if keyword.lower() in tool_name.lower():
            return True, category
    
    # If we can't definitively identify it as an MCP tool
    return False, ""

def get_tool_icon(category: str) -> str:
    """
    Get an icon for a tool category.
    
    Args:
        category: The tool category
        
    Returns:
        A string with an emoji icon
    """
    icons = {
        "time": "🕒",
        "filesystem": "📁",
        "memory": "🧠",
        "web": "🌐",
        "github": "🐙",
        "thinking": "💭",
        "general": "🔧"
    }
    
    return icons.get(category, "🔧")