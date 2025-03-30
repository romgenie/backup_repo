import streamlit as st
import json
import logging
from typing import List, Dict, Any
from ui_components.constants import (
    TOOL_START_TAG, TOOL_END_TAG, ARGS_START_TAG, ARGS_END_TAG,
    TIME_CATEGORY, FILESYSTEM_CATEGORY, MEMORY_CATEGORY, WEB_CATEGORY,
    GITHUB_CATEGORY, THINKING_CATEGORY, GENERAL_CATEGORY
)

# Setup logging
logger = logging.getLogger(__name__)

def render_message_content(content: str, is_user_message: bool = False):
    """
    Renders message content with proper handling for code blocks and tool calls
    
    Args:
        content: The message content to render
        is_user_message: Whether this message is from the user (never parse tool tags)
    """
    # For user messages, never process tool tags - just display the content directly
    if is_user_message:
        # For user content, just handle code blocks if present
        if "```" in content:
            # Split content by code blocks
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Regular text
                    if part.strip():
                        st.write(part.strip())
                else:  # Code block
                    # Check if language is specified
                    code_lines = part.split("\n", 1)
                    if len(code_lines) > 1:
                        lang, code = code_lines[0].strip(), code_lines[1]
                        st.code(code, language=lang if lang else None)
                    else:
                        st.code(part)
        else:
            # Regular text without code blocks
            st.write(content)
        return
            
    # For assistant messages, process tool tags normally
    # Check for tool calls in various formats:
    # 1. <tool>TOOL_NAME</tool>
    # 2. TOOL_START_TAGTOOL_NAMETOOL_END_TAG
    # 3. Special handling for malformed tags like "ART_TAG"
    # 4. Additional malformed patterns like "TOOL_ST" and partial tags
    if ("<tool>" in content and "</tool>" in content) or \
       (TOOL_START_TAG in content and TOOL_END_TAG in content) or \
       ("ART_TAG" in content and "TART_TAG" in content and "D_TAG" in content) or \
       ("TOOL_ST" in content and "TOOL_END_TAG" in content) or \
       ("functions." in content and "ARGS_EN" in content):
        # Extract normal content before processing tool calls
        parts = []
        current_text = ""
        i = 0
        
        while i < len(content):
            # Look for tool call start (all formats)
            tool_start_old = content.find("<tool>", i)
            tool_start_new = content.find(TOOL_START_TAG, i)
            tool_start_malformed1 = content.find("ART_TAG", i)
            tool_start_malformed2 = content.find("TOOL_ST", i)
            tool_start_functions = content.find("functions.", i)
            
            # Find the earliest occurrence of a tool start tag
            positions = {
                "old": tool_start_old,
                "new": tool_start_new,
                "malformed1": tool_start_malformed1,
                "malformed2": tool_start_malformed2,
                "functions": tool_start_functions
            }
            
            # Filter out -1 (not found) positions
            valid_positions = {k: v for k, v in positions.items() if v != -1}
            
            # Debug logging to help diagnose tool tag parsing issues
            logger.debug(f"Tag positions: {positions}")
            
            # Determine which format is being used, or -1 if none is found
            if valid_positions:
                # Find the format with the earliest position
                earliest_format = min(valid_positions.items(), key=lambda x: x[1])
                format_type = earliest_format[0]
                tool_start = earliest_format[1]
                logger.debug(f"Using {format_type} format tags at position {tool_start}")
            else:
                tool_start = -1
                format_type = None
            
            if tool_start == -1:
                # No more tool calls, add remaining text
                current_text += content[i:]
                if current_text.strip():
                    parts.append({"type": "text", "content": current_text.strip()})
                break
                
            # Add text before the tool call
            current_text += content[i:tool_start]
            if current_text.strip():
                parts.append({"type": "text", "content": current_text.strip()})
                current_text = ""
            
            # Find end of tool name based on format
            if format_type == "old":
                tool_name_end = content.find("</tool>", tool_start)
                tag_length = 6  # Length of "<tool>"
                end_tag_length = 7  # Length of "</tool>"
            elif format_type == "new":
                tool_name_end = content.find(TOOL_END_TAG, tool_start)
                tag_length = len(TOOL_START_TAG)
                end_tag_length = len(TOOL_END_TAG)
            elif format_type == "malformed1":
                # For malformed "ART_TAG", locate the end based on the newline or "TART_TAG"
                next_newline = content.find("\n", tool_start)
                tart_tag_pos = content.find("TART_TAG", tool_start)
                
                # Use the position that comes first, or if none found, use end of string
                if next_newline != -1 and (tart_tag_pos == -1 or next_newline < tart_tag_pos):
                    tool_name_end = next_newline
                elif tart_tag_pos != -1:
                    tool_name_end = tart_tag_pos
                else:
                    tool_name_end = len(content)
                    
                tag_length = 7  # Length of "ART_TAG"
                end_tag_length = 0  # No explicit end tag
            elif format_type == "malformed2":
                # For "TOOL_ST" format
                tool_name_end = content.find("TOOL_END_TAG", tool_start)
                tag_length = 7  # Length of "TOOL_ST"
                end_tag_length = 12  # Length of "TOOL_END_TAG"
            elif format_type == "functions":
                # For "functions.xxx" format, find the next space or newline
                next_space = content.find(" ", tool_start)
                next_newline = content.find("\n", tool_start)
                
                if next_space != -1 and (next_newline == -1 or next_space < next_newline):
                    tool_name_end = next_space
                elif next_newline != -1:
                    tool_name_end = next_newline
                else:
                    tool_name_end = len(content)
                
                tag_length = 0  # No prefix to remove
                end_tag_length = 0  # No explicit end tag
                
            if tool_name_end == -1:
                # Malformed - treat rest as text
                current_text += content[tool_start:]
                if current_text.strip():
                    parts.append({"type": "text", "content": current_text.strip()})
                break
                
            # Extract tool name
            tool_name = content[tool_start + tag_length:tool_name_end]
            
            # Clean up tool name for different formats
            if format_type == "functions":
                # For "functions.xxx" format, keep only the part after "functions."
                if tool_name.startswith("functions."):
                    tool_name = tool_name[10:]  # Length of "functions."
            elif format_type == "malformed2":
                # For TOOL_ST format, clean up any trailing whitespace
                tool_name = tool_name.strip()
                
            logger.debug(f"Extracted tool name: '{tool_name}' using format: {format_type}")
            
            # Find args based on format
            if format_type == "old":
                args_start = content.find("<args>", tool_name_end)
                args_tag_length = 6  # Length of "<args>"
                args_end_tag = "</args>"
                args_end_tag_length = 7  # Length of "</args>"
            elif format_type == "new":
                args_start = content.find(ARGS_START_TAG, tool_name_end)
                args_tag_length = len(ARGS_START_TAG)
                args_end_tag = ARGS_END_TAG
                args_end_tag_length = len(ARGS_END_TAG)
            elif format_type == "malformed1":
                # For malformed format, the args might be after "TART_TAG"
                args_start = content.find("TART_TAG", tool_name_end)
                args_tag_length = 8  # Length of "TART_TAG"
                args_end_tag = "D_TAG"
                args_end_tag_length = 5  # Length of "D_TAG"
            elif format_type == "malformed2":
                # For "TOOL_ST" format, look for MCP or empty braces
                mcp_marker = content.find("🔧 MCP", tool_name_end)
                if mcp_marker != -1:
                    args_start = mcp_marker
                    # Find the next "{" after "🔧 MCP"
                    open_brace = content.find("{", mcp_marker)
                    if open_brace != -1:
                        args_start = open_brace
                else:
                    # Look for any JSON object in the text
                    args_start = content.find("{", tool_name_end)
                
                if args_start == -1:
                    args_start = tool_name_end  # Default to end of tool name
                
                args_tag_length = 0  # No explicit args tag
                args_end_tag = "ARGS_EN"
                args_end_tag_length = 7  # Length of "ARGS_EN"
            elif format_type == "functions":
                # For "functions.xxx" format, look for a JSON object after the function name
                open_brace = content.find("{", tool_name_end)
                if open_brace != -1:
                    args_start = open_brace
                    args_tag_length = 0  # No explicit args tag
                    args_end_tag = "}"
                    args_end_tag_length = 1  # Length of "}"
                    
                    # Count braces to handle nested objects
                    depth = 1
                    for i in range(open_brace + 1, len(content)):
                        if content[i] == "{":
                            depth += 1
                        elif content[i] == "}" and depth > 0:
                            depth -= 1
                            if depth == 0:
                                args_end_tag = content[i:i+1]  # Just the closing brace
                                break
                else:
                    # No arguments - use empty dict
                    args_start = tool_name_end
                    args_tag_length = 0
                    args_end_tag = "ARGS_EN"
                    args_end_tag_length = 7
                
            if args_start == -1:
                # Malformed - move past tool tag and continue
                i = tool_name_end + end_tag_length
                continue
                
            args_end = content.find(args_end_tag, args_start)
            if args_end == -1:
                # Try with common fallbacks if the expected end tag isn't found
                common_end_markers = ["\n", "ARGS_EN", "D_TAG", "}", "Tool Result:"]
                for marker in common_end_markers:
                    args_end = content.find(marker, args_start)
                    if args_end != -1:
                        args_end_tag = marker
                        args_end_tag_length = len(marker)
                        logger.debug(f"Used fallback end marker: {marker}")
                        break
                
                # If still not found, use a safe default
                if args_end == -1:
                    # Malformed - use a default empty args or move past tag
                    if format_type in ["malformed2", "functions"]:
                        # For these formats, assume empty args
                        args_end = args_start
                        args_end_tag = ""
                        args_end_tag_length = 0
                        logger.debug("Using empty args as fallback")
                    else:
                        # For other formats, skip and continue
                        i = args_start + args_tag_length
                        logger.debug(f"Skipping malformed tool call, moving to position {i}")
                        continue
                
            # Extract args
            args_text = content[args_start + args_tag_length:args_end]
            
            # Clean up args text for different formats
            args_text = args_text.strip()
            if args_text == "" or args_text == "{}" or args_text == "{}ARGS_EN":
                args_text = "{}"  # Ensure empty args are always formatted as empty JSON object
                
            logger.debug(f"Extracted args: '{args_text[:50]}...' using format: {format_type}")
            
            # Add tool call to parts
            parts.append({
                "type": "tool_call", 
                "tool_name": tool_name, 
                "args": args_text
            })
            
            # Continue after args end
            i = args_end + args_end_tag_length
            current_text = ""
            
        # Render each part accordingly
        for i, part in enumerate(parts):
            # Skip parts that have been marked as processed
            if part.get("processed", False):
                continue
                
            if part["type"] == "text":
                # Handle code blocks in regular text
                text_content = part["content"]
                if "```" in text_content:
                    # Split by code blocks
                    code_parts = text_content.split("```")
                    for j, code_part in enumerate(code_parts):
                        if j % 2 == 0:  # Regular text
                            if code_part.strip():
                                st.write(code_part.strip())
                        else:  # Code block
                            # Check if language is specified
                            code_lines = code_part.split("\n", 1)
                            if len(code_lines) > 1:
                                lang, code = code_lines[0].strip(), code_lines[1]
                                st.code(code, language=lang if lang else None)
                            else:
                                st.code(code_part)
                else:
                    # Regular text
                    st.write(text_content)
                    
            elif part["type"] == "tool_call":
                # Display tool call in a formatted box
                tool_name = part["tool_name"]
                args = part["args"]
                
                try:
                    # Try to format the args as JSON
                    args_json = json.loads(args)
                    args_formatted = json.dumps(args_json, indent=2)
                except:
                    # If not valid JSON, use as is
                    args_formatted = args
                
                # Import the tool identification function
                from ui_components.tools.mcp import is_mcp_tool, get_tool_icon
                
                # Determine if this is an MCP tool and get its category
                is_mcp, category = is_mcp_tool(tool_name)
                
                # Set styling based on tool type
                if is_mcp:
                    # Get the category icon
                    icon = get_tool_icon(category)
                    
                    # Category-specific styling colors
                    category_colors = {
                        TIME_CATEGORY: ("#e6f7ff", "#0077cc"),
                        FILESYSTEM_CATEGORY: ("#e6f9e6", "#008800"),
                        MEMORY_CATEGORY: ("#fff8e6", "#cc7700"),
                        WEB_CATEGORY: ("#eef6ff", "#4169e1"),
                        GITHUB_CATEGORY: ("#f0e6ff", "#6b46c1"),
                        THINKING_CATEGORY: ("#fff0f7", "#cc0066"),
                        GENERAL_CATEGORY: ("#eef6ff", "#4169e1")
                    }
                    
                    # Get the colors for this category or use defaults
                    bg_color, border_color = category_colors.get(category, ("#eef6ff", "#4169e1"))
                    
                    # MCP tool styling with category badge
                    render_tool_badge = f'<span style="background-color: {border_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7em; margin-left: 5px;">{icon} MCP</span>'
                else:
                    # Standard tool styling (purple)
                    render_tool_badge = ""
                    bg_color = "#f5f0ff"
                    border_color = "#6a4dba"
                
                try:
                    # Try to format the args as JSON for better display
                    args_json = json.loads(args)
                    
                    # Extract args for display
                    args_formatted = ""
                    for key, value in args_json.items():
                        if isinstance(value, str):
                            if len(value) > 50:  # Truncate long strings
                                value_display = f'"{value[:50]}..."'
                            else:
                                value_display = f'"{value}"'
                        else:
                            value_display = json.dumps(value)
                        
                        args_formatted += f'<div style="margin: 2px 0;"><span style="color: #444;">{key}</span>: {value_display}</div>'
                        
                    # If no args, show empty object
                    if not args_formatted:
                        args_formatted = "{}"
                        
                except Exception:
                    # If not valid JSON, use as is
                    args_formatted = f'<pre style="margin: 0;">{args}</pre>'
                
                # Create a styled tool call display with potential result
                tool_call_html_markup = f"""
                <div class="tool-call">
                    <div class="text-sm text-neutral-600">Tool Call:</div>
                    <div class="font-semibold mb-sm flex items-center">
                        {tool_name}{render_tool_badge}
                    </div>
                    <div class="tool-result">
                        {args_formatted}
                    </div>
                """
                
                # Check if there's a tool result in the next content section
                next_idx = parts.index(part) + 1 if part in parts else -1
                if next_idx < len(parts) and next_idx >= 0:
                    next_part = parts[next_idx]
                    # Look for patterns that might indicate a tool result
                    if next_part.get("type") == "text" and next_part.get("content", "").strip().startswith("Result:"):
                        result_text = next_part.get("content").strip()
                        result_content = result_text[7:].strip()
                        
                        # Try to format the result as JSON for better display
                        try:
                            json_result = json.loads(result_content)
                            formatted_result = json.dumps(json_result, indent=2)
                        except:
                            formatted_result = result_content
                        
                        tool_call_html_markup += f"""
                        <div class="mt-sm pt-sm" style="border-top: 1px solid var(--neutral-300);">
                            <div class="text-sm text-neutral-600">Tool Result:</div>
                            <div class="tool-result">{formatted_result}</div>
                        </div>
                        """
                        # Mark this part as processed so we don't display it again
                        parts[next_idx]["processed"] = True
                
                # Close the tool call div
                tool_call_html_markup += "</div>"
                
                st.markdown(tool_call_html_markup, unsafe_allow_html=True)
                
                # Add space after tool call
                st.write("")
    # Handle code blocks if present
    elif "```" in content:
        # Split content by code blocks
        parts = content.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text
                if part.strip():
                    st.write(part.strip())
            else:  # Code block
                # Check if language is specified
                code_lines = part.split("\n", 1)
                if len(code_lines) > 1:
                    lang, code = code_lines[0].strip(), code_lines[1]
                    st.code(code, language=lang if lang else None)
                else:
                    st.code(part)
    else:
        # Regular text without code blocks
        st.write(content)

def display_chat(container, conversation_history: List[Dict[str, Any]]) -> int:
    """
    Displays a chat conversation in a Streamlit container
    
    Args:
        container: The Streamlit container to display in
        conversation_history: List of message dictionaries with role and content
        
    Returns:
        int: The number of messages displayed
    """
    # Counter to track if any messages were displayed
    message_count = 0
    
    with container:
        # Check if we have any messages to display
        if len(conversation_history) == 0:
            st.markdown("""
            <div class="info-message text-center">
                No messages yet. Send a message to start the conversation.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display each message in the conversation history
            for message in conversation_history:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "user":
                    # Use Streamlit's native chat_message with "user" avatar and our CSS class
                    with st.chat_message("user", avatar="👤"):
                        # Apply our styling through a container
                        st.markdown('<div class="user-message-container">', unsafe_allow_html=True)
                        # Render user content with special flag to never parse tool tags
                        render_message_content(content, is_user_message=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    # Increment message counter
                    message_count += 1
                        
                elif role == "assistant":
                    # Use Streamlit's native chat_message with "assistant" avatar and our CSS class
                    with st.chat_message("assistant", avatar="🤖"):
                        # Apply our styling through a container
                        st.markdown('<div class="assistant-message-container">', unsafe_allow_html=True)
                        render_message_content(content, is_user_message=False)
                        st.markdown('</div>', unsafe_allow_html=True)
                    # Increment message counter
                    message_count += 1
            
            # Show info message if no messages were displayed despite having entries in the history
            if message_count == 0 and len(conversation_history) > 0:
                st.markdown("""
                <div class="warning-message">
                    There are messages in history but none could be displayed. This may be a format issue.
                </div>
                """, unsafe_allow_html=True)
    
    return message_count