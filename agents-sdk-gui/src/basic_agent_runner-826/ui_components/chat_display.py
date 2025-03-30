import streamlit as st
import json
from typing import List, Dict, Any

def render_message_content(content: str):
    """
    Renders message content with proper handling for code blocks and tool calls
    
    Args:
        content: The message content to render
    """
    # Check for tool calls in the format <tool>TOOL_NAME</tool>
    if "<tool>" in content and "</tool>" in content:
        # Extract normal content before processing tool calls
        parts = []
        current_text = ""
        i = 0
        
        while i < len(content):
            # Look for tool call start
            tool_start = content.find("<tool>", i)
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
            
            # Find end of tool name
            tool_name_end = content.find("</tool>", tool_start)
            if tool_name_end == -1:
                # Malformed - treat rest as text
                current_text += content[tool_start:]
                if current_text.strip():
                    parts.append({"type": "text", "content": current_text.strip()})
                break
                
            # Extract tool name
            tool_name = content[tool_start + 7:tool_name_end]
            
            # Find args
            args_start = content.find("<args>", tool_name_end)
            if args_start == -1:
                # Malformed - move past tool tag and continue
                i = tool_name_end + 8
                continue
                
            args_end = content.find("</args>", args_start)
            if args_end == -1:
                # Malformed - move past tool tag and continue
                i = args_start + 6
                continue
                
            # Extract args
            args_text = content[args_start + 6:args_end]
            
            # Add tool call to parts
            parts.append({
                "type": "tool_call", 
                "tool_name": tool_name, 
                "args": args_text
            })
            
            # Continue after args end
            i = args_end + 7
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
                
                # Create a styled tool call display with potential result
                tool_call_html = f"""
                <div style="background-color: #f0f8ff; border-radius: 5px; padding: 10px; margin-bottom: 10px; border-left: 3px solid #4169e1;">
                    <div style="font-size: 0.8em; color: #555;">Tool Call:</div>
                    <div style="font-weight: bold; margin-bottom: 5px;">{tool_name}</div>
                    <div style="font-family: monospace; background-color: #f5f5f5; padding: 5px; border-radius: 3px; white-space: pre-wrap;">{args}</div>
                """
                
                # Check if there's a tool result in the next content section
                next_idx = parts.index(part) + 1 if part in parts else -1
                if next_idx < len(parts) and next_idx >= 0:
                    next_part = parts[next_idx]
                    # Look for patterns that might indicate a tool result
                    if next_part.get("type") == "text" and next_part.get("content", "").strip().startswith("Result:"):
                        result_text = next_part.get("content").strip()
                        tool_call_html += f"""
                        <div style="margin-top: 5px; font-size: 0.8em; color: #555;">Tool Result:</div>
                        <div style="font-family: monospace; background-color: #e8f5e8; padding: 5px; border-radius: 3px; white-space: pre-wrap;">{result_text[7:].strip()}</div>
                        """
                        # Mark this part as processed so we don't display it again
                        parts[next_idx]["processed"] = True
                
                # Close the tool call div
                tool_call_html += "</div>"
                
                st.markdown(tool_call_html, unsafe_allow_html=True)
                
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
            st.info("No messages yet. Send a message to start the conversation.")
        else:
            # Display each message in the conversation history
            for message in conversation_history:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "user":
                    # Use Streamlit's native chat_message with "user" avatar
                    with st.chat_message("user", avatar="👤"):
                        st.write(content)
                    # Increment message counter
                    message_count += 1
                        
                elif role == "assistant":
                    # Use Streamlit's native chat_message with "assistant" avatar
                    with st.chat_message("assistant", avatar="🤖"):
                        render_message_content(content)
                    # Increment message counter
                    message_count += 1
            
            # Show info message if no messages were displayed despite having entries in the history
            if message_count == 0 and len(conversation_history) > 0:
                st.warning("There are messages in history but none could be displayed. This may be a format issue.")
    
    return message_count