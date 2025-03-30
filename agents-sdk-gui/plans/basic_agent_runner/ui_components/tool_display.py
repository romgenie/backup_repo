import streamlit as st
import json
from typing import List, Dict, Any

def display_tool_calls(container, tool_calls: List[Dict[str, Any]]):
    """
    Displays tool calls in a Streamlit container with nice formatting
    
    Args:
        container: Streamlit container or element to display in
        tool_calls: List of tool call dictionaries with name, time, and args
    """
    with container:
        if tool_calls:
            # Separate MCP tool calls from regular tool calls
            mcp_tool_calls = [t for t in tool_calls if t.get('is_mcp', False)]
            regular_tool_calls = [t for t in tool_calls if not t.get('is_mcp', False)]
            
            # Display regular tool calls
            if regular_tool_calls:
                # Format tool calls with more detail and better styling using our CSS classes
                tool_call_html_markup = """
                <div class="tool-item p-md mt-md">
                    <h4 class="mt-0 mb-sm">Tool Calls</h4>
                    <ul class="p-0" style="list-style-type: none;">
                """
                
                for t in regular_tool_calls:
                    tool_call_html_markup += f"""
                    <li class="tool-call mb-sm">
                        <div class="tool-name">{t['name']}</div>
                        <div class="text-sm text-neutral-500">({t['time']})</div>
                        <div class="tool-result mt-xs">
                            <pre class="m-0">{t['args']}</pre>
                        </div>
                    </li>
                    """
                    
                tool_call_html_markup += """
                    </ul>
                </div>
                """
                
                container.markdown(tool_call_html_markup, unsafe_allow_html=True)
            
            # Display MCP tool calls with special formatting
            if mcp_tool_calls:
                st.subheader("MCP Tool Calls")
                for t in mcp_tool_calls:
                    # Determine expansion state
                    # Show expanded if there's an error or if this is the latest tool call
                    is_latest = mcp_tool_calls.index(t) == len(mcp_tool_calls) - 1
                    has_error = 'error' in t and t['error']
                    auto_expand = is_latest or has_error
                    
                    # Set status icon based on result
                    if 'status' in t:
                        if t['status'] == 'success':
                            status_icon = "✅"
                        elif t['status'] == 'error':
                            status_icon = "❌"
                        else:
                            status_icon = "⏳"
                    else:
                        status_icon = "🔍" if 'result' in t else "⏳"
                    
                    # Create expander with status
                    with st.expander(f"{status_icon} {t['name']} ({t['time']})", expanded=auto_expand):
                        # Server info
                        if 'server' in t:
                            st.write(f"**Server:** {t['server']}")
                            
                        # Add tool info section for common tools with known parameters
                        tool_name = t['name']
                        if tool_name == "get_current_time":
                            st.markdown("""
                            **Tool Info:** 
                            - Gets current time in specified timezone
                            - Parameters:
                              - `timezone`: (optional) Timezone name (e.g., 'America/New_York')
                            """)
                        elif tool_name == 'convert_time':
                            st.markdown("""
                            **Tool Info:**
                            - Converts time between timezones
                            - Parameters:
                              - `time`: Time to convert (ISO format or natural language)
                              - `from_timezone`: Source timezone
                              - `to_timezone`: Target timezone
                            """)
                        
                        # Arguments
                        st.write("**Arguments:**")
                        try:
                            # Try to display as JSON if it's a string
                            if isinstance(t['args'], str):
                                st.json(json.loads(t['args']))
                            else:
                                st.json(t['args'])
                        except:
                            # Fallback to displaying as text
                            st.code(str(t['args']))
                        
                        # Result if available
                        if 'result' in t:
                            st.write("**Result:**")
                            try:
                                # Try to display as JSON
                                if isinstance(t['result'], str) and t['result'].strip().startswith('{'):
                                    st.json(json.loads(t['result']))
                                else:
                                    # For time tool results, try to extract and format nicely
                                    if tool_name == "get_current_time" and isinstance(t['result'], dict):
                                        formatted_time = t['result'].get('formatted', '')
                                        iso_time = t['result'].get('time', '')
                                        timezone = t['result'].get('timezone', '')
                                        
                                        st.markdown(f"""
                                        **Formatted:** {formatted_time}  
                                        **ISO Time:** {iso_time}  
                                        **Timezone:** {timezone}
                                        """)
                                    else:
                                        st.json(t['result'])
                            except Exception as e:
                                # Fallback to displaying as text with error info
                                st.markdown(f"```\n{str(t['result'])}\n```")
                                st.caption(f"Note: Could not parse as JSON ({str(e)})")
                        
                        # Error if available
                        if 'error' in t and t['error']:
                            st.error(f"**Error:** {t['error']}")
                            
                        # Timing information if available
                        if 'timing' in t:
                            st.caption(f"Execution time: {t['timing']} ms")
        else:
            container.markdown("""
            <div class="neutral-message text-center p-md">
                <i>No tool calls made during this conversation</i>
            </div>
            """, unsafe_allow_html=True)