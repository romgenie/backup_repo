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
                # Format tool calls with more detail and better styling
                tool_call_html = """
                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-top: 10px; background-color: #f9f9f9;">
                    <h4 style="margin-top: 0;">Tool Calls</h4>
                    <ul style="list-style-type: none; padding-left: 0;">
                """
                
                for t in regular_tool_calls:
                    tool_call_html += f"""
                    <li style="margin-bottom: 10px; padding: 8px; background-color: #f0f0f0; border-radius: 4px;">
                        <strong>{t['name']}</strong> <span style="color: #777;">({t['time']})</span><br>
                        <pre style="margin: 5px 0; white-space: pre-wrap; word-break: break-all; background-color: #fff; padding: 5px; border-radius: 3px; font-size: 0.9em;">{t['args']}</pre>
                    </li>
                    """
                    
                tool_call_html += """
                    </ul>
                </div>
                """
                
                container.markdown(tool_call_html, unsafe_allow_html=True)
            
            # Display MCP tool calls with special formatting
            if mcp_tool_calls:
                st.subheader("MCP Tool Calls")
                for t in mcp_tool_calls:
                    with st.expander(f"{t['name']} ({t['time']})", expanded=False):
                        # Server info
                        if 'server' in t:
                            st.write(f"**Server:** {t['server']}")
                        
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
                                if isinstance(t['result'], str):
                                    st.json(json.loads(t['result']))
                                else:
                                    st.json(t['result'])
                            except:
                                # Fallback to displaying as text
                                st.markdown(str(t['result']))
        else:
            container.markdown("""
            <div style="text-align: center; color: #777; padding: 20px; background-color: #f9f9f9; border-radius: 5px;">
                <i>No tool calls made during this conversation</i>
            </div>
            """, unsafe_allow_html=True)