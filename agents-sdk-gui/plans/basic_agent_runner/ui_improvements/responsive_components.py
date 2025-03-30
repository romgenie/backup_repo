"""
Responsive UI Components for Agent Runner

This module provides responsive versions of common UI components
that adapt to different screen sizes and content amounts.
"""

import streamlit as st
from typing import Callable, Optional, Dict, Any
import json

def responsive_container(content_function: Callable, 
                         min_height: str = "200px", 
                         max_height: str = "600px", 
                         key: Optional[str] = None):
    """
    Create a responsive container that adapts to content and screen size.
    
    Args:
        content_function: Function that renders the container content
        min_height: Minimum container height in CSS units
        max_height: Maximum container height in CSS units
        key: Optional unique key for the container
    """
    container_id = key or f"responsive-container-{id(content_function)}"
    
    # Create container with fluid height
    container_html = f"""
    <div id="{container_id}"
         class="responsive-container" 
         style="min-height: {min_height}; max-height: {max_height}; overflow-y: auto;">
        <div class="responsive-content"></div>
    </div>
    
    <script>
        // Adjust container height based on content and viewport
        function adjustContainerHeight_{container_id.replace('-', '_')}() {{
            const container = document.getElementById('{container_id}');
            if (!container) return;
            
            const content = container.querySelector('.responsive-content');
            if (!content) return;
            
            const viewportHeight = window.innerHeight;
            const contentHeight = content.scrollHeight;
            
            // Set container height based on content, but respect min/max constraints
            let finalHeight = contentHeight;
            const minHeight = parseInt('{min_height}');
            const maxHeight = Math.min(parseInt('{max_height}'), viewportHeight * 0.7);
            
            if (!isNaN(minHeight) && finalHeight < minHeight) finalHeight = minHeight;
            if (!isNaN(maxHeight) && finalHeight > maxHeight) finalHeight = maxHeight;
            
            container.style.height = finalHeight + 'px';
        }}
        
        // Create MutationObserver to watch for content changes
        const observer_{container_id.replace('-', '_')} = new MutationObserver(function(mutations) {{
            adjustContainerHeight_{container_id.replace('-', '_')}();
        }});
        
        // Start observing when the container exists
        function initObserver_{container_id.replace('-', '_')}() {{
            const container = document.getElementById('{container_id}');
            if (container) {{
                const content = container.querySelector('.responsive-content');
                if (content) {{
                    observer_{container_id.replace('-', '_')}.observe(content, {{ 
                        childList: true, 
                        subtree: true,
                        attributes: true
                    }});
                    adjustContainerHeight_{container_id.replace('-', '_')}();
                }}
            }}
        }}
        
        // Initialize when DOM is loaded
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initObserver_{container_id.replace('-', '_')});
        }} else {{
            initObserver_{container_id.replace('-', '_')}();
        }}
        
        // Adjust on window resize
        window.addEventListener('resize', adjustContainerHeight_{container_id.replace('-', '_')});
    </script>
    """
    
    # Create the container
    st.markdown(container_html, unsafe_allow_html=True)
    
    # Use a regular container to render the content
    # The JavaScript will move this content into the responsive container
    with st.container():
        st.markdown(f'<div id="content-for-{container_id}">', unsafe_allow_html=True)
        content_function()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add script to move content
    st.markdown(f"""
    <script>
        function moveContent_{container_id.replace('-', '_')}() {{
            const sourceContainer = document.getElementById('content-for-{container_id}');
            const targetContainer = document.getElementById('{container_id}');
            
            if (sourceContainer && targetContainer) {{
                const targetContent = targetContainer.querySelector('.responsive-content');
                if (targetContent) {{
                    // Move all child nodes
                    while (sourceContainer.firstChild) {{
                        targetContent.appendChild(sourceContainer.firstChild);
                    }}
                    
                    // Hide the source container
                    sourceContainer.style.display = 'none';
                    
                    // Adjust height
                    adjustContainerHeight_{container_id.replace('-', '_')}();
                }}
            }}
        }}
        
        // Execute when DOM is loaded
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', moveContent_{container_id.replace('-', '_')});
        }} else {{
            moveContent_{container_id.replace('-', '_')}();
        }}
    </script>
    """, unsafe_allow_html=True)

def enhanced_chat_container(min_height: str = "400px", max_height: str = "600px"):
    """
    Creates an enhanced chat container with better styling and responsiveness.
    
    Args:
        min_height: Minimum container height
        max_height: Maximum container height
        
    Returns:
        A container object for rendering chat messages
    """
    # Add the enhanced chat container styling
    st.markdown("""
    <style>
    .enhanced-chat-container {
        border: 1px solid var(--neutral-300);
        border-radius: var(--radius-md);
        background-color: white;
        display: flex;
        flex-direction: column;
        padding: 12px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create the container with responsive behavior
    container = st.container()
    
    # Return the container for message rendering
    return container

def adaptive_columns(ratios=None, count=2, gap="10px"):
    """
    Creates responsive columns that adapt to screen size.
    
    Args:
        ratios: List of width ratios for columns (e.g., [2, 1] for 2:1 ratio)
        count: Number of equal columns if ratios not provided
        gap: Gap between columns
        
    Returns:
        List of column containers
    """
    # Calculate column widths
    if ratios:
        total = sum(ratios)
        widths = [f"{(r/total) * 100}%" for r in ratios]
    else:
        # Equal width columns
        width = f"{100 / count}%"
        widths = [width] * count
    
    # Create column style
    column_style = f"""
    <style>
    .adaptive-columns {{
        display: flex;
        gap: {gap};
        width: 100%;
        flex-wrap: wrap;
    }}
    
    .adaptive-column {{
        box-sizing: border-box;
    }}
    
    /* Responsive behavior */
    @media (max-width: 768px) {{
        .adaptive-columns {{
            flex-direction: column;
        }}
        
        .adaptive-column {{
            width: 100% !important;
            margin-bottom: 10px;
        }}
    }}
    </style>
    """
    
    st.markdown(column_style, unsafe_allow_html=True)
    
    # Start the columns container
    columns_html = '<div class="adaptive-columns">'
    for i, width in enumerate(widths):
        columns_html += f'<div id="adaptive-column-{i}" class="adaptive-column" style="width: {width};"></div>'
    columns_html += '</div>'
    
    st.markdown(columns_html, unsafe_allow_html=True)
    
    # Create Streamlit containers for each column
    containers = []
    for i in range(len(widths)):
        container = st.container()
        containers.append(container)
        
        # Add script to move content
        with container:
            st.markdown(f"""
            <script>
                const targetColumn_{i} = document.getElementById('adaptive-column-{i}');
                if (targetColumn_{i}) {{
                    targetColumn_{i}.appendChild(this.element);
                }}
            </script>
            """, unsafe_allow_html=True)
    
    return containers

def responsive_status_bar():
    """
    Creates a responsive status bar that stays visible.
    
    Returns:
        A container for status information
    """
    st.markdown("""
    <style>
    .status-bar {
        position: sticky;
        top: 0;
        background-color: white;
        border-bottom: 1px solid var(--neutral-300);
        padding: 8px 12px;
        z-index: 100;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .status-bar-left {
        display: flex;
        align-items: center;
    }
    
    .status-bar-right {
        display: flex;
        align-items: center;
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        margin-right: 16px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    
    .status-indicator-label {
        font-size: 14px;
        white-space: nowrap;
    }
    
    @media (max-width: 768px) {
        .status-bar {
            flex-direction: column;
            align-items: flex-start;
        }
        
        .status-bar-right {
            margin-top: 8px;
        }
    }
    </style>
    
    <div class="status-bar">
        <div class="status-bar-left">
            <div id="status-left-content"></div>
        </div>
        <div class="status-bar-right">
            <div id="status-right-content"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create containers for the left and right sides
    left_container = st.container()
    right_container = st.container()
    
    return left_container, right_container

def detect_screen_size():
    """
    Detects the client screen size using JavaScript.
    Sets screen_width and screen_height in session state.
    """
    screen_js = """
    <script>
        // Function to send screen dimensions to Streamlit
        function sendScreenDimensions() {
            const width = window.innerWidth;
            const height = window.innerHeight;
            
            window.parent.postMessage({
                type: 'streamlit:setSessionState',
                data: { screen_width: width, screen_height: height }
            }, '*');
        }
        
        // Send on load and resize
        window.addEventListener('load', sendScreenDimensions);
        window.addEventListener('resize', sendScreenDimensions);
    </script>
    """
    
    st.markdown(screen_js, unsafe_allow_html=True)
    
    # Get dimensions from session state with defaults
    width = st.session_state.get('screen_width', 1200)
    height = st.session_state.get('screen_height', 800)
    
    # Determine device category
    if width < 768:
        device = "mobile"
    elif width < 1024:
        device = "tablet"
    else:
        device = "desktop"
    
    return {
        "width": width,
        "height": height,
        "device": device
    }

# Example usage:
"""
def render_chat_content():
    st.write("This is the chat content")
    st.write("It will adapt to the available height")
    for i in range(10):
        st.write(f"Message {i}")

responsive_container(render_chat_content, min_height="200px", max_height="500px")

# Detect screen size and adapt layout
screen = detect_screen_size()
if screen["device"] == "mobile":
    st.write("Mobile layout")
elif screen["device"] == "tablet":
    st.write("Tablet layout")
else:
    st.write("Desktop layout")

# Create responsive columns
col1, col2 = adaptive_columns(ratios=[2, 1])
with col1:
    st.write("Main content")
with col2:
    st.write("Sidebar content")
"""