# Detailed Agent Runner UI Improvements

This document provides a comprehensive implementation plan addressing the key usability observations and focusing on creating a guided setup wizard for first-time users.

## 1. First-Time Experience Enhancement

### Setup Wizard Implementation

The setup wizard will guide new users through necessary configuration steps before they can chat with an agent. The wizard will replace the initial empty chat tab with a guided, linear flow.

#### Technical Implementation

```python
def render_setup_wizard():
    """Render the setup wizard for first-time users."""
    # Get current wizard step (default to 1)
    current_step = st.session_state.get("wizard_step", 1)
    total_steps = 4
    
    # Render progress bar
    progress_percentage = (current_step - 1) / total_steps
    st.progress(progress_percentage)
    
    # Step indicator
    st.markdown(f"""
    <div class="wizard-steps-indicator">
        <span class="current-step">Step {current_step}</span> of {total_steps}
    </div>
    """, unsafe_allow_html=True)
    
    # Render appropriate step content
    if current_step == 1:
        _render_wizard_welcome()
    elif current_step == 2:
        _render_wizard_api_key()
    elif current_step == 3:
        _render_wizard_agent_config()
    elif current_step == 4:
        _render_wizard_final_step()
        
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if current_step > 1:
            if st.button("⬅️ Previous"):
                st.session_state.wizard_step = current_step - 1
                st.rerun()
    
    with col2:
        if current_step < total_steps:
            if st.button("Next ➡️", type="primary"):
                # Validate current step before proceeding
                if _validate_current_step(current_step):
                    st.session_state.wizard_step = current_step + 1
                    st.rerun()
        else:
            if st.button("Finish and Start Chatting 🚀", type="primary"):
                st.session_state.wizard_completed = True
                st.session_state.pop("wizard_step", None)
                st.rerun()
```

#### Wizard Step Components

```python
def _render_wizard_welcome():
    """Step 1: Welcome screen with introduction."""
    st.header("Welcome to Agent Runner 👋")
    st.markdown("""
    Let's set up your first AI agent in just a few simple steps:
    
    1. Connect to OpenAI API
    2. Configure your agent
    3. Start chatting!
    
    This guided setup will help you get started quickly.
    """)
    
    # Display sample conversation to set expectations
    with st.expander("Preview Example Conversation", expanded=False):
        st.markdown("""
        **You**: What's the weather in New York?
        
        **Agent**: I'll check the weather for New York.
        
        [Tool Call: get_weather]
        
        Based on the current weather data, it's 72°F and sunny in New York City.
        """)
    
def _render_wizard_api_key():
    """Step 2: API key configuration."""
    st.header("Connect to OpenAI API")
    st.markdown("""
    Your agent needs access to OpenAI's API. Don't worry - your API key stays on your computer.
    """)
    
    # API key input with security note
    api_key = st.text_input(
        "Enter your OpenAI API key:",
        type="password",
        help="Your API key starts with 'sk-'",
        value=st.session_state.get("temp_api_key", "")
    )
    
    # Save temporarily
    if api_key:
        st.session_state.temp_api_key = api_key
    
    # Visual indicator of valid key format
    if api_key and api_key.startswith("sk-"):
        st.success("API key format looks good! 👍")
    elif api_key:
        st.warning("API keys typically start with 'sk-'")
    
    # Help link
    st.markdown("[Need an API key? Learn how to get one →](https://platform.openai.com/account/api-keys)")
```

#### Step Validation Logic

```python
def _validate_current_step(step):
    """Validate the current wizard step before proceeding."""
    if step == 1:
        # Welcome step - no validation needed
        return True
    elif step == 2:
        # API key validation
        api_key = st.session_state.get("temp_api_key", "")
        if not api_key:
            st.error("Please enter an API key to continue")
            return False
        elif not api_key.startswith("sk-"):
            # Allow proceeding but with warning
            st.warning("Your API key format is unusual, but you can continue")
        
        # Actually save the API key to the application
        on_api_key_save(api_key)
        return True
    elif step == 3:
        # Agent configuration validation
        agent_name = st.session_state.get("temp_agent_name", "")
        if not agent_name:
            st.error("Please give your agent a name")
            return False
        
        # Create the agent using the configuration
        _create_agent_from_wizard_config()
        return True
    
    return True
```

### Auto-Detection of Configuration Status

The app will automatically detect if critical configuration is missing and prompt the user to complete setup:

```python
def check_configuration_status():
    """Check if the application is properly configured and guide the user if not."""
    issues = []
    
    # Check for API key
    if not st.session_state.get("api_key"):
        issues.append("API key is missing")
    
    # Check for agent configuration
    if not SessionManager.has_agent():
        issues.append("No agent is configured")
    
    # If issues found, show guided setup card
    if issues:
        st.warning(f"Configuration needed: {', '.join(issues)}")
        if st.button("Run Setup Wizard"):
            st.session_state.wizard_step = 1
            st.rerun()
        return False
    
    return True
```

## 2. Improved Navigation System

### Context-Aware Navigation Bar

A dynamic navigation system that adjusts based on setup status and current context:

```python
def render_smart_navigation():
    """Render a context-aware navigation system rather than static tabs."""
    # Get current state
    has_agent = SessionManager.has_agent()
    has_api_key = bool(st.session_state.get("api_key"))
    current_view = st.session_state.get("current_view", "chat")
    
    # Define navigation items with conditional availability
    nav_items = [
        {"id": "chat", "label": "Chat", "icon": "💬", "enabled": has_agent},
        {"id": "configure", "label": "Configure", "icon": "⚙️", "enabled": True},
        {"id": "tools", "label": "Tools", "icon": "🧰", "enabled": has_agent},
        {"id": "history", "label": "History", "icon": "📚", "enabled": has_agent},
        {"id": "settings", "label": "Settings", "icon": "🔧", "enabled": True},
    ]
    
    # Render navigation with visual distinction and indicators
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    # Create columns for each nav item
    cols = st.columns(len(nav_items))
    
    for i, item in enumerate(nav_items):
        with cols[i]:
            button_class = "nav-button active" if current_view == item["id"] else "nav-button"
            if not item["enabled"]:
                button_class += " disabled"
                
            # Add indicator for required setup steps
            indicator = ""
            if item["id"] == "configure" and not has_api_key:
                indicator = "🔴"  # Red dot for required action
            elif item["id"] == "configure" and not has_agent:
                indicator = "🔶"  # Orange dot for recommended action
                
            html = f"""
            <div class="{button_class}" onclick="handleNavClick('{item['id']}')">
                <div class="nav-icon">{item['icon']}</div>
                <div class="nav-label">{item['label']} {indicator}</div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
    
    # JavaScript for handling clicks
    st.markdown("""
    <script>
    function handleNavClick(viewId) {
        window.parent.postMessage({
            type: 'streamlit:setSessionState',
            data: { current_view: viewId }
        }, '*');
    }
    </script>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Return current view for content rendering
    return current_view
```

### Visual Path Indicators

Add breadcrumb-style path indicators to show users where they are in the workflow:

```python
def render_breadcrumbs(path_segments):
    """Render breadcrumb navigation to show current path."""
    html = '<div class="breadcrumbs">'
    
    for i, segment in enumerate(path_segments):
        if i > 0:
            html += '<span class="breadcrumb-separator">›</span>'
        
        if i == len(path_segments) - 1:
            # Last item is current location
            html += f'<span class="breadcrumb-current">{segment}</span>'
        else:
            # Previous items are clickable
            html += f'<span class="breadcrumb-item" onclick="navigateTo(\'{i}\')">{segment}</span>'
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
```

## 3. Enhanced Visual Hierarchy

### Design System Implementation

Create a consistent component system with clear visual hierarchy:

```python
def primary_button(label, key=None, help=None, on_click=None):
    """Render a primary action button with consistent styling."""
    button_html = f"""
    <button 
        class="primary-button" 
        id="{key if key else label.lower().replace(' ', '_')}"
        title="{help if help else ''}"
    >
        {label}
    </button>
    """
    
    # Use a container to add custom HTML
    container = st.container()
    container.markdown(button_html, unsafe_allow_html=True)
    
    # Use empty space with the same key for the actual button functionality
    clicked = st.button(label, key=key, help=help, on_click=on_click, type="primary")
    return clicked

def secondary_button(label, key=None, help=None, on_click=None):
    """Render a secondary action button with consistent styling."""
    # Similar implementation to primary_button but with secondary styling
    clicked = st.button(label, key=key, help=help, on_click=on_click, type="secondary")
    return clicked

def danger_button(label, key=None, help=None, on_click=None):
    """Render a dangerous action button with warning styling."""
    # Implement with danger styling and confirmation dialog
    clicked = st.button(label, key=key, help=help, on_click=on_click, type="secondary")
    
    # Add confirmation dialog if clicked
    if clicked:
        st.warning(f"Are you sure you want to {label.lower()}?")
        col1, col2 = st.columns(2)
        with col1:
            cancel = st.button("Cancel", key=f"{key}_cancel" if key else f"{label}_cancel")
        with col2:
            confirm = st.button("Confirm", key=f"{key}_confirm" if key else f"{label}_confirm")
        
        return confirm and not cancel
    
    return False
```

### Visual Card Components

```python
def render_action_card(title, description, action_label, icon="➡️", key=None, on_click=None, importance="medium"):
    """Render a card with an action button."""
    # Determine card styling based on importance
    card_class = f"action-card {importance}-importance"
    
    html = f"""
    <div class="{card_class}">
        <div class="card-content">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        <div class="card-action">
            <span class="card-icon">{icon}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    
    # Place the actual button at the bottom of the card
    return st.button(action_label, key=key, on_click=on_click)
```

### Status Indicators

```python
def status_indicator(status, label=None):
    """Render a colored status indicator."""
    colors = {
        "success": "#4CAF50",  # Green
        "warning": "#FF9800",  # Orange
        "error": "#F44336",    # Red
        "info": "#2196F3",     # Blue
        "neutral": "#9E9E9E"   # Gray
    }
    
    color = colors.get(status, colors["neutral"])
    
    html = f"""
    <div class="status-indicator" style="display: flex; align-items: center;">
        <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {color}; margin-right: 6px;"></div>
        {f'<span>{label}</span>' if label else ''}
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)
```

## 4. Enhanced Feedback Mechanisms

### Toast Notifications System

```python
def show_toast(message, type="info", duration=3):
    """Show a temporary toast notification."""
    # Generate a unique ID for this toast
    import random
    toast_id = f"toast_{random.randint(10000, 99999)}"
    
    # Map type to color
    type_colors = {
        "success": "#4CAF50",
        "error": "#F44336",
        "warning": "#FF9800",
        "info": "#2196F3"
    }
    color = type_colors.get(type, "#2196F3")
    
    # Create toast HTML
    toast_html = f"""
    <div id="{toast_id}" class="toast {type}-toast" style="display: none;">
        <div class="toast-content">
            <div class="toast-icon">{{"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}}["{type}"]</div>
            <div class="toast-message">{message}</div>
        </div>
    </div>
    
    <script>
        (function() {{
            // Get toast element
            const toast = document.getElementById('{toast_id}');
            
            // Show toast with animation
            setTimeout(function() {{
                toast.style.display = 'flex';
                toast.style.opacity = '0';
                toast.style.transform = 'translateY(20px)';
                
                setTimeout(function() {{
                    toast.style.opacity = '1';
                    toast.style.transform = 'translateY(0)';
                }}, 50);
                
                // Hide toast after duration
                setTimeout(function() {{
                    toast.style.opacity = '0';
                    toast.style.transform = 'translateY(-20px)';
                    
                    // Remove from DOM after animation
                    setTimeout(function() {{
                        toast.remove();
                    }}, 300);
                }}, {duration * 1000});
            }}, 100);
        }})();
    </script>
    """
    
    st.markdown(toast_html, unsafe_allow_html=True)
```

### Guided Error Messages

```python
def show_guided_error(error_title, error_message, suggestion=None, actions=None):
    """Show an error with guidance and potential actions."""
    st.error(error_title)
    
    st.markdown(f"""
    <div class="guided-error">
        <div class="error-message">{error_message}</div>
        {f'<div class="error-suggestion">{suggestion}</div>' if suggestion else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Render action buttons if provided
    if actions:
        cols = st.columns(len(actions))
        for i, action in enumerate(actions):
            with cols[i]:
                if st.button(action["label"], key=f"error_action_{i}"):
                    # Execute the action's callback if clicked
                    if "callback" in action and callable(action["callback"]):
                        action["callback"]()
```

## 5. Responsive Design Implementation

### Fluid Container System

```python
def responsive_container(content_function, min_height="200px", max_height="600px"):
    """Create a responsive container that adapts to content and screen size."""
    # Create container with fluid height
    container_html = f"""
    <div class="responsive-container" 
         style="min-height: {min_height}; max-height: {max_height}; overflow-y: auto;">
        <div id="container-content"></div>
    </div>
    """
    
    # Create the container first
    container = st.container()
    container.markdown(container_html, unsafe_allow_html=True)
    
    # Then render the content inside it
    with container:
        content_function()
        
    # Add responsive behavior with JavaScript
    st.markdown("""
    <script>
        // Adjust container height based on content and viewport
        function adjustContainerHeight() {
            const containers = document.querySelectorAll('.responsive-container');
            const viewportHeight = window.innerHeight;
            
            containers.forEach(container => {
                const content = container.querySelector('#container-content');
                const contentHeight = content.scrollHeight;
                
                // Set container height based on content, but respect min/max constraints
                container.style.height = Math.min(
                    Math.max(contentHeight, parseInt(container.style.minHeight)), 
                    Math.min(parseInt(container.style.maxHeight), viewportHeight * 0.7)
                ) + 'px';
            });
        }
        
        // Run on load and resize
        window.addEventListener('load', adjustContainerHeight);
        window.addEventListener('resize', adjustContainerHeight);
    </script>
    """, unsafe_allow_html=True)
```

### Screen Size Detection

```python
def detect_screen_size():
    """Detect and adapt to different screen sizes."""
    # Inject JavaScript to detect screen size
    st.markdown("""
    <script>
        // Send screen dimensions to Streamlit
        function sendScreenSize() {
            const width = window.innerWidth;
            const height = window.innerHeight;
            
            window.parent.postMessage({
                type: 'streamlit:setSessionState',
                data: { screen_width: width, screen_height: height }
            }, '*');
        }
        
        // Send on load and resize
        window.addEventListener('load', sendScreenSize);
        window.addEventListener('resize', sendScreenSize);
    </script>
    """, unsafe_allow_html=True)
    
    # Get dimensions from session state
    width = st.session_state.get('screen_width', 1200)  # Default fallback
    height = st.session_state.get('screen_height', 800)  # Default fallback
    
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
```

## Implementation Plan

### Phase 1: Setup Wizard (2 weeks)

1. **Week 1**: Create wizard framework and step components
   - Implement progress tracking
   - Build step validation logic
   - Create step navigation system

2. **Week 2**: Integrate wizard with existing app
   - Connect to agent creation flow
   - Implement API key management
   - Add automatic detection for missing configuration

### Phase 2: Navigation Enhancements (2 weeks)

1. **Week 1**: Implement context-aware navigation
   - Create smart navigation bar
   - Build breadcrumb system
   - Implement tab switching logic

2. **Week 2**: Improve information architecture
   - Reorganize tab content
   - Add visual indicators for required actions
   - Implement guided navigation hints

### Phase 3: Visual System & Feedback (2 weeks)

1. **Week 1**: Design system implementation
   - Create consistent button components
   - Build status indicators
   - Implement card components

2. **Week 2**: Enhanced feedback system
   - Build toast notification system
   - Create guided error messages
   - Improve success messages and confirmations

### Phase 4: Responsive Design (1 week)

1. Create fluid container system
2. Implement screen size detection and adaptation
3. Optimize layout for mobile devices

## CSS Styling for Implementation

The implementation will require additional CSS for the new components. Here's a sample of the key styles:

```css
/* Setup Wizard Styling */
.wizard-container {
    border: 1px solid var(--neutral-300);
    border-radius: var(--radius-lg);
    padding: var(--space-lg);
    background-color: var(--neutral-100);
    box-shadow: var(--shadow-md);
    margin: var(--space-lg) 0;
}

.wizard-steps-indicator {
    margin-bottom: var(--space-md);
    color: var(--neutral-600);
    font-size: var(--font-sm);
}

.wizard-steps-indicator .current-step {
    color: var(--primary-color);
    font-weight: var(--font-weight-bold);
}

/* Navigation System */
.nav-container {
    display: flex;
    justify-content: space-between;
    margin-bottom: var(--space-md);
    border-bottom: 1px solid var(--neutral-300);
    padding-bottom: var(--space-sm);
}

.nav-button {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: var(--space-sm);
    cursor: pointer;
    border-radius: var(--radius-md);
    transition: background-color var(--transition-fast);
}

.nav-button:hover {
    background-color: var(--neutral-200);
}

.nav-button.active {
    background-color: var(--primary-light);
    color: white;
}

.nav-button.disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.nav-icon {
    font-size: var(--font-lg);
    margin-bottom: var(--space-xs);
}

.nav-label {
    font-size: var(--font-sm);
}

/* Breadcrumbs */
.breadcrumbs {
    display: flex;
    margin-bottom: var(--space-md);
    font-size: var(--font-sm);
    color: var(--neutral-600);
}

.breadcrumb-item {
    cursor: pointer;
    color: var(--primary-color);
}

.breadcrumb-separator {
    margin: 0 var(--space-xs);
}

.breadcrumb-current {
    font-weight: var(--font-weight-medium);
    color: var(--neutral-700);
}

/* Toast Notifications */
.toast {
    position: fixed;
    bottom: 20px;
    right: 20px;
    padding: var(--space-sm) var(--space-md);
    border-radius: var(--radius-md);
    background-color: white;
    box-shadow: var(--shadow-lg);
    z-index: 1000;
    display: flex;
    align-items: center;
    min-width: 250px;
    max-width: 400px;
    transition: opacity 0.3s, transform 0.3s;
}

.toast-content {
    display: flex;
    align-items: center;
}

.toast-icon {
    margin-right: var(--space-sm);
    font-size: var(--font-md);
}

.toast-message {
    flex: 1;
    font-size: var(--font-sm);
}

.success-toast {
    border-left: 4px solid var(--success-color);
}

.error-toast {
    border-left: 4px solid var(--error-color);
}

.warning-toast {
    border-left: 4px solid var(--warning-color);
}

.info-toast {
    border-left: 4px solid var(--info-color);
}
```

## Usability Testing Plan

To ensure the new UI meets user needs, a testing plan should be implemented:

1. **First-time User Testing**
   - Task: Set up a new agent without prior instructions
   - Metrics: Time to completion, error rate, need for help

2. **Navigation Testing**
   - Task: Find specific features across the interface
   - Metrics: Time to locate, navigation path, clicks required

3. **Responsive Design Testing**
   - Environments: Desktop, tablet, mobile viewports
   - Metrics: Usability on each device, content visibility

4. **Component Performance**
   - Test: Load time for different components
   - Metrics: Rendering time, interaction responsiveness

## Conclusion

This detailed plan provides specific implementation guidance for addressing the key observations about the Agent Runner UI. The focus on first-time experience through a setup wizard, combined with improved navigation and visual hierarchy, will significantly enhance usability while maintaining compatibility with the existing codebase structure.