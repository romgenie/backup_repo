import streamlit as st

def streaming_config_module():
    st.header("Streaming Configuration")
    
    # Initialize session state
    if 'use_streaming' not in st.session_state:
        st.session_state.use_streaming = False
    if 'stream_raw_responses' not in st.session_state:
        st.session_state.stream_raw_responses = True
    if 'stream_complete_items' not in st.session_state:
        st.session_state.stream_complete_items = True
    if 'stream_agent_updates' not in st.session_state:
        st.session_state.stream_agent_updates = True
    if 'stream_token_delay' not in st.session_state:
        st.session_state.stream_token_delay = 0.0
    if 'stream_buffer_size' not in st.session_state:
        st.session_state.stream_buffer_size = 1
    
    # Create form
    with st.form(key="streaming_config_form"):
        st.markdown("""
        ### Streaming vs. Non-Streaming
        
        Configure whether to use streaming for agent responses:
        
        **Streaming Mode**:
        - Shows responses as they're generated (token by token)
        - Requires async handling (with `await` and `async for`)
        - Uses `Runner.run_streamed()` instead of `Runner.run()`
        
        **Non-Streaming Mode**:
        - Shows responses only after completion
        - Can be used synchronously with `Runner.run_sync()`
        - Better for situations where you don't need real-time updates
        """)
        
        use_streaming = st.checkbox(
            "Enable Streaming",
            value=st.session_state.use_streaming,
            help="When enabled, agent responses will stream in real-time"
        )
        
        if use_streaming:
            # Stream Event Filtering
            st.subheader("Stream Event Filtering")
            with st.expander("Configure which stream events to process"):
                st.markdown("""
                Different stream event types provide different levels of information:
                
                - **Raw Response Events**: Token-by-token updates directly from the model
                - **Complete Item Events**: Notification when a complete item is generated (messages, tool calls, etc.)
                - **Agent Update Events**: Notification when the agent changes during handoffs
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    stream_raw_responses = st.checkbox(
                        "Process Raw Responses",
                        value=st.session_state.stream_raw_responses,
                        help="Process token-by-token updates"
                    )
                    
                    stream_complete_items = st.checkbox(
                        "Process Complete Items",
                        value=st.session_state.stream_complete_items,
                        help="Process complete items like messages and tool calls"
                    )
                
                with col2:
                    stream_agent_updates = st.checkbox(
                        "Process Agent Updates",
                        value=st.session_state.stream_agent_updates,
                        help="Process agent handoff events"
                    )
            
            # Advanced Streaming Performance
            st.subheader("Streaming Performance Options")
            with st.expander("Advanced streaming options"):
                st.markdown("""
                These settings affect how streaming data is processed:
                
                - **Token Delay**: Artificial delay between tokens (useful for UX)
                - **Stream Buffer Size**: Number of tokens to buffer before processing
                """)
                
                stream_token_delay = st.slider(
                    "Token Delay (seconds)",
                    min_value=0.0,
                    max_value=0.5,
                    value=st.session_state.stream_token_delay,
                    step=0.01,
                    help="Add artificial delay between tokens for better UX"
                )
                
                stream_buffer_size = st.number_input(
                    "Stream Buffer Size",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.stream_buffer_size,
                    help="Number of tokens to buffer before processing (higher values may improve performance but increase latency)"
                )
        
        # Submit button
        submitted = st.form_submit_button("Save Streaming Configuration")
        
        if submitted:
            st.session_state.use_streaming = use_streaming
            
            if use_streaming:
                st.session_state.stream_raw_responses = stream_raw_responses
                st.session_state.stream_complete_items = stream_complete_items
                st.session_state.stream_agent_updates = stream_agent_updates
                st.session_state.stream_token_delay = stream_token_delay
                st.session_state.stream_buffer_size = stream_buffer_size
            
            st.success("Streaming configuration saved!")
    
    # Show generated code examples
    st.subheader("Streaming Configuration Code")
    
    if st.session_state.use_streaming:
        code = """# Streaming mode example (async)
from agents import Agent, Runner
from typing import Optional

agent = Agent(name="My Agent", instructions="Help the user")

# Async streaming example
async def run_agent_streamed(input_text):
    result = Runner.run_streamed(agent, input=input_text)
    
    # Process streaming events"""
        
        if st.session_state.stream_raw_responses:
            code += """
    async for event in result.stream_events():
        if event.type == "raw_response_event" and hasattr(event.data, "delta"):
            # Display token by token"""
            
            if st.session_state.stream_token_delay > 0:
                code += f"""
            import asyncio
            print(event.data.delta, end="", flush=True)
            await asyncio.sleep({st.session_state.stream_token_delay})  # Add token delay"""
            else:
                code += """
            print(event.data.delta, end="", flush=True)"""
        
        if st.session_state.stream_complete_items:
            code += """
        elif event.type == "run_item_stream_event":
            # Process complete items (messages, tool calls, etc.)
            item = event.item
            if item.type == "message_output_item":
                print(f"\\nMessage: {item.output}")
            elif item.type == "tool_call_item":
                print(f"\\nTool Call: {item.name}")"""
        
        if st.session_state.stream_agent_updates:
            code += """
        elif event.type == "agent_updated_stream_event":
            # Handle agent handoffs
            print(f"\\nSwitched to agent: {event.new_agent.name}")"""
            
        code += """
    
    return result.final_output
"""
    else:
        code = """# Non-streaming mode examples
from agents import Agent, Runner

agent = Agent(name="My Agent", instructions="Help the user")

# Sync example
def run_agent_sync(input_text):
    result = Runner.run_sync(agent, input=input_text)
    return result.final_output

# Async example (without streaming)
async def run_agent_async(input_text):
    result = await Runner.run(agent, input=input_text)
    return result.final_output
"""
    
    st.code(code, language="python")
    
    # Return the current configuration
    return {
        "use_streaming": st.session_state.use_streaming,
        "stream_events": {
            "raw_responses": st.session_state.stream_raw_responses if st.session_state.use_streaming else False,
            "complete_items": st.session_state.stream_complete_items if st.session_state.use_streaming else False,
            "agent_updates": st.session_state.stream_agent_updates if st.session_state.use_streaming else False
        },
        "stream_performance": {
            "token_delay": st.session_state.stream_token_delay if st.session_state.use_streaming else 0,
            "buffer_size": st.session_state.stream_buffer_size if st.session_state.use_streaming else 1
        }
    }

# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Streaming Configuration", layout="wide")
    streaming_config = streaming_config_module()
    
    # Show the current configuration
    with st.expander("Current Configuration"):
        st.json(streaming_config)