import streamlit as st
from agents import Agent

def display_agent_info(agent: Agent):
    """
    Display information about the current agent
    
    Args:
        agent: The agent to display information about
    """
    st.divider()
    
    st.subheader("Current Agent Configuration")
    
    st.markdown(f"**Name:** {agent.name}")
    st.markdown(f"**Model:** {agent.model}")
    
    if hasattr(agent, 'model_settings') and agent.model_settings:
        st.markdown(f"**Temperature:** {agent.model_settings.temperature}")
        st.markdown(f"**Max Tokens:** {agent.model_settings.max_tokens}")
    
    st.markdown("**Instructions:**")
    st.markdown(f"```\n{agent.instructions}\n```")
    
    # Display tools if any
    if hasattr(agent, 'tools') and agent.tools:
        st.markdown("**Tools:**")
        for tool in agent.tools:
            tool_type = tool.__class__.__name__
            tool_name = getattr(tool, 'name', tool_type)
            st.markdown(f"- {tool_name} ({tool_type})")