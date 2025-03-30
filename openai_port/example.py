"""
Example usage of the enhanced reliability functionality for the OpenAI Agents SDK.
"""

import asyncio
import os
from typing import List

from agents import Agent, Runner, TResponseInputItem

from reliability import (
    ReliableRunner, ReliableAgent, 
    get_default_config, get_high_reliability_config, get_fast_config, get_balanced_config,
    ReliabilityConfig, SupportedLanguages
)


async def example_standard_agent_with_reliable_runner():
    """
    Example of using a standard agent with the ReliableRunner.
    """
    print("\n=== Example: Standard Agent with ReliableRunner ===\n")
    
    # Create a standard agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant. Answer questions factually based on the given context.",
    )
    
    # Create some context with factual information
    context = [
        {
            "role": "system",
            "content": "The first iPhone was released in 2007. The Eiffel Tower is 330 meters tall."
        },
        {
            "role": "user",
            "content": "When was the iPhone released and how tall is the Eiffel Tower?"
        }
    ]
    
    # Run the agent with reliability processing
    print("Running agent with reliability processing...")
    result = await ReliableRunner.run(
        agent,
        context,
        reliability_level=10,
    )
    
    print(f"Result: {result.final_output}")
    print()
    
    # Now try with a question that has no context
    context_with_hallucination = [
        {
            "role": "system",
            "content": "Answer questions factually."
        },
        {
            "role": "user",
            "content": "What is the population of Mars and who was the first person to land there?"
        }
    ]
    
    # Run the agent with reliability processing
    print("Running agent with potential hallucination...")
    result = await ReliableRunner.run(
        agent,
        context_with_hallucination,
        reliability_level=10,
    )
    
    print(f"Result: {result.final_output}")


async def example_reliable_agent():
    """
    Example of using a ReliableAgent.
    """
    print("\n=== Example: ReliableAgent ===\n")
    
    # Create a reliable agent
    agent = ReliableAgent(
        name="Assistant",
        instructions="You are a helpful assistant. Answer questions factually based on the given context.",
        reliability_level=10,
    )
    
    # Create some context with factual information
    context = [
        {
            "role": "system",
            "content": "The first iPhone was released in 2007. The Eiffel Tower is 330 meters tall."
        },
        {
            "role": "user",
            "content": "When was the iPhone released and how tall is the Eiffel Tower?"
        }
    ]
    
    # Run the agent
    print("Running reliable agent...")
    result = await agent.run(context)
    
    print(f"Result: {result.final_output}")
    print()
    
    # Now try with a question that has no context
    context_with_hallucination = [
        {
            "role": "system",
            "content": "Answer questions factually."
        },
        {
            "role": "user",
            "content": "What is the population of Mars and who was the first person to land there?"
        }
    ]
    
    # Run the agent
    print("Running reliable agent with potential hallucination...")
    result = await agent.run(context_with_hallucination)
    
    print(f"Result: {result.final_output}")


async def example_different_reliability_levels():
    """
    Example of using different reliability levels.
    """
    print("\n=== Example: Different Reliability Levels ===\n")
    
    # Create a standard agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant. Answer questions factually based on the given context.",
    )
    
    # Context with potential hallucination
    context_with_hallucination = [
        {
            "role": "system",
            "content": "Answer questions factually."
        },
        {
            "role": "user",
            "content": "What is the population of Mars and who was the first person to land there?"
        }
    ]
    
    # Run with reliability level 0 (no processing)
    print("Running with reliability level 0 (no processing)...")
    result = await ReliableRunner.run(
        agent,
        context_with_hallucination,
        reliability_level=0,
    )
    
    print(f"Result: {result.final_output}")
    print()
    
    # Run with reliability level 10 (full processing)
    print("Running with reliability level 10 (full processing)...")
    result = await ReliableRunner.run(
        agent,
        context_with_hallucination,
        reliability_level=10,
    )
    
    print(f"Result: {result.final_output}")


async def example_subtasks():
    """
    Example of using subtasks.
    """
    print("\n=== Example: Subtasks ===\n")
    
    # Create a reliable agent
    agent = ReliableAgent(
        name="Assistant",
        instructions="You are a helpful assistant. Follow instructions step by step.",
        reliability_level=10,
    )
    
    # Define a sequence of subtasks
    subtasks = [
        "Generate three potential topics for a blog post about artificial intelligence.",
        "Choose the most interesting topic from the list above and create an outline for the blog post.",
        "Write the introduction section for this blog post based on the outline.",
    ]
    
    # Run subtasks
    print("Running subtasks...")
    result = await agent.run_subtasks(subtasks)
    
    print(f"Final Result: {result}")


async def example_language_transformation():
    """
    Example of using language transformation.
    """
    print("\n=== Example: Language Transformation ===\n")
    
    # Create a reliable agent with language transformation
    agent = ReliableAgent(
        name="Assistant",
        instructions="You are a helpful assistant. Answer questions factually based on the given context.",
        reliability_level=10,
        target_language=SupportedLanguages.SPANISH,
    )
    
    # Create some context with factual information
    context = [
        {
            "role": "system",
            "content": "The first iPhone was released in 2007. The Eiffel Tower is 330 meters tall."
        },
        {
            "role": "user",
            "content": "When was the iPhone released and how tall is the Eiffel Tower?"
        }
    ]
    
    # Run the agent
    print("Running agent with language transformation to Spanish...")
    result = await agent.run(context)
    
    print(f"Result: {result.final_output}")


async def example_custom_configuration():
    """
    Example of using custom configuration.
    """
    print("\n=== Example: Custom Configuration ===\n")
    
    # Create a custom configuration
    config = ReliabilityConfig()
    
    # Modify configuration
    config.validation.url_validator.confidence_threshold = 0.8
    config.validation.number_validator.confidence_threshold = 0.8
    config.editor.additional_instructions = "If information is flagged as suspicious, replace it with 'unknown'."
    config.cache.enabled = True
    config.cache.expiry_time = 1800  # 30 minutes
    
    # Create a reliable agent with custom configuration
    agent = ReliableAgent(
        name="Assistant",
        instructions="You are a helpful assistant. Answer questions factually based on the given context.",
        config=config,
    )
    
    # Context with potential hallucination
    context_with_hallucination = [
        {
            "role": "system",
            "content": "Answer questions factually."
        },
        {
            "role": "user",
            "content": "What is the population of Mars and who was the first person to land there?"
        }
    ]
    
    # Run the agent
    print("Running agent with custom configuration...")
    result = await agent.run(context_with_hallucination)
    
    print(f"Result: {result.final_output}")


async def example_predefined_configurations():
    """
    Example of using predefined configurations.
    """
    print("\n=== Example: Predefined Configurations ===\n")
    
    # Context with potential hallucination
    context_with_hallucination = [
        {
            "role": "system",
            "content": "Answer questions factually."
        },
        {
            "role": "user",
            "content": "What is the population of Mars and who was the first person to land there?"
        }
    ]
    
    # Create an agent with high reliability configuration
    high_reliability_agent = ReliableAgent(
        name="High Reliability Assistant",
        instructions="You are a helpful assistant. Answer questions factually based on the given context.",
        config=get_high_reliability_config(),
    )
    
    # Run the agent
    print("Running agent with high reliability configuration...")
    result = await high_reliability_agent.run(context_with_hallucination)
    
    print(f"High Reliability Result: {result.final_output}")
    print()
    
    # Create an agent with fast configuration
    fast_agent = ReliableAgent(
        name="Fast Assistant",
        instructions="You are a helpful assistant. Answer questions factually based on the given context.",
        config=get_fast_config(),
    )
    
    # Run the agent
    print("Running agent with fast configuration...")
    result = await fast_agent.run(context_with_hallucination)
    
    print(f"Fast Result: {result.final_output}")
    print()
    
    # Create an agent with balanced configuration
    balanced_agent = ReliableAgent(
        name="Balanced Assistant",
        instructions="You are a helpful assistant. Answer questions factually based on the given context.",
        config=get_balanced_config(),
    )
    
    # Run the agent
    print("Running agent with balanced configuration...")
    result = await balanced_agent.run(context_with_hallucination)
    
    print(f"Balanced Result: {result.final_output}")


async def main():
    """
    Run all examples.
    """
    await example_standard_agent_with_reliable_runner()
    await example_reliable_agent()
    await example_different_reliability_levels()
    await example_subtasks()
    await example_language_transformation()
    await example_custom_configuration()
    await example_predefined_configurations()


if __name__ == "__main__":
    # Make sure the OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        print("Example: export OPENAI_API_KEY=sk-...")
        exit(1)
    
    asyncio.run(main())
