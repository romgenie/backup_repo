# Integrating DSPy with AI-Scientist LLM System

This guide explains how to integrate DSPy with the existing LLM system in AI-Scientist.

## Overview

The AI-Scientist project uses a custom LLM client in `ai_scientist/llm.py` that supports multiple LLM providers (OpenAI, Anthropic, etc.). DSPy has its own LLM client system that can be integrated with this existing system.

## Integration Approach

### Option 1: Using DSPy's LLM Client

```python
import dspy

# Configure DSPy to use the same LLM as AI-Scientist
# For OpenAI models
lm = dspy.LM('openai/gpt-4o-mini')
# For Anthropic models
lm = dspy.LM('anthropic/claude-3-5-sonnet-20240620')
# Configure DSPy to use this LLM
dspy.configure(lm=lm)
```

### Option 2: Creating a Custom LLM Adapter

You can create a custom adapter that wraps the AI-Scientist LLM client:

```python
import dspy
from ai_scientist.llm import get_response_from_llm, create_client

class AIScientistLM(dspy.LM):
    def __init__(self, model_name):
        self.client, self.model = create_client(model_name)
        self.system_message = "You are a helpful AI assistant."
        
    def basic_request(self, prompt, **kwargs):
        response, _ = get_response_from_llm(
            msg=prompt,
            client=self.client,
            model=self.model,
            system_message=self.system_message,
            temperature=kwargs.get('temperature', 0.7)
        )
        return [response]

# Use the custom LLM adapter
lm = AIScientistLM("gpt-4o-mini-2024-07-18")
dspy.configure(lm=lm)
```
