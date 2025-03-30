# DSPy ReAct Module

The `ReAct` module implements the Reasoning and Acting framework, enabling LLMs to interleave reasoning steps with actions.

## Basic Usage

```python
import dspy

# Define a signature for your task
class ProblemSolver(dspy.Signature):
    """Solve complex problems that require external tools."""
    
    problem = dspy.InputField()
    solution = dspy.OutputField()

# Create a ReAct module
solver = dspy.ReAct(ProblemSolver)

# Use the module
result = solver(problem="What is the population of the capital of France?")
print(f"Solution: {result.solution}")
```

## AI-Scientist Integration

ReAct is particularly useful for AI-Scientist tasks that require interacting with external tools or APIs:

1. **Data Collection**: Gathering research papers or datasets needed for experiments
2. **Experiment Execution**: Running experiments with external tools and analyzing results
3. **Literature Review**: Searching and analyzing related work

### Example: Enhanced Experiment Runner

```python
import dspy
import torch
import numpy as np

# Define available tools
def run_nanoGPT_experiment(params):
    # Run NanoGPT with given parameters
    # Return results
    return results

def analyze_results(data):
    # Analyze experimental results
    return analysis

# Define a signature for experiment running
class ExperimentRunner(dspy.Signature):
    """Run and analyze NanoGPT experiments."""
    
    experiment_config = dspy.InputField()
    results = dspy.OutputField()

# Create a ReAct module with this signature
runner = dspy.ReAct(ExperimentRunner)
# Define the available tools
runner.set_tools([
    run_nanoGPT_experiment,
    analyze_results
])

# Use the module
result = runner(experiment_config={"model_size": "124M", "learning_rate": 0.0003})
```
