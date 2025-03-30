# DSPy ChainOfThought Module

The `ChainOfThought` module is one of DSPy's core reasoning modules. It prompts the language model to generate a step-by-step reasoning process before producing the final answer.

## Basic Usage

```python
import dspy

# Define a signature for your task
class MathProblemSolver(dspy.Signature):
    """Solve math problems step by step."""
    
    problem = dspy.InputField()
    reasoning = dspy.OutputField(desc="Step-by-step reasoning process")
    solution = dspy.OutputField(desc="The final numerical answer")

# Create a ChainOfThought module with this signature
solver = dspy.ChainOfThought(MathProblemSolver)

# Use the module
result = solver(problem="If 3x + 7 = 22, what is the value of x?")
print(f"Reasoning: {result.reasoning}")
print(f"Solution: {result.solution}")
```

## AI-Scientist Integration

ChainOfThought can be used to enhance various aspects of the AI-Scientist workflow:

1. **Idea Generation**: Use it to create more structured and well-reasoned research ideas
2. **Experiment Design**: Generate experimental setups with clear reasoning about methodological choices
3. **Result Analysis**: Analyze experimental results with step-by-step reasoning

### Example: Enhanced Review Process

```python
import dspy
from ai_scientist.perform_review import perform_review

class ExperimentReviewer(dspy.Signature):
    """Review AI experiment results thoroughly."""
    
    experiment_results = dspy.InputField()
    evaluation_criteria = dspy.InputField()
    reasoning = dspy.OutputField(desc="Detailed analysis of results")
    review = dspy.OutputField(desc="Final review summary and recommendations")

reviewer = dspy.ChainOfThought(ExperimentReviewer)
```
