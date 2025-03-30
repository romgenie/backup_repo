# DSPy Signatures

Signatures define the input and output structure for DSPy modules.

## Basic Signature

```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

## Short Form Signature

You can also use a short-form string syntax:

```python
math = dspy.ChainOfThought("question -> answer: float")
```
