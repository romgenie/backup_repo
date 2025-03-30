# DSPy Evaluation

```python
from dspy.evaluate import Evaluate

evaluate_program = Evaluate(devset=devset, metric=your_defined_metric, num_threads=NUM_THREADS, display_progress=True, display_table=num_rows_to_display)

evaluate_program(your_dspy_program)
```