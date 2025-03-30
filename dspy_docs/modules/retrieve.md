# DSPy Retrieve Module

The `Retrieve` module provides retrieval-augmented generation capabilities, allowing you to retrieve relevant information from a corpus.

## Basic Usage

```python
import dspy

# Set up a retrieval model (RM)
retriever = dspy.ColBERTv2(url="http://example.com/colbert")
dspy.settings.configure(rm=retriever)

# Create a Retrieve module
retrieve = dspy.Retrieve(k=3)  # Retrieve top 3 passages

# Use the module
result = retrieve("What are the key components of a transformer architecture?")
for i, passage in enumerate(result.passages):
    print(f"Passage {i+1}: {passage}")
```

## AI-Scientist Integration

Retrieve is valuable for AI-Scientist tasks that require knowledge retrieval:

1. **Literature Review**: Retrieving relevant papers and research findings
2. **Background Knowledge**: Retrieving factual information for experiment design
3. **Citation Generation**: Finding relevant citations for paper writing

### Example: Enhanced Literature Review

```python
import dspy

# Set up a retrieval model with research papers corpus
papers_retriever = dspy.ColBERTv2(url="http://example.com/papers_corpus")
dspy.settings.configure(rm=papers_retriever)

# Define a signature for literature review
class LiteratureReviewer(dspy.Signature):
    """Review literature relevant to a research topic."""
    
    topic = dspy.InputField()
    context = dspy.InputField()
    review = dspy.OutputField()

# Create a module combining retrieval and reasoning
class RAGReviewer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.summarize = dspy.ChainOfThought(LiteratureReviewer)
    
    def forward(self, topic):
        # Retrieve relevant papers
        retrieved = self.retrieve(topic)
        # Summarize the findings
        review = self.summarize(topic=topic, context=retrieved.passages)
        return review

# Use the module
reviewer = RAGReviewer()
result = reviewer("Transformer architecture efficiency improvements")
print(result.review)
```
