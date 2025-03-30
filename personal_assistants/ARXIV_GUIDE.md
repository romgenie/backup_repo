# ArXiv Tools Guide for Search Agent

This guide provides detailed instructions for the Search Agent on how to effectively use the ArXiv tools for academic paper search and download workflows.

## Tools Overview

The Search Agent has access to the following ArXiv tools:

1. **search_arxiv**: Find papers on ArXiv with advanced filtering
2. **download_paper_from_arxiv**: Download a specific paper by ID
3. **batch_download_papers**: Download multiple papers at once
4. **search_and_download_papers**: Search and download in one operation

## Workflow Patterns

### Pattern 1: Search then Download

Use this pattern when you want to first search for papers, evaluate the results, and then selectively download:

```
1. Use search_arxiv to find papers matching query
2. Evaluate search results to identify relevant papers
3. Use download_paper_from_arxiv for each relevant paper
4. Report downloaded papers and their locations
```

### Pattern 2: Direct Combined Operation

Use this pattern for efficiency when you want to automatically download the top results:

```
1. Use search_and_download_papers with appropriate filters
2. Report the downloaded papers, locations, and brief summaries
```

### Pattern 3: Known Paper Retrieval

Use this pattern when the user knows specific papers they want:

```
1. Use batch_download_papers with the list of ArXiv IDs
2. Report the downloaded papers and their locations
```

## ArXiv Categories

When searching papers, consider using category filters for more relevant results. Common categories:

- **Computer Science**:
  - `cs.AI`: Artificial Intelligence
  - `cs.CL`: Computation and Language (NLP)
  - `cs.CV`: Computer Vision
  - `cs.LG`: Machine Learning
  - `cs.NE`: Neural and Evolutionary Computing
  - `cs.RO`: Robotics

- **Math and Statistics**:
  - `math.ST`: Statistics
  - `stat.ML`: Machine Learning (Statistics)
  - `stat.TH`: Statistical Theory

- **Physics**:
  - `quant-ph`: Quantum Physics
  - `physics.comp-ph`: Computational Physics

## User Intent Recognition

Recognize these common academic paper search intents:

1. **Specific Paper Request**: User mentions a paper by title or authors
   ```
   "Find the Transformer paper by Vaswani"
   ```

2. **Topic Exploration**: User wants papers on a topic
   ```
   "Find papers about reinforcement learning"
   ```

3. **Literature Review**: User wants a comprehensive set of papers
   ```
   "Get the most cited papers on image recognition from the last 3 years"
   ```

4. **Follow-up Research**: User wants related papers
   ```
   "Find papers that cite this one"
   ```

## Response Formatting Guidelines

When reporting search and download results to users:

1. **For search results**:
   - Include paper titles, authors, and publication dates
   - Provide brief summaries of the key findings
   - Mention categories and citation counts when available

2. **For download results**:
   - Confirm which papers were successfully downloaded
   - Provide the file paths where papers are saved
   - Mention file sizes to confirm complete downloads
   - Note if any papers were skipped (already exist) or failed

3. **For multi-step workflows**:
   - Explain the workflow steps you're performing
   - Provide intermediate results as appropriate
   - Summarize the final state (e.g., "3 papers downloaded to knowledge directory")

## Integration with Vector Store

After downloading papers, inform users that:

1. The papers are now available in the knowledge directory
2. They can be added to the vector store using the Vector Store Agent
3. Once added, they can search within these papers using semantic queries

Example: "I've downloaded 3 papers about transformers to your knowledge directory. You can now ask the Vector Store Agent to add these to your vector store for semantic searching."

## Input Parameter Guidelines

### search_arxiv

- **query**: Keep search terms specific but not too narrow
- **max_results**: Default to 5 unless user requests more
- **categories**: Use when topic clearly matches specific categories
- **author**: Include when author names are provided
- **date_range**: Format as "YYYY-MM-DD,YYYY-MM-DD"

### download_paper_from_arxiv

- **paper_id**: Extract from search results or user input
- **download_dir**: Use knowledge directory by default
- **overwrite**: Set to True when explicitly requested to replace existing files

### batch_download_papers

- **paper_ids**: Collect IDs from search results or user input
- **download_dir**: Use knowledge directory by default

### search_and_download_papers

- **query**: Keep search terms focused on specific topics
- **max_results**: Limit to 3 by default to avoid overwhelming
- **categories**: Always specify when known to improve relevance

## Handling Common User Requests

### "Find me papers about X"

```python
search_and_download_papers(
    query="X",
    max_results=3,
    categories=["relevant_category"]
)
```

### "Download the paper by Author Y on topic Z"

```python
# First search to find the paper
search_result = search_arxiv(
    query="Z",
    author="Y",
    max_results=5
)

# Then download the most relevant result
paper_id = search_result["papers"][0]["arxiv_id"]
download_paper_from_arxiv(paper_id=paper_id)
```

### "Get the latest research on X from the last year"

```python
from datetime import datetime, timedelta

# Calculate date range for last year
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
date_range = f"{start_date},{end_date}"

search_and_download_papers(
    query="X",
    max_results=5,
    date_range=date_range,
    sort_by="submittedDate",
    sort_order="descending"
)
```