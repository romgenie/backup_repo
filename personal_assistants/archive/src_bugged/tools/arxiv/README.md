# ArXiv Tools Module

A comprehensive set of tools for searching, downloading, and managing academic papers from ArXiv.

## Overview

This module provides a set of powerful tools for working with the ArXiv repository of academic papers. It supports advanced search capabilities, paper downloads (both individual and batch), and combined search-and-download operations.

## Features

- **Advanced Search**: Search for papers with filters for categories, authors, title, abstract, and date ranges
- **Smart Downloads**: Download papers with intelligent file naming, overwrite protection, and error handling
- **Batch Operations**: Download multiple papers in one operation
- **Robust Implementation**: Includes rate limiting, retry logic, and error handling
- **Context-Aware Workflows**: Designed to work with multi-step agent workflows
- **Claude Integration**: Works seamlessly with Claude for academic research tasks

## Usage Options

### 1. Using with Agents

These tools are designed to be used with the OpenAI Agents SDK. When creating an agent, pass the tool objects directly:

```python
from agents import Agent
from src.tools.arxiv.arxiv_tools import (
    search_arxiv,
    download_paper_from_arxiv,
    batch_download_papers,
    search_and_download_papers
)

agent = Agent(
    name="ArXivResearchAgent",
    instructions="You help with academic research...",
    tools=[
        search_arxiv,  # Pass the tool object directly, not a function call
        download_paper_from_arxiv,
        batch_download_papers,
        search_and_download_papers
    ]
)
```

### 2. Direct API

For direct usage in Python code (not via agents), use the `direct` module which provides access to the underlying functions:

```python
from src.tools.arxiv.direct import search_papers, download_paper, batch_download, search_and_download

# Search for papers
results = search_papers(
    query="attention is all you need",
    max_results=5,
    categories=["cs.AI", "cs.LG"]
)

# Download a paper
download_result = download_paper(
    paper_id="1706.03762"
)

# Search and download in one operation
combined_result = search_and_download(
    query="transformers",
    max_results=3
)
```

## Tools

### 1. search_arxiv

Search for papers on ArXiv with advanced filtering options.

```python
result = search_arxiv(
    query="attention mechanisms",
    max_results=5,
    sort_by="relevance",
    sort_order="descending",
    categories=["cs.AI", "cs.LG"],
    author="Vaswani",
    title="Attention",
    abstract="transformer",
    date_range="2017-01-01,2023-12-31"
)
```

### 2. download_paper_from_arxiv

Download a specific paper by its ArXiv ID.

```python
result = download_paper_from_arxiv(
    paper_id="1706.03762",      # Original Transformer paper
    download_dir="/path/to/save",  # Optional 
    filename="transformer_paper.pdf",  # Optional
    overwrite=False  # Whether to overwrite existing files
)
```

### 3. batch_download_papers

Download multiple papers in a single operation.

```python
result = batch_download_papers(
    paper_ids=["1706.03762", "2005.14165", "2104.09864"],
    download_dir="/path/to/save",  # Optional
    overwrite=False  # Whether to overwrite existing files
)
```

### 4. search_and_download_papers

Search for papers and automatically download the results in one operation.

```python
result = search_and_download_papers(
    query="BERT language model",
    max_results=3,
    categories=["cs.CL"],
    author="Devlin",
    date_range="2018-01-01,2022-12-31",
    download_dir="/path/to/save",  # Optional
    overwrite=False  # Whether to overwrite existing files
)
```

## Usage with Claude

When using with Claude as an agent, consider these example prompts for effective workflows:

### Example 1: Finding and downloading specific papers

```
Find and download the original Transformer paper by Vaswani et al. and the BERT paper by Devlin et al., then add them to my vector store.
```

### Example 2: Collecting papers on a topic

```
Search for papers about language models published in the last two years, download the top 3 most relevant ones, and add them to my vector store for later searching.
```

### Example 3: Research workflow

```
I'm researching attention mechanisms in transformer models. Find relevant papers, download them to my knowledge directory, add them to my vector store, and then help me search for information about multi-head attention within these papers.
```

## Implementation Details

- **Rate Limiting**: Implements smart rate limiting to avoid overloading the ArXiv API
- **Retry Logic**: Automatically retries failed requests with exponential backoff
- **Error Handling**: Comprehensive error handling and reporting
- **File Management**: Smart naming and handling of downloaded papers
- **Category Metadata**: Human-readable category information

## Integration

This module is designed to be used by agents in a larger system, particularly as part of multi-step workflows that involve:

1. Searching for academic papers
2. Downloading papers to a knowledge directory
3. Adding those papers to a vector store
4. Searching within the papers using vector queries

The tools seamlessly coordinate to enable efficient research workflows.