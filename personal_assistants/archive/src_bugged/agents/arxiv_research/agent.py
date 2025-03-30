"""
ArXiv Research agent implementation.
"""
from typing import Dict, Any, List, Optional
from agents import Agent, function_tool
from src.agents.arxiv_research.prompts import BASE_INSTRUCTIONS, VOICE_INSTRUCTIONS
from src.tools.arxiv.arxiv_tools import (
    search_arxiv,
    download_paper_from_arxiv,
    batch_download_papers,
    search_and_download_papers
)
from pydantic import BaseModel

# Create function_tool wrappers for the arxiv functions
@function_tool
def agent_search_arxiv(
    query: str,
    max_results: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    categories: Optional[List[str]] = None,
    author: Optional[str] = None,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
    date_range: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search ArXiv for papers matching the query with advanced filtering options.
    
    Args:
        query: Search query string (e.g., "machine learning", "attention is all you need")
        max_results: Maximum number of results to return (default: 5, max: 100)
        sort_by: Sort results by "relevance", "lastUpdatedDate", or "submittedDate" (default: "relevance")
        sort_order: Sort order - "ascending" or "descending" (default: "descending")
        categories: List of ArXiv categories to filter by (e.g., ["cs.AI", "cs.LG"])
        author: Author name to filter by
        title: Search specifically in the title
        abstract: Search specifically in the abstract
        date_range: Date range in format "YYYY-MM-DD,YYYY-MM-DD" (start,end)
    """
    return search_arxiv(
        query=query,
        max_results=5 if max_results is None else max_results,
        sort_by="relevance" if sort_by is None else sort_by,
        sort_order="descending" if sort_order is None else sort_order,
        categories=categories,
        author=author,
        title=title,
        abstract=abstract,
        date_range=date_range
    )

@function_tool
def agent_download_paper(
    paper_id: str,
    download_dir: Optional[str] = None,
    filename: Optional[str] = None,
    overwrite: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Download a paper from ArXiv by its ID.
    
    Args:
        paper_id: ArXiv paper ID (e.g., "1706.03762" or "arxiv:1706.03762")
        download_dir: Directory to save the paper to (default: knowledge directory)
        filename: Custom filename (default: uses the paper title)
        overwrite: Whether to overwrite existing files (default: False)
    """
    return download_paper_from_arxiv(
        paper_id=paper_id,
        download_dir=download_dir,
        filename=filename,
        overwrite=False if overwrite is None else overwrite
    )

@function_tool
def agent_batch_download(
    paper_ids: List[str],
    download_dir: Optional[str] = None,
    overwrite: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Download multiple papers from ArXiv by their IDs.
    
    Args:
        paper_ids: List of ArXiv paper IDs
        download_dir: Directory to save papers to (default: knowledge directory)
        overwrite: Whether to overwrite existing files (default: False)
    """
    return batch_download_papers(
        paper_ids=paper_ids,
        download_dir=download_dir,
        categories=None,
        overwrite=False if overwrite is None else overwrite
    )

@function_tool
def agent_search_and_download(
    query: str,
    max_results: Optional[int] = None,
    download_dir: Optional[str] = None,
    categories: Optional[List[str]] = None,
    author: Optional[str] = None,
    date_range: Optional[str] = None,
    overwrite: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Search for papers on ArXiv and download the found papers in one operation.
    
    Args:
        query: Search query string
        max_results: Maximum number of papers to search and download (default: 3)
        download_dir: Directory to save papers to (default: knowledge directory)
        categories: List of ArXiv categories to filter by (e.g., ["cs.AI", "cs.LG"])
        author: Author name to filter by
        date_range: Date range in format "YYYY-MM-DD,YYYY-MM-DD" (start,end)
        overwrite: Whether to overwrite existing files (default: False)
    """
    return search_and_download_papers(
        query=query,
        max_results=3 if max_results is None else max_results,
        download_dir=download_dir,
        categories=categories,
        author=author,
        date_range=date_range,
        overwrite=False if overwrite is None else overwrite
    )

@function_tool
def transfer_to_VectorStoreAgent(message: str) -> str:
    """
    Transfer the conversation to the VectorStoreAgent to add papers to the vector store.
    
    This function must be called when the user wants to add papers to the vector store.
    You must provide explicit information about the papers in the message.
    
    Args:
        message: A clear message describing which papers to add to the vector store.
                 Include paper titles and file paths when available.
                 Example: "Add the following papers to the vector store: 'Attention is All You Need' at /path/file1.pdf, 'BERT' at /path/file2.pdf"
            
    Returns:
        str: Confirmation message
    """
    # This is a handoff function - the actual implementation is handled by the Agents SDK
    # We just need to define the signature and docstring to guide the LLM
    return f"Transferring to VectorStoreAgent with message: {message}"

def create_arxiv_research_agent(use_voice_optimization: bool = False) -> Agent:
    """
    Create an ArXiv research agent that specializes in academic paper searches and downloads.
    
    Args:
        use_voice_optimization: Whether to use voice-optimized instructions
        
    Returns:
        Agent: The initialized ArXiv research agent
    """
    instructions = VOICE_INSTRUCTIONS if use_voice_optimization else BASE_INSTRUCTIONS
    
    return Agent(
        name="ArXivResearchAgent",
        instructions=instructions,
        tools=[
            agent_search_arxiv,
            agent_download_paper,
            agent_batch_download,
            agent_search_and_download,
            transfer_to_VectorStoreAgent  # Added explicit handoff function
        ],
    )