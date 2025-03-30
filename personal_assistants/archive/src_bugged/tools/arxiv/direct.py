"""
Direct API for ArXiv tools.

This module provides direct function access to the ArXiv tools for use outside of agents.
Instead of using the function_tool decorated functions from arxiv_tools.py, this module
implements the same functionality but with normal Python functions that aren't decorated
for use with the OpenAI Agents SDK.

Usage:
    from src.tools.arxiv.direct import search_papers, download_paper, search_and_download
    
    # Search for papers
    results = search_papers(
        query="attention is all you need",
        max_results=5,
        categories=["cs.AI", "cs.LG"]
    )
    
    # Download a paper
    download_result = download_paper("1706.03762")
    
    # Search and download in one operation
    combined_result = search_and_download(
        query="transformers",
        max_results=3
    )
"""
import os
import time
import logging
import arxiv
import requests
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from src.config.settings import KNOWLEDGE_DIRECTORY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arxiv_direct")

# Constants - copied from arxiv_tools.py
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MAX_RESULTS_LIMIT = 100
RATE_LIMIT_DELAY = 1  # seconds between requests
CATEGORIES = {
    "cs": "Computer Science",
    "math": "Mathematics",
    "physics": "Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
    "econ": "Economics",
    "eess": "Electrical Engineering and Systems Science",
    "astro-ph": "Astrophysics",
    "cond-mat": "Condensed Matter",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nlin": "Nonlinear Sciences",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "quant-ph": "Quantum Physics"
}

def _safe_request_with_retry(func, *args, **kwargs):
    """
    Execute a function with retry logic for network operations.
    
    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function call
    """
    retries = 0
    last_error = None
    
    while retries < MAX_RETRIES:
        try:
            # Add rate limiting
            if retries > 0:
                time.sleep(RETRY_DELAY)
            else:
                time.sleep(RATE_LIMIT_DELAY)
                
            return func(*args, **kwargs)
        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            retries += 1
            last_error = e
            logger.warning(f"Request failed (attempt {retries}/{MAX_RETRIES}): {str(e)}")
            # Exponential backoff
            time.sleep(RETRY_DELAY * (2 ** (retries - 1)))
        except Exception as e:
            # For non-network errors, don't retry
            raise e
    
    # If we get here, all retries failed
    raise last_error

def _normalize_arxiv_id(paper_id: str) -> str:
    """
    Normalize ArXiv ID by removing prefixes and handling versions.
    
    Args:
        paper_id: ArXiv paper ID in any format
        
    Returns:
        Normalized ID suitable for ArXiv API
    """
    # Strip "arxiv:" prefix if present
    if paper_id.lower().startswith("arxiv:"):
        paper_id = paper_id[6:]
        
    # Remove URL components if present
    if "/" in paper_id and "arxiv.org" in paper_id:
        # Extract just the ID part from URL
        parts = paper_id.split("/")
        for part in parts:
            if part.strip() and (part.isdigit() or "." in part):
                paper_id = part
                break
                
    # Clean up any remaining non-ID characters
    paper_id = paper_id.strip("/").strip()
    
    return paper_id

def _create_safe_filename(title: str, paper_id: str, file_extension: str = ".pdf") -> str:
    """
    Create a safe filename from a paper title and ID.
    
    Args:
        title: Paper title
        paper_id: ArXiv paper ID
        file_extension: File extension (default: .pdf)
        
    Returns:
        Safe filename
    """
    if not title:
        return f"arxiv_paper_{paper_id}{file_extension}"
        
    # Remove problematic characters
    safe_title = title.replace('/', '_').replace('\\', '_')
    safe_title = ''.join(c if c.isalnum() or c in [' ', '_', '-', '.'] else '_' for c in safe_title)
    safe_title = safe_title.strip()
    
    # Truncate if too long
    if len(safe_title) > 100:
        safe_title = safe_title[:97] + "..."
        
    # Add ID and extension
    filename = f"{safe_title}_{paper_id}{file_extension}"
    
    return filename

def _download_paper_with_retry(paper, file_path: str) -> bool:
    """
    Download a paper with retry logic.
    
    Args:
        paper: ArXiv paper object
        file_path: Path to save the file
        
    Returns:
        Boolean indicating success
    """
    def _do_download():
        paper.download_pdf(filename=file_path)
        return True
        
    try:
        return _safe_request_with_retry(_do_download)
    except Exception as e:
        logger.error(f"Failed to download paper: {str(e)}")
        return False

def search_papers(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
    sort_order: str = "descending",
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
        
    Returns:
        Dictionary containing search results with paper details
    """
    try:
        # Validate inputs
        if max_results > MAX_RESULTS_LIMIT:
            logger.warning(f"Requested {max_results} results, limiting to {MAX_RESULTS_LIMIT}")
            max_results = MAX_RESULTS_LIMIT
            
        # Configure sort options
        sort_options = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate
        }
        
        sort_order_options = {
            "ascending": arxiv.SortOrder.Ascending,
            "descending": arxiv.SortOrder.Descending
        }
        
        # Build advanced query
        advanced_query = []
        
        # Add main query if provided
        if query:
            advanced_query.append(f"({query})")
            
        # Add title-specific search
        if title:
            advanced_query.append(f"ti:\"{title}\"")
            
        # Add abstract-specific search
        if abstract:
            advanced_query.append(f"abs:\"{abstract}\"")
            
        # Add author search
        if author:
            advanced_query.append(f"au:\"{author}\"")
            
        # Add category filtering
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            advanced_query.append(f"({cat_query})")
            
        # Add date range
        if date_range:
            try:
                start_date, end_date = date_range.split(",")
                # Validate dates
                datetime.strptime(start_date.strip(), "%Y-%m-%d")
                datetime.strptime(end_date.strip(), "%Y-%m-%d")
                advanced_query.append(f"submittedDate:[{start_date.strip()} TO {end_date.strip()}]")
            except (ValueError, IndexError):
                logger.warning(f"Invalid date range format: {date_range}. Expected format: YYYY-MM-DD,YYYY-MM-DD")
        
        # Combine all query parts
        final_query = " AND ".join(filter(None, advanced_query)) if advanced_query else query
        
        logger.info(f"Executing ArXiv search with query: {final_query}")
        
        # Create search client
        client = arxiv.Client()
        
        # Configure search
        search = arxiv.Search(
            query=final_query,
            max_results=max_results,
            sort_by=sort_options.get(sort_by, arxiv.SortCriterion.Relevance),
            sort_order=sort_order_options.get(sort_order, arxiv.SortOrder.Descending)
        )
        
        # Execute search with retry logic
        def _execute_search():
            return list(client.results(search))
            
        results = _safe_request_with_retry(_execute_search)
        
        # Format results
        formatted_results = []
        for paper in results:
            formatted_results.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "published": str(paper.published),
                "updated": str(paper.updated),
                "pdf_url": paper.pdf_url,
                "entry_id": paper.entry_id,
                "arxiv_id": paper.get_short_id(),
                "primary_category": paper.primary_category,
                "categories": paper.categories,
                "links": [link.href for link in paper.links],
                "doi": getattr(paper, "doi", None)
            })
        
        # Provide category information in human-readable format
        category_info = {}
        if formatted_results:
            all_categories = set()
            for paper in formatted_results:
                all_categories.update(paper["categories"])
                
            for category in all_categories:
                main_category = category.split('.')[0] if '.' in category else category
                sub_category = category.split('.')[1] if '.' in category else None
                
                readable_name = CATEGORIES.get(main_category, main_category)
                if sub_category:
                    category_info[category] = f"{readable_name} > {sub_category}"
                else:
                    category_info[category] = readable_name
        
        return {
            "query": final_query,
            "original_query": query,
            "total_results": len(formatted_results),
            "papers": formatted_results,
            "categories": category_info,
            "search_parameters": {
                "max_results": max_results,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "author": author,
                "title": title,
                "abstract": abstract,
                "categories": categories,
                "date_range": date_range
            }
        }
    except Exception as e:
        logger.error(f"ArXiv search error: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "query": query
        }

def download_paper(
    paper_id: str,
    download_dir: Optional[str] = None,
    filename: Optional[str] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Download a paper from ArXiv by its ID.
    
    Args:
        paper_id: ArXiv paper ID (e.g., "1706.03762" or "arxiv:1706.03762")
        download_dir: Directory to save the paper to (default: knowledge directory)
        filename: Custom filename (default: uses the paper title)
        overwrite: Whether to overwrite existing files (default: False)
        
    Returns:
        Dictionary with download status and file path
    """
    try:
        # Normalize the paper ID
        normalized_id = _normalize_arxiv_id(paper_id)
        logger.info(f"Downloading paper with ID: {normalized_id}")
            
        # Configure search for specific paper
        client = arxiv.Client()
        search = arxiv.Search(id_list=[normalized_id])
        
        # Get paper details with retry
        def _get_paper_details():
            return list(client.results(search))
            
        papers = _safe_request_with_retry(_get_paper_details)
        
        if not papers:
            logger.warning(f"Paper with ID {normalized_id} not found")
            return {
                "status": "error",
                "message": f"Paper with ID {normalized_id} not found",
                "paper_id": normalized_id
            }
        
        paper = papers[0]
        
        # Determine download directory
        if not download_dir:
            download_dir = KNOWLEDGE_DIRECTORY
            
        # Create directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            filename = _create_safe_filename(paper.title, normalized_id)
        elif not filename.lower().endswith('.pdf'):
            # Ensure filename has .pdf extension
            filename += '.pdf'
            
        # Full file path
        file_path = os.path.join(download_dir, filename)
        
        # Check if file already exists
        if os.path.exists(file_path) and not overwrite:
            logger.info(f"File already exists: {file_path}")
            return {
                "status": "skipped",
                "message": "File already exists (use overwrite=True to force download)",
                "paper_id": normalized_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "download_dir": download_dir
            }
        
        # Download the paper with retry logic
        download_success = _download_paper_with_retry(paper, file_path)
        
        if download_success and os.path.exists(file_path):
            logger.info(f"Successfully downloaded paper to {file_path}")
            return {
                "status": "success",
                "paper_id": normalized_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "download_dir": download_dir,
                "categories": paper.categories,
                "published": str(paper.published),
                "doi": getattr(paper, "doi", None)
            }
        else:
            logger.error(f"Download failed for paper ID {normalized_id}")
            return {
                "status": "error",
                "message": "Download failed",
                "paper_id": normalized_id
            }
    except Exception as e:
        logger.error(f"Error downloading paper {paper_id}: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "paper_id": paper_id
        }

def batch_download(
    paper_ids: List[str],
    download_dir: Optional[str] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Download multiple papers from ArXiv by their IDs.
    
    Args:
        paper_ids: List of ArXiv paper IDs
        download_dir: Directory to save papers to (default: knowledge directory)
        overwrite: Whether to overwrite existing files (default: False)
        
    Returns:
        Dictionary with download results for each paper
    """
    if not paper_ids:
        return {
            "status": "error",
            "message": "No paper IDs provided",
            "results": []
        }
        
    # Determine download directory
    if not download_dir:
        download_dir = KNOWLEDGE_DIRECTORY
        
    # Create directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Track results
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    # Process each paper ID
    for i, paper_id in enumerate(paper_ids):
        logger.info(f"Processing paper {i+1}/{len(paper_ids)}: {paper_id}")
        
        # Rate limiting between requests
        if i > 0:
            time.sleep(RATE_LIMIT_DELAY)
            
        # Download paper
        result = download_paper(
            paper_id=paper_id,
            download_dir=download_dir,
            overwrite=overwrite
        )
        
        # Track statistics
        if result["status"] == "success":
            successful += 1
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed += 1
            
        results.append(result)
    
    return {
        "status": "complete",
        "total_papers": len(paper_ids),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "download_dir": download_dir,
        "results": results
    }
    
def search_and_download(
    query: str,
    max_results: int = 3,
    download_dir: Optional[str] = None,
    categories: Optional[List[str]] = None,
    author: Optional[str] = None,
    date_range: Optional[str] = None,
    overwrite: bool = False
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
        
    Returns:
        Dictionary with search and download results
    """
    logger.info(f"Searching and downloading papers for query: {query}")
    
    # First search for papers
    search_result = search_papers(
        query=query,
        max_results=max_results,
        categories=categories,
        author=author,
        date_range=date_range
    )
    
    # Check if search was successful
    if "status" in search_result and search_result["status"] == "error":
        return {
            "status": "error",
            "message": f"Search failed: {search_result.get('message', 'Unknown error')}",
            "query": query
        }
    
    # If no papers found, return early
    if search_result["total_results"] == 0:
        return {
            "status": "no_results",
            "message": "No papers found matching the query",
            "query": query,
            "search_results": search_result
        }
    
    # Extract paper IDs for download
    paper_ids = [paper["arxiv_id"] for paper in search_result["papers"]]
    
    # Download the papers
    download_result = batch_download(
        paper_ids=paper_ids,
        download_dir=download_dir,
        overwrite=overwrite
    )
    
    # Combine results
    combined_result = {
        "status": "complete",
        "query": query,
        "search_results": {
            "total_found": search_result["total_results"],
            "papers": search_result["papers"]
        },
        "download_results": {
            "total_papers": download_result["total_papers"],
            "successful": download_result["successful"],
            "failed": download_result["failed"],
            "skipped": download_result["skipped"]
        },
        "downloaded_files": [
            {
                "title": result["title"],
                "file_path": result["file_path"],
                "authors": result["authors"],
                "paper_id": result["paper_id"]
            }
            for result in download_result["results"]
            if result["status"] in ["success", "skipped"]
        ],
        "search_parameters": {
            "query": query,
            "max_results": max_results,
            "categories": categories,
            "author": author,
            "date_range": date_range
        }
    }
    
    return combined_result