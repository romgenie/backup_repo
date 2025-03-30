"""
File download tool for the vector store agent.
"""
from typing import Dict, Any
import os
import tempfile
import requests
from agents import function_tool

@function_tool
def download_file_from_url(url: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Download a file from a URL to a specified directory.
    
    Args:
        url: The URL to download the file from
        output_dir: Directory to save the file in (optional, uses a temporary directory if not specified)
        
    Returns:
        Dictionary with download status and file path
    """
    try:
        # Determine output directory
        if not output_dir:
            output_dir = tempfile.gettempdir()
        
        # Make sure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get filename from URL or use a generic name
        file_name = os.path.basename(url.split('?')[0])
        if not file_name:
            file_name = "downloaded_file"
            
        # Add an extension based on content type if possible
        if '.' not in file_name:
            # Send a HEAD request to get content type
            head_response = requests.head(url, allow_redirects=True)
            content_type = head_response.headers.get('Content-Type', '')
            
            # Map content type to extension
            content_type_map = {
                'application/pdf': '.pdf',
                'text/plain': '.txt',
                'text/html': '.html',
                'application/json': '.json',
                'application/octet-stream': '.bin'
            }
            
            extension = content_type_map.get(content_type, '')
            if extension:
                file_name += extension
        
        # Final output path
        output_path = os.path.join(output_dir, file_name)
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return {
            "status": "success",
            "file_path": output_path,
            "file_name": file_name,
            "file_size": os.path.getsize(output_path),
            "url": url
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "url": url
        }