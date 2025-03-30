##############################################################################
# Multi-Agent Classes
##############################################################################

class IngestionAgent:
    """
    Ingestion Agent (IA):
    1) Retrieves and standardizes raw documents (PDF, text)
    2) Extracts minimal metadata if available
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Ingestion Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Ingestion Agent (IA). Your responsibility is to:
1. Retrieve raw publications from designated sources (e.g., PubMed, internal repositories).
2. Convert various file formats (PDF, HTML, XML) into a consistent normalized text format.
3. Extract metadata such as the title, authors, journal/conference name, publication date, and unique identifiers (DOI, PubMed ID).

Key Requirements:
- Handle OCR artifacts if the PDF is scanned (e.g., correct typical OCR errors where possible).
- Normalize non-ASCII characters (Greek letters, special symbols) to ASCII or minimal LaTeX markup when relevant (e.g., \\alpha).
- If certain fields cannot be extracted, leave them as empty or "N/A" but do not remove the key from the JSON.

Error Handling:
- In case of partial or unreadable text, mark the corrupted portions with placeholders (e.g., "[UNREADABLE]").
- If the document is locked or inaccessible, set an error flag in the output JSON.

POSITIVE EXAMPLE:
Input: A complex PDF with LaTeX symbols and tables about IL-6 inhibition in rheumatoid arthritis
Output: {
  "metadata": {
    "title": "Effects of IL-6 Inhibition on Inflammatory Markers in Rheumatoid Arthritis",
    "authors": ["Jane Smith", "Robert Johnson"],
    "journal": "Journal of Immunology",
    "pub_date": "2021-05-15",
    "doi": "10.1234/jimmunol.2021.05.123",
    "pmid": "33123456"
  },
  "content": "Introduction\\nInterleukin-6 (IL-6) is a key cytokine in the pathogenesis of rheumatoid arthritis (RA)...Methods\\nPatients (n=120) were randomized to receive either IL-6 inhibitor (n=60) or placebo (n=60)..."
}

NEGATIVE EXAMPLE:
Input: A complex PDF with LaTeX symbols and tables about IL-6 inhibition
Bad Output: {
  "title": "Effects of IL-6 Inhibition",
  "text": "Interleukin-6 inhibition showed p<0.05 significance..."
}
This is incorrect because it doesn't use the expected metadata/content structure and omits required metadata fields.
"""

    def ingest_document(self, raw_text: str) -> Dict:
        """
        Standardize the raw text into a structured format with metadata.
        
        Args:
            raw_text: Input text to process
            
        Returns:
            Dict containing metadata and content
        """
        prompt = f"""
        Please analyze this document and extract the following metadata if available:
        - Title
        - Authors
        - Journal or source
        - Publication date
        - DOI or other identifiers
        
        If any field cannot be determined, mark it as "Unknown" or "N/A".
        
        Document:
        {raw_text[:5000]}  # Truncate for efficiency
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            extracted_text = response.choices[0].message.content
            
            # Extract metadata from the response
            metadata = {
                "title": "Unknown Title",
                "authors": [],
                "journal": "Unknown Journal", 
                "pub_date": "N/A",
                "doi": "N/A",
                "pmid": "N/A"
            }
            
            # Simple parsing of the LLM response
            for line in extracted_text.split('\n'):
                line = line.strip()
                if line.startswith("Title:"):
                    metadata["title"] = line[6:].strip()
                elif line.startswith("Authors:"):
                    authors_text = line[8:].strip()
                    metadata["authors"] = [a.strip() for a in authors_text.split(',') if a.strip()]
                elif line.startswith("Journal:"):
                    metadata["journal"] = line[8:].strip()
                elif line.startswith("Publication date:"):
                    metadata["pub_date"] = line[17:].strip()
                elif line.startswith("DOI:"):
                    metadata["doi"] = line[4:].strip()
                elif line.startswith("PMID:"):
                    metadata["pmid"] = line[5:].strip()
            
            return {
                "metadata": metadata,
                "content": raw_text
            }
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            # Return a default structure on error
            return {
                "metadata": {
                    "title": "Unknown Title",
                    "authors": [],
                    "journal": "Unknown Journal", 
                    "pub_date": "N/A",
                    "doi": "N/A",
                    "pmid": "N/A",
                    "error": str(e)
                },
                "content": raw_text
            }