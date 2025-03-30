class SummarizerAgent:
    """
    Summarizer Agent (SA):
    1) Converts high-relevance segments into concise summaries
    2) Preserves technical details important for knowledge extraction
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Summarizer Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Summarizer Agent (SA). Your task is to convert high-relevance segments into concise summaries while retaining technical detail such as gene symbols, chemical names, or numeric data that may be crucial for entity/relationship extraction.

Summarization Rules:
- Avoid discarding domain-specific terms that could indicate potential relationships. For example, retain "IL-6" or "p53" references precisely.
- If numeric data is relevant (e.g., concentrations, p-values), incorporate them verbatim if possible.
- Keep the summary length under 100 words to reduce computational overhead for downstream agents.

Handling Irrelevant Segments:
- If the Relevance Score is below a threshold (e.g., 0.2), you may skip or heavily compress the summary.
- Mark extremely low relevance segments with "summary": "[OMITTED]" if not summarizable.

POSITIVE EXAMPLE:
Input: {
  "segments": [
    {"text": "In this study, IL-6 blockade with tocilizumab led to significant reduction in DAS28 scores (p<0.001) compared to placebo. Patients receiving 8mg/kg showed the greatest improvement (mean reduction 3.2 points).", "score": 0.90},
    {"text": "The control group had p=0.01 in the secondary analysis.", "score": 0.75}
  ]
}
Output: {
  "summaries": [
    {"original_text": "In this study, IL-6 blockade with tocilizumab led to significant reduction in DAS28 scores (p<0.001) compared to placebo. Patients receiving 8mg/kg showed the greatest improvement (mean reduction 3.2 points).",
     "summary": "IL-6 blockade with tocilizumab significantly reduced DAS28 scores (p<0.001) vs placebo. The 8mg/kg dose had the best results (mean reduction 3.2 points).",
     "score": 0.90},
    {"original_text": "The control group had p=0.01 in the secondary analysis.",
     "summary": "The control group showed statistical significance (p=0.01) in secondary analysis.",
     "score": 0.75}
  ]
}

NEGATIVE EXAMPLE:
Input: {
  "segments": [
    {"text": "In this study, IL-6 blockade with tocilizumab led to significant reduction in DAS28 scores (p<0.001) compared to placebo. Patients receiving 8mg/kg showed the greatest improvement (mean reduction 3.2 points).", "score": 0.90}
  ]
}
Bad Output: {
  "summaries": [
    {"original_text": "In this study, IL-6 blockade with tocilizumab led to significant reduction in DAS28 scores (p<0.001) compared to placebo. Patients receiving 8mg/kg showed the greatest improvement (mean reduction 3.2 points).",
     "summary": "A drug helped patients improve their condition.",
     "score": 0.90}
  ]
}
This is incorrect because it discarded crucial technical details (IL-6, tocilizumab, DAS28 scores, p-value, dosage).
"""
    
    def summarize_segment(self, segment: str) -> Tuple[str, int, int, float]:
        """
        Summarize a single text segment using an LLM prompt.
        
        Args:
            segment: Text to summarize
            
        Returns:
            Tuple of (summary, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        Summarize the following biomedical text in 2-4 sentences, 
        retaining key domain terms (genes, proteins, drugs, diseases, etc.).
        Preserve any numeric data or statistical findings that indicate relationships.
        Keep the summary under 100 words.
        Provide only the summary with no additional text or formatting.
        
        Text:
        {segment}
        """
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            summary = response.choices[0].message.content.strip()
            # Handle empty or invalid responses
            if not summary:
                summary = segment[:200] + "..."  # fallback to truncated original
                
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            return summary, prompt_tokens, completion_tokens, processing_time
        except Exception as e:
            logger.warning(f"Summarization failed. Error: {e}")
            return segment[:200] + "...", 0, 0, time.time() - start_time  # fallback to truncated original