class ReaderAgent:
    """
    Reader Agent (RA):
    1) Segments normalized text into logical chunks
    2) Assigns a relevance score to each segment
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Reader Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Reader Agent (RA). Your goal is to parse the normalized text and generate logical segments (e.g., paragraph-level chunks) that are likely to contain relevant knowledge. Each segment must be accompanied by a numeric Relevance Score indicating its importance for downstream extraction tasks.

Scoring Heuristics:
- Use domain knowledge (e.g., presence of known keywords, synonyms, or known entity patterns) to increase the score.
- Use structural cues (e.g., headings like "Results", "Discussion" might have higher relevance for new discoveries).
- If a segment is purely methodological (e.g., protocols or references to equipment) with no new knowledge, assign a lower score.

Edge Cases:
- Very short segments (<30 characters) or references sections might be assigned a minimal score.
- If certain sections are incomplete or corrupted, still generate a segment but label it with "score": 0.0.

POSITIVE EXAMPLE:
Input: {
  "metadata": {"title": "Antimicrobial Study"...},
  "content": "Abstract\n We tested new...\n Methods\n The protocol was...\n Results\n The compound inhibited growth of S. aureus with MIC of 0.5 μg/mL\n"
}
Output: {
  "segments": [
    {"text": "Abstract We tested new...", "score": 0.85},
    {"text": "Methods The protocol was...", "score": 0.30},
    {"text": "Results The compound inhibited growth of S. aureus with MIC of 0.5 μg/mL", "score": 0.95}
  ]
}

NEGATIVE EXAMPLE:
Input: {
  "metadata": {"title": "Antimicrobial Study"...},
  "content": "Abstract\n We tested new...\n Methods\n The protocol was...\n Results\n The compound inhibited growth of S. aureus with MIC of 0.5 μg/mL\n"
}
Bad Output: {
  "segments": [
    {"text": "The entire paper discusses antimicrobial compounds", "score": 0.5}
  ]
}
This is incorrect because it doesn't segment the text properly into logical chunks and doesn't assign differentiated relevance scores.
"""
    
    def split_into_segments(self, content: str) -> List[Dict]:
        """
        Split content into logical segments.
        
        Args:
            content: Text to segment
            
        Returns:
            List of segment dictionaries with text and estimated score
        """
        # First do a basic split on paragraph breaks
        raw_segments = content.split("\n\n")
        segments = [{"text": seg.strip(), "score": 0.0} for seg in raw_segments if seg.strip()]
        
        # Process in batches to avoid context limitations
        batch_size = 5
        processed_segments = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            batch_texts = [f"Segment {j+1}:\n{seg['text']}" for j, seg in enumerate(batch)]
            
            scores = self._batch_score_relevance(batch_texts)
            
            for j, seg in enumerate(batch):
                if j < len(scores):
                    seg["score"] = scores[j]
                processed_segments.append(seg)
        
        return processed_segments

    def _batch_score_relevance(self, segments: List[str]) -> List[float]:
        """
        Query the LLM for domain-specific relevance scores for multiple segments.
        
        Args:
            segments: List of text segments to score
            
        Returns:
            List of relevance scores
        """
        prompt = f"""
        You are a biomedical text relevance scorer.
        Rate how relevant each of the following segments is (0 to 1) for extracting
        new biomedical knowledge (e.g., relationships between diseases, drugs, genes).
        
        Consider:
        - Sections with experiments, results, or discussions usually have higher relevance
        - Methodology sections without findings have lower relevance
        - References have very low relevance
        
        For each segment, return only a single float value between 0.0 and 1.0, with no other text.
        
        {chr(10).join(segments)}
        
        Return one score per line, with no labels:
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
            
            lines = response.choices[0].message.content.strip().split('\n')
            scores = []
            
            for line in lines:
                try:
                    # Extract number from line
                    import re
                    match = re.search(r"[-+]?\d*\.\d+|\d+", line)
                    if match:
                        score = float(match.group())
                        # Ensure score is in range [0,1]
                        score = max(0.0, min(1.0, score))
                        scores.append(score)
                    else:
                        scores.append(0.5)  # Default if no number found
                except:
                    scores.append(0.5)  # Default on error
                    
            return scores
        except Exception as e:
            logger.warning(f"Failed to score segments. Error: {e}")
            return [0.5] * len(segments)  # Default on overall failure

    def score_relevance(self, segment: str) -> Tuple[float, int, int, float]:
        """
        Query the LLM for a domain-specific relevance score for a single segment.
        
        Args:
            segment: Text segment to score
            
        Returns:
            Tuple of (relevance_score, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        You are a biomedical text relevance scorer.
        Rate how relevant the following text is (0 to 1) for extracting
        new biomedical knowledge (e.g., relationships between diseases, drugs, genes):

        Text:
        {segment}

        Return only a single float value between 0.0 and 1.0, with no other text.
        Example valid responses:
        0.75
        0.3
        """
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            score_str = response.choices[0].message.content.strip()
            # Extract first float value found in response
            import re
            float_matches = re.findall(r"[-+]?\d*\.\d+|\d+", score_str)
            if float_matches:
                score = float(float_matches[0])
            else:
                score = 0.5  # default if no float found
            
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time

            return max(0.0, min(1.0, score)), prompt_tokens, completion_tokens, processing_time
        except Exception as e:
            logger.warning(f"Failed to parse relevance for segment. Error: {e}")
            return 0.5, 0, 0, time.time() - start_time  # default