
class ConflictResolutionAgent:
    """
    Conflict Resolution Agent (CRA):
    1) Checks if newly extracted triplets conflict with existing knowledge
    2) Decides whether to keep, discard, or flag them for manual review
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Conflict Resolution Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Conflict Resolution Agent (CRA). Sometimes new triplets are detected that contradict existing knowledge (e.g., (DrugX, causes, DiseaseY) vs. (DrugX, treats, DiseaseY)). Your role is to classify these into Contradict, Agree, or Ambiguous, and decide whether the new triplet should be discarded, flagged for expert review, or integrated with caution.

LLM-based Debate:
- Use domain knowledge to see if relationships can coexist (e.g., inhibits vs. activates are typically contradictory for the same target).
- Consider partial contexts, e.g., different dosages or subpopulations.

Escalation Criteria:
- If the new triplet has high confidence but conflicts with old data that has lower confidence, consider overriding or review.
- If both are high confidence, label Contradict, prompt manual verification.

POSITIVE EXAMPLE:
Input: {
  "t_new": {"head": "Aspirin", "relation": "treats", "tail": "Headache", "confidence": 0.95},
  "t_existing": {"head": "Aspirin", "relation": "causes", "tail": "Headache", "confidence": 0.70}
}
Output: {
  "decision": "Contradict",
  "resolution": {
    "action": "review",
    "rationale": "These represent opposite effects. However, Aspirin can both treat existing headaches and cause headaches as a side effect in some individuals. Expert validation needed to clarify contexts."
  }
}

NEGATIVE EXAMPLE:
Input: {
  "t_new": {"head": "DrugX", "relation": "treats", "tail": "DiseaseY", "confidence": 0.95},
  "t_existing": {"head": "DrugX", "relation": "causes", "tail": "DiseaseY", "confidence": 0.40}
}
Bad Output: {
  "decision": "Agree",
  "resolution": {"action": "integrate", "rationale": "Both can be true."}
}
This is incorrect because it fails to recognize the direct contradiction between treats and causes, and doesn't provide a sufficiently detailed rationale.
"""

    def resolve_conflicts(
        self, 
        new_triples: List[KnowledgeTriple],
        existing_triples: List[KnowledgeTriple]
    ) -> Tuple[List[KnowledgeTriple], int, int, float]:
        """
        Compare each new triple against existing ones for direct contradictions.
        
        Args:
            new_triples: Newly extracted triples
            existing_triples: Existing knowledge graph triples
            
        Returns:
            Tuple of (final_triples, prompt_tokens, completion_tokens, processing_time)
        """
        start_time = time.time()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        final_triples = []
        for nt in new_triples:
            # Check if new triple directly contradicts any existing triple
            conflicting_triple = self._find_contradiction(nt, existing_triples)
            
            if conflicting_triple:
                # Use LLM to decide between conflicting triples
                keep, pt, ct, _ = self._resolve_contradiction(nt, conflicting_triple)
                total_prompt_tokens += pt
                total_completion_tokens += ct
                
                if keep:
                    final_triples.append(nt)
            else:
                # No conflict, keep the triple
                final_triples.append(nt)
                
        processing_time = time.time() - start_time
        return final_triples, total_prompt_tokens, total_completion_tokens, processing_time

    def _find_contradiction(
        self, 
        new_t: KnowledgeTriple, 
        existing_list: List[KnowledgeTriple]
    ) -> Optional[KnowledgeTriple]:
        """
        Check if a new triple contradicts any existing triples.
        
        Args:
            new_t: New triple to check
            existing_list: List of existing triples
            
        Returns:
            The conflicting triple if found, None otherwise
        """
        # Define opposite relation pairs (bidirectional)
        opposite_relations = {
            ("treats", "causes"),
            ("inhibits", "activates"),
            ("increases", "decreases"),
            ("upregulates", "downregulates")
        }
        
        # Create a set of relation pairs that are opposites
        contradiction_pairs = set()
        for a, b in opposite_relations:
            contradiction_pairs.add((a, b))
            contradiction_pairs.add((b, a))  # Add reverse pair too

        for ex in existing_list:
            # Check for triples about the same entities but with potentially opposing relations
            if (ex.head.lower() == new_t.head.lower() and 
                ex.tail.lower() == new_t.tail.lower()):
                # Check if relation pair is an opposite
                if (ex.relation, new_t.relation) in contradiction_pairs:
                    return ex
        return None

    def _resolve_contradiction(self, new_triple: KnowledgeTriple, existing_triple: KnowledgeTriple) -> Tuple[bool, int, int, float]:
        """
        LLM-based evaluation to decide which of two contradicting triples to keep.
        
        Args:
            new_triple: New triple being evaluated
            existing_triple: Existing conflicting triple
            
        Returns:
            Tuple of (keep_decision, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        We have two potentially contradicting biomedical statements:
        
        NEW: {new_triple.head} -> {new_triple.relation} -> {new_triple.tail}
        EXISTING: {existing_triple.head} -> {existing_triple.relation} -> {existing_triple.tail}
        
        Analyze these statements and decide:
        1. Do they truly contradict each other, or could both be valid in different contexts?
        2. Which statement appears more credible based on biological knowledge?
        
        Return one of:
        - "KEEP_NEW" if the new statement should replace or complement the existing one
        - "KEEP_EXISTING" if the existing statement is more reliable
        - "KEEP_BOTH" if they can coexist (different contexts, conditions, etc.)
        - "REVIEW" if expert human review is needed due to high uncertainty
        
        Provide only the decision code with no additional text.
        """
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                temperature=0.1
            )
            
            decision = response.choices[0].message.content.strip().upper()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            # Parse the decision
            keep_new = decision in ["KEEP_NEW", "KEEP_BOTH"]
            
            return keep_new, prompt_tokens, completion_tokens, processing_time
            
        except Exception as e:
            logger.warning(f"Conflict resolution failed. Error: {e}")
            # Default to keeping existing knowledge when uncertain
            return False, 0, 0, time.time() - start_time
