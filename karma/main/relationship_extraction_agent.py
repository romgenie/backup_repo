class RelationshipExtractionAgent:
    """
    Relationship Extraction Agent (REA):
    1) Identifies relationships between extracted entities
    2) Classifies relationship types
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Relationship Extraction Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Relationship Extraction Agent (REA). Given a text snippet plus a set of recognized entities, your mission is to detect possible relationships (e.g., treats, causes, interactsWith, inhibits).

LLM-based Relation Classification:
- Consider grammar structures (e.g., "X was observed to inhibit Y") and domain patterns ("X reduces expression of Y").
- Allow multiple relationship candidates if the text is ambiguous or suggests multiple interactions.

Negative Relation Handling:
- If the text says "Aspirin does not treat migraine," the relationship (Aspirin, treats, migraine) is negative. Output no relationship in such cases.
- Recognize negation cues ("no effect", "absence of association").

POSITIVE EXAMPLE:
Input: {
  "summary": "Aspirin was shown to reduce headaches by inhibiting prostaglandin synthesis. It has no effect on hypertension.",
  "entities": [
    {"mention": "Aspirin", "type": "Drug", "normalized_id": "MESH:D001241"},
    {"mention": "headaches", "type": "Disease", "normalized_id": "MESH:D006261"},
    {"mention": "prostaglandin", "type": "Chemical", "normalized_id": "MESH:D011453"},
    {"mention": "hypertension", "type": "Disease", "normalized_id": "MESH:D006973"}
  ]
}
Output: {
  "relationships": [
    {"head": "Aspirin", "relation": "treats", "tail": "headaches", "confidence": 0.95},
    {"head": "Aspirin", "relation": "inhibits", "tail": "prostaglandin", "confidence": 0.90}
  ]
}
Note: No relationship is extracted between Aspirin and hypertension due to the negation.

NEGATIVE EXAMPLE:
Input: {
  "summary": "Aspirin was shown to reduce headaches by inhibiting prostaglandin synthesis.",
  "entities": [
    {"mention": "Aspirin", "type": "Drug", "normalized_id": "MESH:D001241"},
    {"mention": "headaches", "type": "Disease", "normalized_id": "MESH:D006261"},
    {"mention": "prostaglandin", "type": "Chemical", "normalized_id": "MESH:D011453"}
  ]
}
Bad Output: {
  "relationships": [
    {"head": "prostaglandin", "relation": "causes", "tail": "headaches", "confidence": 0.8}
  ]
}
This is incorrect because the text doesn't explicitly state this relationship. While it might be inferred, we should only extract relationships directly supported by the text.
"""

    def extract_relationships(self, text: str, entities: List[KGEntity]) -> Tuple[List[KnowledgeTriple], int, int, float]:
        """
        Query the LLM to identify relationships among provided entities.
        
        Args:
            text: Source text
            entities: List of entities to find relationships between
            
        Returns:
            Tuple of (relationship_list, prompt_tokens, completion_tokens, processing_time)
        """
        if not entities:
            return [], 0, 0, 0.0

        # Format the entity list for the LLM
        entity_bullets = "\n".join(f"- {ent.name} (Type: {ent.entity_type})" for ent in entities)
        
        prompt = f"""
        We have these entities of interest:
        {entity_bullets}

        From the text below, identify direct relationships between these entities.
        Use standardized relationships such as 'treats', 'causes', 'inhibits', 'activates', 'interacts_with', 'associated_with', etc.
        
        For each relationship, provide:
        1. Head entity (subject)
        2. Relation type
        3. Tail entity (object)
        4. A confidence score (0-1) for how certain the relationship is expressed in the text
        
        Format as JSON: {{"head": "...", "relation": "...", "tail": "...", "confidence": 0.X}}
        
        If no relationships are found, return an empty array.

        Text:
        {text}
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
            
            content = response.choices[0].message.content.strip()
            triples: List[KnowledgeTriple] = []
            
            # Try to parse JSON from the response
            try:
                if "[" in content and "]" in content:
                    json_str = content[content.find("["):content.rfind("]")+1]
                    relations_data = json.loads(json_str)
                    
                    for rel in relations_data:
                        if isinstance(rel, dict) and "head" in rel and "relation" in rel and "tail" in rel:
                            # Basic data validation
                            head = rel.get("head", "").strip()
                            relation = rel.get("relation", "").strip()
                            tail = rel.get("tail", "").strip()
                            
                            if head and relation and tail:
                                confidence = float(rel.get("confidence", 0.5))
                                # Get clarity and relevance scores
                                clarity, clarity_tokens = self._estimate_metric(head, relation, tail, "clarity")
                                relevance, relevance_tokens = self._estimate_metric(head, relation, tail, "relevance")
                                
                                triples.append(KnowledgeTriple(
                                    head=head,
                                    relation=relation,
                                    tail=tail,
                                    confidence=confidence,
                                    clarity=clarity,
                                    relevance=relevance,
                                    source="relationship_extraction"
                                ))
                else:
                    # Fallback: extract relationships line by line
                    for line in content.split('\n'):
                        if '->' in line:
                            parts = [p.strip() for p in line.split('->')]
                            if len(parts) == 3:
                                head, relation, tail = parts
                                triples.append(KnowledgeTriple(
                                    head=head,
                                    relation=relation,
                                    tail=tail,
                                    confidence=0.5,  # Default confidence
                                    clarity=0.5,     # Default clarity
                                    relevance=0.5,   # Default relevance
                                    source="relationship_extraction"
                                ))
            except json.JSONDecodeError:
                # Simple line-by-line fallback
                for line in content.split('\n'):
                    if '->' in line:
                        parts = [p.strip() for p in line.split('->')]
                        if len(parts) == 3:
                            head, relation, tail = parts
                            triples.append(KnowledgeTriple(
                                head=head,
                                relation=relation,
                                tail=tail,
                                confidence=0.5,
                                clarity=0.5,
                                relevance=0.5,
                                source="relationship_extraction"
                            ))
                        
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens 
            processing_time = time.time() - start_time
            
            return triples, prompt_tokens, completion_tokens, processing_time
        except Exception as e:
            logger.warning(f"Relationship extraction failed. Error: {e}")
            return [], 0, 0, time.time() - start_time

    def _estimate_metric(self, head: str, relation: str, tail: str, metric_type: str) -> Tuple[float, int]:
        """
        Estimate clarity or relevance for a triple.
        
        Args:
            head: Subject entity
            relation: Relationship type
            tail: Object entity
            metric_type: Either "clarity" or "relevance"
            
        Returns:
            Tuple of (metric_score, tokens_used)
        """
        prompt = f"""
        Evaluate the {metric_type} of this relationship triple:
        "{head} -> {relation} -> {tail}"

        For clarity, consider:
        - Whether the terms are specific and unambiguous
        - Whether the relationship is well-defined
        
        For relevance, consider:
        - How important this relationship is in the biomedical domain
        - Whether it captures significant knowledge
        
        Return a single float between 0.01 and 0.99 representing the {metric_type} score.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You evaluate {metric_type} of biomedical relationships."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=4096
            )
            
            content = response.choices[0].message.content.strip()
            tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens
            
            # Extract the score
            import re
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", content)
            if matches:
                score = float(matches[0])
                # Ensure score is in range [0.01, 0.99]
                score = max(0.01, min(0.99, score))
                return score, tokens_used
            else:
                return 0.5, tokens_used  # Default
        except Exception as e:
            logger.warning(f"Failed to estimate {metric_type}. Error: {e}")
            return 0.5, 0  # Default on error
