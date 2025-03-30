class SchemaAlignmentAgent:
    """
    Schema Alignment Agent (SAA):
    1) Maps extracted entities to standard schema types
    2) Normalizes relationship labels
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Schema Alignment Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Schema Alignment Agent (SAA). Newly extracted entities or relationships may not match existing KG classes or relation types. Your job is to determine how they should map onto the existing ontology or schema.

Ontology Reference:
- For each unknown entity, propose a parent type from {Drug, Disease, Gene, Chemical, Protein, Pathway, Symptom, ...} if not in the KG.
- For each unknown relation, map it to an existing relation if semantically close. Otherwise, propose a new label.

Confidence Computation:
- Consider lexical similarity, embedding distance, or domain rules (e.g., if an entity ends with "-in" or "-ase", it might be a protein or enzyme).
- Provide a final numeric score for how certain you are of the proposed alignment.

POSITIVE EXAMPLE:
Input: {
  "unknown_entities": ["TNF-alpha", "miR-21", "PDE4", "blood-brain barrier"],
  "unknown_relations": ["overexpresses", "disrupts"]
}
Output: {
  "alignments": [
    {"id": "TNF-alpha", "proposed_type": "Protein", "status": "mapped", "confidence": 0.95},
    {"id": "miR-21", "proposed_type": "RNA", "status": "new", "confidence": 0.90},
    {"id": "PDE4", "proposed_type": "Enzyme", "status": "mapped", "confidence": 0.85},
    {"id": "blood-brain barrier", "proposed_type": "Anatomical_Structure", "status": "mapped", "confidence": 0.95}
  ],
  "new_relations": [
    {"relation": "overexpresses", "closest_match": "upregulates", "status": "mapped", "confidence": 0.85},
    {"relation": "disrupts", "closest_match": "damages", "status": "new", "confidence": 0.70}
  ]
}

NEGATIVE EXAMPLE:
Input: {
  "unknown_entities": ["TNF-alpha", "miR-21"],
  "unknown_relations": ["overexpresses"]
}
Bad Output: {
  "alignments": [
    {"id": "TNF-alpha", "proposed_type": "Unknown", "status": "unknown"},
    {"id": "miR-21", "proposed_type": "Unknown", "status": "unknown"}
  ],
  "new_relations": [
    {"relation": "overexpresses", "closest_match": "unknown", "status": "unknown"}
  ]
}
This is incorrect because the agent should use domain knowledge to propose appropriate entity types and relation mappings, rather than marking everything as unknown.
"""

    def align_entities(self, entities: List[KGEntity]) -> Tuple[List[KGEntity], int, int, float]:
        """
        Attempt to classify each entity type via LLM or external ontology.
        
        Args:
            entities: List of entities to classify
            
        Returns:
            Tuple of (aligned_entities, prompt_tokens, completion_tokens, processing_time)
        """
        start_time = time.time()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        aligned = []
        
        # Process entities in batches to be more efficient
        batch_size = 10
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            batch_results, pt, ct = self._batch_classify_entity_types(batch)
            
            total_prompt_tokens += pt
            total_completion_tokens += ct
            
            for original_entity, classification in zip(batch, batch_results):
                if classification:
                    original_entity.entity_type = classification
                aligned.append(original_entity)
            
        processing_time = time.time() - start_time
        return aligned, total_prompt_tokens, total_completion_tokens, processing_time

    def _batch_classify_entity_types(self, entities: List[KGEntity]) -> Tuple[List[str], int, int]:
        """
        Classify multiple entities at once for efficiency.
        
        Args:
            entities: List of entities to classify
            
        Returns:
            Tuple of (classifications, prompt_tokens, completion_tokens)
        """
        entity_names = [ent.name for ent in entities]
        entities_text = "\n".join([f"{i+1}. {name}" for i, name in enumerate(entity_names)])
        
        prompt = f"""
        Classify each entity by its biomedical type: Drug, Disease, Gene, Protein, Chemical, RNA, Pathway, Cell, or Other.
        
        Entities:
        {entities_text}
        
        Return classifications one per line, numbered to match the input:
        1. [Type]
        2. [Type]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            lines = response.choices[0].message.content.strip().split('\n')
            classifications = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse numbered responses
                if '.' in line:
                    parts = line.split('.', 1)
                    if len(parts) == 2 and parts[1].strip():
                        classifications.append(parts[1].strip())
                else:
                    # If response isn't numbered, just use the line
                    classifications.append(line)
            
            # Pad with "Unknown" if some entities didn't get classified
            while len(classifications) < len(entities):
                classifications.append("Unknown")
                
            # Truncate if we somehow got more classifications than entities
            classifications = classifications[:len(entities)]
            
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            return classifications, prompt_tokens, completion_tokens
        except Exception as e:
            logger.warning(f"Entity type classification failed. Error: {e}")
            return ["Unknown"] * len(entities), 0, 0

    def align_relationships(self, triples: List[KnowledgeTriple]) -> List[KnowledgeTriple]:
        """
        Unify relationship labels to a standard set.
        
        Args:
            triples: List of triples to normalize relations for
            
        Returns:
            List of triples with normalized relations
        """
        aligned_triples = []
        for t in triples:
            normalized_rel = self._normalize_relation(t.relation)
            t.relation = normalized_rel
            aligned_triples.append(t)
        return aligned_triples

    def _normalize_relation(self, relation: str) -> str:
        """
        Standardize relation labels to canonical forms.
        
        Args:
            relation: Relation label to normalize
            
        Returns:
            Normalized relation label
        """
        # Standard mapping of similar relations, just an example
        synonyms = {
            "inhibit": "inhibits",
            "inhibited": "inhibits",
            "inhibits": "inhibits",
            "treat": "treats",
            "treated": "treats",
            "treats": "treats",
            "cause": "causes",
            "caused": "causes",
            "causes": "causes",
            "activate": "activates",
            "activates": "activates",
            "regulates": "regulates",
            "regulate": "regulates",
            "regulated": "regulates",
            "associated with": "associated_with",
            "associatedwith": "associated_with",
            "associated_with": "associated_with",
            "interacts with": "interacts_with",
            "interactswith": "interacts_with",
            "interacts_with": "interacts_with",
            "binds to": "binds_to",
            "bindsto": "binds_to",
            "binds_to": "binds_to"
        }
        
        base = relation.lower().strip()
        # Return the mapped version or the original if no mapping exists
        return synonyms.get(base, base)