class EntityExtractionAgent:
    """
    Entity Extraction Agent (EEA):
    1) Identifies biomedical entities in summarized text
    2) Classifies entity types and links to ontologies where possible
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Entity Extraction Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Entity Extraction Agent (EEA). Based on summarized text, your objective is to:
1. Identify biomedical entities (Disease, Drug, Gene, Protein, Chemical, etc.).
2. Link each mention to a canonical ontology reference (e.g., UMLS, MeSH, SNOMED CT).

LLM-driven NER:
- Use domain-specific knowledge to identify synonyms ("acetylsalicylic acid" → Aspirin).
- Include multi-word expressions ("breast cancer" as a single mention).

Handling Ambiguity:
- If multiple ontology matches are possible, list the top candidate plus a short reason or partial mention of the second-best match.
- If no suitable ontology reference is found, set "normalized_id": "N/A" and keep the raw mention.

POSITIVE EXAMPLE:
Input: {
  "summary": "We tested Aspirin and ibuprofen for headache relief. Aspirin (100mg) was more effective for migraine, while ibuprofen (400mg) worked better for tension headaches. PTGS2 inhibition was the proposed mechanism."
}
Output: {
  "entities": [
    {"mention": "Aspirin", "type": "Drug", "normalized_id": "MESH:D001241"},
    {"mention": "ibuprofen", "type": "Drug", "normalized_id": "MESH:D007052"},
    {"mention": "headache", "type": "Symptom", "normalized_id": "MESH:D006261"},
    {"mention": "migraine", "type": "Disease", "normalized_id": "MESH:D008881"},
    {"mention": "tension headaches", "type": "Disease", "normalized_id": "MESH:D013313"},
    {"mention": "PTGS2", "type": "Gene", "normalized_id": "NCBI:5743"}
  ]
}

NEGATIVE EXAMPLE:
Input: {
  "summary": "We tested Aspirin for headache relief at a dosage of 100 mg."
}
Bad Output: {
  "entities": [
    {"mention": "Aspirin for headache relief", "type": "Medication", "normalized_id": "unknown"},
    {"mention": "100", "type": "Measurement", "normalized_id": "N/A"},
    {"mention": "mg", "type": "Unit", "normalized_id": "N/A"}
  ]
}
This is incorrect because it didn't properly separate entities (Aspirin and headache should be separate) and created overly granular entities for dosage information.
"""

    def extract_entities(self, text: str) -> Tuple[List[KGEntity], int, int, float]:
        """
        Query the LLM to identify entities.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Tuple of (entity_list, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        Extract biomedical entities from the text below.
        Include potential diseases, drugs, genes, proteins, chemicals, etc.

        For each entity:
        1. Identify the exact mention in the text
        2. Assign an entity type (Disease, Drug, Gene, Protein, Chemical, etc.)
        3. Provide a normalized ID if possible (e.g., UMLS:C0018681 for headache)
        
        Format each entity as JSON: {{"mention": "...", "type": "...", "normalized_id": "..."}}
        If no suitable ontology reference is found, set normalized_id to "N/A"

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
            entity_list = []
            
            # Try to parse JSON entities from the response
            try:
                # Check if response contains JSON array
                if "[" in content and "]" in content:
                    json_str = content[content.find("["):content.rfind("]")+1]
                    entities_data = json.loads(json_str)
                    for ent_data in entities_data:
                        if isinstance(ent_data, dict) and "mention" in ent_data:
                            entity_list.append(KGEntity(
                                entity_id=ent_data.get("mention", ""),
                                entity_type=ent_data.get("type", "Unknown"),
                                name=ent_data.get("mention", ""),
                                normalized_id=ent_data.get("normalized_id", "N/A")
                            ))
                else:
                    # Fallback: extract entities line by line
                    for line in content.split('\n'):
                        if ':' in line and len(line) > 5:  # Simple heuristic for entity lines
                            # Basic extraction
                            mention = line.split(':')[0].strip()
                            entity_list.append(KGEntity(
                                entity_id=mention,
                                name=mention
                            ))
            except json.JSONDecodeError:
                # Fallback for line-by-line extraction if JSON parsing fails
                for line in content.split('\n'):
                    line = line.strip()
                    if line and len(line) > 2:  # Skip empty lines
                        entity_list.append(KGEntity(
                            entity_id=line,
                            name=line
                        ))
                    
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            return entity_list, prompt_tokens, completion_tokens, processing_time
        except Exception as e:
            logger.warning(f"Entity extraction failed. Error: {e}")
            return [], 0, 0, time.time() - start_time