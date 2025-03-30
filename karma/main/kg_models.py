##############################################################################
# Data Structures
##############################################################################

@dataclass
class KnowledgeTriple:
    """
    Data class representing a single knowledge triple in the biomedical domain.
    
    Attributes:
        head: The subject entity
        relation: The relationship type
        tail: The object entity
        confidence: Model confidence score [0-1]
        source: Origin of the triple
        relevance: Domain relevance score [0-1]
        clarity: Linguistic clarity score [0-1]
    """
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    source: str = "unknown"
    relevance: float = 0.0
    clarity: float = 0.0
    
    def __str__(self) -> str:
        """String representation of the knowledge triple."""
        return f"({self.head}) -[{self.relation}]-> ({self.tail})"

@dataclass
class KGEntity:
    """
    Data class representing a canonical entity in the knowledge graph.
    
    Attributes:
        entity_id: Unique identifier
        entity_type: Semantic type (e.g. Drug, Disease)
        name: Display name
        normalized_id: Reference to standard ontology (e.g., UMLS:C0004238)
    """
    entity_id: str
    entity_type: str = "Unknown"
    name: str = ""
    normalized_id: str = "N/A"
    
    def __str__(self) -> str:
        """String representation of the entity."""
        return f"{self.name} ({self.entity_type})"

@dataclass
class IntermediateOutput:
    """
    Data class for storing intermediate outputs from each agent.
    
    Tracks the full pipeline state including raw inputs, 
    intermediate results, and final outputs.
    """
    raw_text: str = ""
    segments: List[Dict] = field(default_factory=list)
    relevant_segments: List[Dict] = field(default_factory=list)
    summaries: List[Dict] = field(default_factory=list)
    entities: List[KGEntity] = field(default_factory=list)
    relationships: List[KnowledgeTriple] = field(default_factory=list)
    aligned_entities: List[KGEntity] = field(default_factory=list)
    aligned_triples: List[KnowledgeTriple] = field(default_factory=list)
    final_triples: List[KnowledgeTriple] = field(default_factory=list)
    integrated_triples: List[KnowledgeTriple] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, list) and value and hasattr(value[0], '__dict__'):
                result[key] = [item.__dict__ for item in value]
            else:
                result[key] = value
        return result