    ##########################################################################
    # Utility Methods
    ##########################################################################
    def export_knowledge_graph(self, output_path: str = None) -> Dict:
        """
        Export the knowledge graph as a dictionary or save to a file.
        
        Args:
            output_path: Optional file path to save the knowledge graph
            
        Returns:
            Dictionary representation of the knowledge graph
        """
        # Convert the KG to a serializable format
        kg_export = {
            "entities": list(self.knowledge_graph["entities"]),
            "triples": [asdict(triple) for triple in self.knowledge_graph["triples"]]
        }
        
        # Save to file if path is provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(kg_export, f, indent=2)
                
        return kg_export
    
    def save_intermediate_results(self, output_path: str):
        """
        Save all intermediate results from the pipeline for analysis.
        
        Args:
            output_path: Path to save the results
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.intermediate.to_dict(), f, indent=2)
            logger.info(f"Intermediate results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {str(e)}")
    
    def clear_knowledge_graph(self):
        """Reset the knowledge graph to empty state."""
        self.knowledge_graph = {
            "entities": set(),
            "triples": []
        }
        logger.info("Knowledge graph cleared")

    def print_statistics(self):
        """Print statistics about the current knowledge graph."""
        entity_count = len(self.knowledge_graph["entities"])
        triple_count = len(self.knowledge_graph["triples"])
        
        # Calculate types distribution
        entity_types = {}
        relation_types = {}
        
        for triple in self.knowledge_graph["triples"]:
            relation_types[triple.relation] = relation_types.get(triple.relation, 0) + 1
        
        # Print summary
        print(f"Knowledge Graph Statistics:")
        print(f"  - Entities: {entity_count}")
        print(f"  - Relationships: {triple_count}")
        print(f"  - Unique relation types: {len(relation_types)}")
        
        if relation_types:
            print("\nTop relation types:")
            for rel, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {rel}: {count}")
