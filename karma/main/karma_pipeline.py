    ##########################################################################
    # The Main Pipeline
    ##########################################################################
    def process_document(self, source: Union[str, os.PathLike], domain: str = "biomedical") -> List[KnowledgeTriple]:
        """
        Pipeline entry point to process a document and extract knowledge.
        
        If `source` ends with '.pdf', we assume it is a PDF file path;
        otherwise, treat `source` as raw text input.
        
        Args:
            source: Text content or path to PDF file
            domain: Domain context for relevance scoring
            
        Returns:
            List of integrated KnowledgeTriple objects
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_time = 0
        pipeline_start_time = time.time()
        
        # Handle various input types
        if isinstance(source, str) and source.lower().endswith('.pdf'):
            raw_text = self._read_pdf(source)
        elif isinstance(source, os.PathLike) and str(source).lower().endswith('.pdf'):
            raw_text = self._read_pdf(str(source))
        else:
            # treat source directly as text
            raw_text = source
            
        self.intermediate.raw_text = raw_text
        
        # === (1) Ingestion ===
        step_start = time.time()
        doc_dict = self.ingestion_agent.ingest_document(raw_text)
        step_time = time.time() - step_start
        total_time += step_time
        self._log(f"[1] Ingestion completed in {step_time:.2f}s. Document standardized.")

        # === (2) Reader: Segment + Score Relevance ===
        step_start = time.time()
        segments = self.reader_agent.split_into_segments(doc_dict["content"])
        self.intermediate.segments = segments
        
        relevant_content = []
        for seg in segments:
            score, prompt_tokens, completion_tokens, processing_time = self.reader_agent.score_relevance(seg)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_time += processing_time
            if score > 0.2:  # Keep only segments with relevance above threshold
                relevant_content.append(seg)
        
        self.intermediate.relevant_segments = relevant_content
        step_time = time.time() - step_start
        self._log(f"[2] Reader completed in {step_time:.2f}s. Total segments: {len(segments)}, relevant: {len(relevant_content)}")

        # === (3) Summarizer ===
        step_start = time.time()
        summaries = []
        for seg in relevant_content:
            summary, prompt_tokens, completion_tokens, processing_time = self.summarizer_agent.summarize_segment(seg)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_time += processing_time
            summaries.append(summary)
        
        self.intermediate.summaries = summaries
        step_time = time.time() - step_start
        self._log(f"[3] Summarizer completed in {step_time:.2f}s. Summaries produced: {len(summaries)}")

        # === (4) Entity Extraction ===
        step_start = time.time()
        all_entities: List[KGEntity] = []
        for summary in summaries:
            extracted, prompt_tokens, completion_tokens, processing_time = self.entity_ex_agent.extract_entities(summary)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_time += processing_time
            all_entities.extend(extracted)
        
        # Deduplicate entities by name (case-insensitive)
        unique_entities_map = {}
        for ent in all_entities:
            unique_entities_map[ent.name.lower()] = ent
        all_entities = list(unique_entities_map.values())
        
        self.intermediate.entities = all_entities
        step_time = time.time() - step_start
        self._log(f"[4] Entity Extraction completed in {step_time:.2f}s. Unique entities found: {len(all_entities)}")

        # === (5) Relationship Extraction ===
        step_start = time.time()
        all_relationships: List[KnowledgeTriple] = []
        for summary in summaries:
            new_trips, prompt_tokens, completion_tokens, processing_time = self.relation_ex_agent.extract_relationships(summary, all_entities)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_time += processing_time
            all_relationships.extend(new_trips)
            
        # Deduplicate relationships (exact matches only)
        unique_rel_map = {}
        for rel in all_relationships:
            key = f"{rel.head.lower()}__{rel.relation.lower()}__{rel.tail.lower()}"
            # Keep the one with higher confidence if duplicates exist
            if key not in unique_rel_map or rel.confidence > unique_rel_map[key].confidence:
                unique_rel_map[key] = rel
        all_relationships = list(unique_rel_map.values())
        
        self.intermediate.relationships = all_relationships
        step_time = time.time() - step_start
        self._log(f"[5] Relationship Extraction completed in {step_time:.2f}s. Relationships found: {len(all_relationships)}")

        # === (6) Schema Alignment ===
        step_start = time.time()
        # Align entities to standard types (e.g., Drug, Disease, Gene)
        aligned_entities, prompt_tokens, completion_tokens, processing_time = self.schema_agent.align_entities(all_entities)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_time += processing_time
        
        # Align relationships to standard forms
        aligned_triples = self.schema_agent.align_relationships(all_relationships)
        
        self.intermediate.aligned_entities = aligned_entities
        self.intermediate.aligned_triples = aligned_triples
        step_time = time.time() - step_start
        self._log(f"[6] Schema Alignment completed in {step_time:.2f}s. Entities and relationships aligned to schema.")

        # === (7) Conflict Resolution ===
        step_start = time.time()
        # Check new triples against existing knowledge graph for contradictions
        non_conflicting_triples, prompt_tokens, completion_tokens, processing_time = self.conflict_agent.resolve_conflicts(
            aligned_triples, self.knowledge_graph["triples"]
        )
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_time += processing_time
        
        step_time = time.time() - step_start
        self._log(f"[7] Conflict Resolution completed in {step_time:.2f}s. Non-conflicting triples: {len(non_conflicting_triples)}/{len(aligned_triples)}")

        # === (8) Evaluation ===
        step_start = time.time()
        # Score and filter triples by confidence, clarity, and relevance
        integrated_triples, prompt_tokens, completion_tokens, processing_time = self.evaluator_agent.finalize_triples(non_conflicting_triples)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_time += processing_time
        
        self.intermediate.integrated_triples = integrated_triples
        step_time = time.time() - step_start
        self._log(f"[8] Evaluation completed in {step_time:.2f}s. Final integrated triples: {len(integrated_triples)}/{len(non_conflicting_triples)}")

        # === Integration into KG ===
        # Add new entities to the knowledge graph
        for entity in aligned_entities:
            self.knowledge_graph["entities"].add(entity.name)
            
        # Add new triples to the knowledge graph
        for triple in integrated_triples:
            self.knowledge_graph["triples"].append(triple)
            
        # Update tracking metrics
        self.intermediate.prompt_tokens = total_prompt_tokens
        self.intermediate.completion_tokens = total_completion_tokens
        self.intermediate.processing_time = total_time
        
        total_pipeline_time = time.time() - pipeline_start_time
        self._log(f"KARMA pipeline completed in {total_pipeline_time:.2f}s. Added {len(integrated_triples)} new knowledge triples.")
        
        return integrated_triples