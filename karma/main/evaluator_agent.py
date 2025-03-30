class EvaluatorAgent:
    """
    Evaluator Agent (EA):
    1) Aggregates various confidence signals
    2) Produces a final integration score for each triple
    3) Evaluates clarity and relevance in addition to confidence
    """
    def __init__(self, client: OpenAI, model_name: str, integrate_threshold=0.6):
        """
        Initialize the Evaluator Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
            integrate_threshold: Minimum score to integrate knowledge
        """
        self.client = client
        self.model_name = model_name
        self.integrate_threshold = integrate_threshold
        self.system_prompt = """You are the Evaluator Agent (EA). After extraction, alignment, and conflict resolution, each candidate triplet has multiple verification scores. Your duty is to aggregate these signals into final confidence, clarity, and relevance scores and decide whether to integrate each triplet into the KG.

For CONFIDENCE evaluation:
- Assess the factual correctness of the triple based on biomedical knowledge
- Consider the source reliability and extraction confidence
- Account for any conflict resolution outcomes

For CLARITY evaluation:
- Determine how unambiguous and well-defined the entities and relation are
- Check for vague terms or imprecise relationship descriptions
- Assess whether the triple would be interpretable to domain experts

For RELEVANCE evaluation:
- Evaluate how important and appropriate the triple is for the knowledge graph
- Consider whether it aligns with the domain focus
- Assess its potential utility for downstream applications

POSITIVE EXAMPLE:
Input: {
  "triple": {"head": "Metformin", "relation": "decreases", "tail": "blood glucose levels"}
}
Output: {
  "confidence": 0.95,
  "clarity": 0.90,
  "relevance": 0.85,
  "rationale": "Well-established mechanism of action for this first-line antidiabetic drug. The entities and relationship are clearly defined. Highly relevant for a biomedical knowledge graph."
}

NEGATIVE EXAMPLE:
Input: {
  "triple": {"head": "Drug X", "relation": "may influence", "tail": "some cellular processes"}
}
Output: {
  "confidence": 0.85,
  "clarity": 0.40,
  "relevance": 0.30,
  "rationale": "The relationship is vaguely defined with uncertain terms. The entities lack specificity. Limited utility in a biomedical knowledge graph."
}
"""

    def finalize_triples(self, candidate_triples: List[KnowledgeTriple]) -> Tuple[List[KnowledgeTriple], int, int, float]:
        """
        Evaluate and filter triples based on confidence, clarity, and relevance.
        
        Args:
            candidate_triples: Triples to evaluate
            
        Returns:
            Tuple of (integrated_triples, prompt_tokens, completion_tokens, processing_time)
        """
        start_time = time.time()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        integrated_triples = []
        for i, triple in enumerate(candidate_triples):
            # Evaluate confidence, clarity, and relevance if missing
            if triple.confidence < 0.01:
                conf, conf_pt, conf_ct, _ = self._evaluate_confidence(triple)
                triple.confidence = conf
                total_prompt_tokens += conf_pt
                total_completion_tokens += conf_ct
                
            if triple.clarity < 0.01:
                clarity, clar_pt, clar_ct, _ = self._evaluate_clarity(triple)
                triple.clarity = clarity
                total_prompt_tokens += clar_pt
                total_completion_tokens += clar_ct
                
            if triple.relevance < 0.01:
                relevance, rel_pt, rel_ct, _ = self._evaluate_relevance(triple)
                triple.relevance = relevance
                total_prompt_tokens += rel_pt
                total_completion_tokens += rel_ct
            
            # Compute final integration score
            integration_score = self._aggregate_scores(triple)
            
            # Keep triple if score meets threshold
            if integration_score >= self.integrate_threshold:
                integrated_triples.append(triple)
        
        processing_time = time.time() - start_time
        return integrated_triples, total_prompt_tokens, total_completion_tokens, processing_time

    def _aggregate_scores(self, triple: KnowledgeTriple) -> float:
        """
        Combine confidence metrics into final score.
        
        Args:
            triple: Triple to score
            
        Returns:
            Aggregated confidence score
        """
        # Weighting factors for different metrics (customizable)
        w_conf, w_clarity, w_rel = 0.5, 0.25, 0.25
        
        # Ensure all scores are in [0.0, 1.0] range
        conf = max(0.0, min(1.0, triple.confidence))
        clarity = max(0.0, min(1.0, triple.clarity))
        relevance = max(0.0, min(1.0, triple.relevance))
        
        # Weighted average
        return (
            w_conf * conf +
            w_clarity * clarity +
            w_rel * relevance
        )

    def _evaluate_confidence(self, triple: KnowledgeTriple) -> Tuple[float, int, int, float]:
        """
        LLM-based evaluation of triple confidence.
        
        Args:
            triple: Triple to evaluate
            
        Returns:
            Tuple of (confidence_score, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        Evaluate the factual confidence of this biomedical statement:
        "{triple.head} -> {triple.relation} -> {triple.tail}"

        I need you to assess how confident we can be that this relationship is scientifically accurate based on established biomedical knowledge.
        
        Consider factors like:
        1. Is this a well-established scientific fact?
        2. Is there substantial evidence in the literature supporting this claim?
        3. Would biomedical experts broadly agree with this statement?
        4. Does it align with current scientific understanding?
        
        Rate your confidence from 0.0 (completely uncertain) to 1.0 (extremely confident).
        Return only a single float value between 0.0 and 1.0.
        
        Example ratings:
        - 0.95-0.99: Well-established scientific facts with overwhelming evidence
        - 0.80-0.94: Strong scientific consensus with substantial supporting evidence
        - 0.60-0.79: Reasonably supported claims with some evidence base
        - 0.40-0.59: Mixed evidence or emerging hypotheses
        - 0.20-0.39: Limited evidence, preliminary findings
        - 0.01-0.19: Very weak evidence, highly speculative

        POSITIVE EXAMPLES:
        - "Metformin -> decreases -> blood glucose levels" = 0.97
        - "Statins -> inhibit -> HMG-CoA reductase" = 0.95
        - "Insulin resistance -> contributes to -> type 2 diabetes" = 0.93

        NEGATIVE EXAMPLES:
        - "Vitamin C -> cures -> cancer" = 0.12
        - "Unknown compound -> may affect -> some pathways" = 0.25
        
        Return only the numeric value as a float between 0.0 and 1.0.
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
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            # Extract float value
            score = self._extract_float_score(content)
            return score, prompt_tokens, completion_tokens, processing_time
            
        except Exception as e:
            logger.warning(f"Confidence evaluation failed. Error: {e}")
            return 0.5, 0, 0, time.time() - start_time

    def _evaluate_clarity(self, triple: KnowledgeTriple) -> Tuple[float, int, int, float]:
        """
        LLM-based evaluation of triple linguistic clarity.
        
        Args:
            triple: Triple to evaluate
            
        Returns:
            Tuple of (clarity_score, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        Evaluate the clarity and specificity of this biomedical relationship:
        "{triple.head} -> {triple.relation} -> {triple.tail}"

        Assess how clear, specific, and unambiguous this statement is to biomedical experts.
        
        Consider these aspects:
        1. Are the entities precise and unambiguous? (e.g., "ACE inhibitors" is clearer than "some drugs")
        2. Is the relationship type specific and well-defined? (e.g., "inhibits" is clearer than "affects")
        3. Would biomedical experts interpret this statement consistently?
        4. Are there any vague terms or imprecise language that reduce clarity?
        5. Does the statement avoid unnecessary hedging words (e.g., "may", "possibly", "might")?
        
        Rate clarity from 0.0 (very ambiguous) to 1.0 (perfectly clear).
        Return only a single float value between 0.0 and 1.0.
        
        Example ratings:
        - 0.95-0.99: Crystal clear, highly specific statements with no ambiguity
        - 0.80-0.94: Very clear statements with minor room for interpretation
        - 0.60-0.79: Mostly clear statements with some potential ambiguity
        - 0.40-0.59: Statements with moderate ambiguity or vagueness
        - 0.20-0.39: Significantly vague or ambiguous statements
        - 0.01-0.19: Extremely vague statements with minimal specificity

        POSITIVE EXAMPLES:
        - "Atorvastatin -> inhibits -> HMG-CoA reductase" = 0.95
        - "Aspirin -> irreversibly inhibits -> cyclooxygenase-1" = 0.97
        - "TNF-alpha -> induces -> apoptosis in tumor cells" = 0.85

        NEGATIVE EXAMPLES:
        - "Some medicines -> may affect -> various biological processes" = 0.20
        - "Drug X -> possibly influences -> certain cellular pathways" = 0.35
        
        Return only the numeric value as a float between 0.0 and 1.0.
        """
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You evaluate clarity of biomedical relationships."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            # Extract float value
            score = self._extract_float_score(content)
            return score, prompt_tokens, completion_tokens, processing_time
            
        except Exception as e:
            logger.warning(f"Clarity evaluation failed. Error: {e}")
            return 0.5, 0, 0, time.time() - start_time

    def _evaluate_relevance(self, triple: KnowledgeTriple, domain="biomedical") -> Tuple[float, int, int, float]:
        """
        LLM-based evaluation of triple domain relevance.
        
        Args:
            triple: Triple to evaluate
            domain: Domain topic for relevance
            
        Returns:
            Tuple of (relevance_score, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        Evaluate how relevant this relationship is to the {domain} domain:
        "{triple.head} -> {triple.relation} -> {triple.tail}"

        Assess how important and appropriate this statement is for inclusion in a specialized {domain} knowledge graph.
        
        Consider these aspects:
        1. Is this directly relevant to {domain} research or practice?
        2. Does this provide valuable information to {domain} experts?
        3. Would this knowledge be useful for {domain} applications (research, clinical practice, drug discovery, etc.)?
        4. Is this specialized knowledge rather than general world knowledge?
        5. Does it align with the core interests of the {domain} field?
        
        Rate relevance from 0.0 (completely irrelevant) to 1.0 (highly relevant).
        Return only a single float value between 0.0 and 1.0.
        
        Example ratings:
        - 0.95-0.99: Core {domain} knowledge essential to the field
        - 0.80-0.94: Highly relevant information for {domain} specialists
        - 0.60-0.79: Moderately relevant information with clear applications
        - 0.40-0.59: Somewhat relevant but not central to the domain
        - 0.20-0.39: Marginally relevant information
        - 0.01-0.19: Largely irrelevant to the {domain} domain

        POSITIVE EXAMPLES:
        - "Metformin -> decreases -> insulin resistance" = 0.95
        - "BRCA1 mutation -> increases risk of -> breast cancer" = 0.97
        - "Tumor necrosis factor -> stimulates -> inflammatory response" = 0.90

        NEGATIVE EXAMPLES:
        - "William Shakespeare -> wrote -> Hamlet" = 0.05 (literature knowledge, not biomedical)
        - "Earth -> orbits -> Sun" = 0.01 (general knowledge, not specialized)
        - "Company X -> manufactures -> medical devices" = 0.30 (business information, not core biomedical knowledge)
        
        Return only the numeric value as a float between 0.0 and 1.0.
        """
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You evaluate {domain} relevance of relationships."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            # Extract float value
            score = self._extract_float_score(content)
            return score, prompt_tokens, completion_tokens, processing_time
            
        except Exception as e:
            logger.warning(f"Relevance evaluation failed. Error: {e}")
            return 0.5, 0, 0, time.time() - start_time

    def _extract_float_score(self, content: str) -> float:
        """
        Extract a float value from LLM response.
        
        Args:
            content: Text containing float value(s)
            
        Returns:
            Parsed and normalized float value in [0.0, 1.0]
        """
        # Look for float numbers in the content
        import re
        float_matches = re.findall(r"[-+]?\d*\.\d+|\d+", content)
        if float_matches:
            try:
                score = float(float_matches[0])
                # Clamp to [0.0, 1.0]
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        # Default fallback
        return 0.5
