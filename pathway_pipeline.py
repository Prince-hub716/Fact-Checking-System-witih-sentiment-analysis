"""
Pathway-based real-time streaming data processing pipeline for fact-checking claims.
This module handles the core Pathway functionality for processing incoming claims
and preparing them for fact verification through RAG-based retrieval.
"""

import pathway as pw
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PathwayPipeline:
    """
    Main Pathway pipeline for processing fact-checking claims in real-time.
    Handles text preprocessing, embedding generation, and similarity searches.
    """
    
    def __init__(self):
        """Initialize the Pathway pipeline with necessary components."""
        self.setup_pipeline()
        self.fact_database = self._load_fact_database()
        
    def setup_pipeline(self):
        """
        Set up the Pathway streaming pipeline for claim processing.
        This creates the necessary transformations and data flows.
        """
        logger.info("Setting up Pathway pipeline...")
        
        # Initialize pipeline components
        # Create a simple schema class for demonstration
        # In production, use proper Pathway schema definition
        self.claim_schema = {
            'claim_id': str,
            'text': str,
            'timestamp': float,
            'processed': bool
        }
        
        logger.info("Pathway pipeline setup complete")
    
    def _load_fact_database(self) -> List[Dict[str, Any]]:
        """
        Load the fact-checking source database.
        In a production system, this would connect to a real fact-checking API.
        """
        try:
            with open('data/fact_check_sources.json', 'r') as f:
                data = json.load(f)
                # Extract individual claims from the source data structure
                facts = []
                for source in data.get('sources', []):
                    for claim_data in source.get('verification_claims', []):
                        fact = {
                            'id': f"{source['id']}_{claim_data.get('claim', '')[:20]}",
                            'claim': claim_data.get('claim', ''),
                            'status': claim_data.get('status', 'unverified'),
                            'confidence': claim_data.get('confidence', 0.0),
                            'evidence': claim_data.get('evidence', ''),
                            'source_title': source.get('title', ''),
                            'source_url': source.get('url', ''),
                            'credibility': source.get('credibility_score', 0.0)
                        }
                        facts.append(fact)
                return facts
        except FileNotFoundError:
            logger.warning("Fact database not found, using fallback data")
            return self._create_fallback_database()
    
    def _create_fallback_database(self) -> List[Dict[str, Any]]:
        """Create a fallback fact database if the main file is not available."""
        return [
            {
                "id": "fact_001",
                "claim": "Vaccines are safe and effective",
                "status": "verified",
                "sources": ["WHO", "CDC", "FDA"],
                "confidence": 0.95,
                "summary": "Multiple health organizations confirm vaccine safety and efficacy"
            },
            {
                "id": "fact_002", 
                "claim": "Climate change is primarily caused by human activities",
                "status": "verified",
                "sources": ["IPCC", "NASA", "NOAA"],
                "confidence": 0.98,
                "summary": "Scientific consensus confirms human-caused climate change"
            },
            {
                "id": "fact_003",
                "claim": "5G networks cause coronavirus",
                "status": "false",
                "sources": ["WHO", "IEEE", "FCC"],
                "confidence": 0.99,
                "summary": "No scientific evidence supports connection between 5G and coronavirus"
            }
        ]
    
    def process_claim(self, claim_text: str) -> Dict[str, Any]:
        """
        Process a single claim through the Pathway pipeline.
        
        Args:
            claim_text: The claim text to process
            
        Returns:
            Dictionary containing processing results and metadata
        """
        logger.info(f"Processing claim: {claim_text[:50]}...")
        
        # Generate unique claim ID
        claim_id = f"claim_{datetime.now().timestamp()}"
        
        # Preprocess the claim text
        processed_text = self._preprocess_text(claim_text)
        
        # Extract key claims from the text
        key_claims = self._extract_key_claims(processed_text)
        
        # Generate embeddings for similarity search
        embeddings = self._generate_embeddings(key_claims)
        
        # Perform similarity search against fact database
        similar_facts = self._similarity_search(embeddings, top_k=5)
        
        # Compile processing results
        processing_result = {
            "claim_id": claim_id,
            "original_text": claim_text,
            "processed_text": processed_text,
            "key_claims": key_claims,
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "similar_facts": similar_facts,
            "processing_timestamp": datetime.now().isoformat(),
            "pathway_version": "streaming_v1.0"
        }
        
        logger.info(f"Claim processing complete: {claim_id}")
        return processing_result
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess claim text for better analysis.
        
        Args:
            text: Raw claim text
            
        Returns:
            Preprocessed text
        """
        # Basic text cleaning
        processed = text.strip().lower()
        
        # Remove extra whitespace
        processed = ' '.join(processed.split())
        
        # Basic normalization (could be expanded with more sophisticated NLP)
        replacements = {
            "don't": "do not",
            "won't": "will not", 
            "can't": "cannot",
            "isn't": "is not",
            "aren't": "are not"
        }
        
        for old, new in replacements.items():
            processed = processed.replace(old, new)
        
        return processed
    
    def _extract_key_claims(self, text: str) -> List[str]:
        """
        Extract key factual claims from the processed text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of extracted key claims
        """
        # Simple sentence splitting for key claim extraction
        # In production, this would use more sophisticated NLP
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Filter for sentences that likely contain factual claims
        key_claims = []
        claim_indicators = [
            'is', 'are', 'causes', 'prevents', 'increases', 'decreases',
            'shows', 'proves', 'demonstrates', 'according to', 'studies show'
        ]
        
        for sentence in sentences:
            if any(indicator in sentence for indicator in claim_indicators):
                key_claims.append(sentence)
        
        # If no key claims found, use the full text
        if not key_claims:
            key_claims = [text]
            
        return key_claims[:3]  # Limit to top 3 claims
    
    def _generate_embeddings(self, claims: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for the claims.
        
        Args:
            claims: List of claim texts
            
        Returns:
            List of embedding vectors
        """
        # Simulate embedding generation
        # In production, this would use OpenAI embeddings or similar
        embeddings = []
        
        for claim in claims:
            # Create a pseudo-embedding based on text characteristics
            # This is a simplified version for demonstration
            embedding = self._create_pseudo_embedding(claim)
            embeddings.append(embedding)
        
        return embeddings
    
    def _create_pseudo_embedding(self, text: str) -> List[float]:
        """
        Create a pseudo-embedding for demonstration purposes.
        In production, use proper embedding models.
        """
        # Create a deterministic embedding based on text features
        words = text.split()
        
        # Simple features: length, word count, character distribution
        features = [
            len(text) / 100.0,  # Normalized length
            len(words) / 20.0,  # Normalized word count
            text.count('a') / len(text) if text else 0,
            text.count('e') / len(text) if text else 0,
            text.count('i') / len(text) if text else 0,
            text.count('o') / len(text) if text else 0,
            text.count('u') / len(text) if text else 0,
        ]
        
        # Pad to 384 dimensions (common embedding size)
        while len(features) < 384:
            features.append(0.0)
        
        return features[:384]
    
    def _similarity_search(self, query_embeddings: List[List[float]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search against the fact database.
        
        Args:
            query_embeddings: Query embeddings to search for
            top_k: Number of top results to return
            
        Returns:
            List of similar facts with similarity scores
        """
        similar_facts = []
        
        for fact in self.fact_database:
            # Generate embedding for the fact
            claim_text = fact.get('claim', '') if isinstance(fact, dict) else str(fact)
            fact_embedding = self._create_pseudo_embedding(claim_text)
            
            # Calculate similarity with each query embedding
            max_similarity = 0.0
            for query_embedding in query_embeddings:
                similarity = self._cosine_similarity(query_embedding, fact_embedding)
                max_similarity = max(max_similarity, similarity)
            
            # Add to results if above threshold
            if max_similarity > 0.1:  # Minimum similarity threshold
                fact_copy = fact.copy()
                fact_copy['similarity_score'] = max_similarity
                similar_facts.append(fact_copy)
        
        # Sort by similarity score and return top_k
        similar_facts.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_facts[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Convert to numpy arrays for calculation
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    def create_streaming_table(self, input_connector: Any) -> Any:
        """
        Create a Pathway streaming table for real-time claim processing.
        
        Args:
            input_connector: Pathway input connector
            
        Returns:
            Configured Pathway table
        """
        # For demonstration purposes, return a simple structure
        # In production, this would create actual Pathway streaming tables
        return {
            'schema': self.claim_schema,
            'connector': input_connector,
            'status': 'configured'
        }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get current pipeline statistics and health metrics.
        
        Returns:
            Dictionary containing pipeline statistics
        """
        return {
            "fact_database_size": len(self.fact_database),
            "pipeline_status": "active",
            "last_update": datetime.now().isoformat(),
            "version": "1.0.0"
        }
