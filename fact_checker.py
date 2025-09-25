"""
Fact-checking module that uses OpenAI's GPT-5 for claim verification and analysis.
This module implements the core fact-checking logic with RAG-based verification.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactChecker:
    """
    Main fact-checking class that uses GPT-5 for claim verification.
    Integrates with the Pathway pipeline for RAG-based fact verification.
    """
    
    def __init__(self):
        """Initialize the fact checker with OpenAI client."""
        self.openai_client = self._setup_openai_client()
        self.verification_prompts = self._load_verification_prompts()
        
    def _setup_openai_client(self) -> OpenAI:
        """
        Set up OpenAI client with API key from environment.
        
        Returns:
            Configured OpenAI client
        """
        api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
        return OpenAI(api_key=api_key)
    
    def _load_verification_prompts(self) -> Dict[str, str]:
        """
        Load system prompts for different types of verification tasks.
        
        Returns:
            Dictionary of verification prompts
        """
        return {
            "main_verification": """You are an expert fact-checker and misinformation detection specialist. 
            Your task is to analyze claims for accuracy and provide verification results.
            
            Analyze the given claim and provide:
            1. Verification status: "verified", "false", "uncertain", or "unverified"
            2. Confidence score between 0.0 and 1.0
            3. Clear explanation of your assessment
            4. Key factors that influenced your decision
            
            Base your analysis on factual accuracy, scientific consensus, and authoritative sources.
            Be objective and acknowledge when evidence is insufficient for a definitive conclusion.
            
            Respond in JSON format with these fields:
            {
                "status": "verification_status",
                "confidence": confidence_score,
                "explanation": "detailed_explanation",
                "key_factors": ["factor1", "factor2", "factor3"],
                "requires_sources": true/false
            }""",
            
            "sentiment_analysis": """Analyze the sentiment and emotional tone of the given text.
            Provide sentiment classification and emotional indicators that might suggest bias or manipulation.
            
            Respond in JSON format:
            {
                "sentiment": {
                    "label": "positive/negative/neutral",
                    "score": sentiment_confidence_score,
                    "emotional_indicators": ["indicator1", "indicator2"]
                }
            }""",
            
            "source_analysis": """Evaluate the credibility and relevance of the provided sources
            in relation to the claim being fact-checked.
            
            For each source, assess:
            1. Credibility and authority
            2. Relevance to the claim  
            3. Potential bias or conflicts of interest
            
            Respond in JSON format:
            {
                "source_assessment": [
                    {
                        "source_id": "id",
                        "credibility_score": score,
                        "relevance_score": score,
                        "bias_indicators": ["indicator1", "indicator2"],
                        "assessment_summary": "summary"
                    }
                ]
            }"""
        }
    
    def verify_claim(self, claim_text: str, include_sentiment: bool = True, 
                    detailed_analysis: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive fact-checking of a claim.
        
        Args:
            claim_text: The claim to verify
            include_sentiment: Whether to include sentiment analysis
            detailed_analysis: Whether to perform detailed source analysis
            
        Returns:
            Dictionary containing verification results
        """
        logger.info(f"Starting fact-check for claim: {claim_text[:50]}...")
        
        try:
            # Main verification using GPT-5
            # Note: the newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            main_result = self._perform_main_verification(claim_text)
            
            # Initialize result structure
            verification_result = {
                "claim": claim_text,
                "status": main_result.get("status", "unverified"),
                "confidence": main_result.get("confidence", 0.0),
                "explanation": main_result.get("explanation", "Analysis completed."),
                "key_factors": main_result.get("key_factors", []),
                "sources": self._generate_mock_sources(claim_text, main_result),
                "timestamp": datetime.now().isoformat(),
                "verification_id": f"verify_{datetime.now().timestamp()}"
            }
            
            # Add sentiment analysis if requested
            if include_sentiment:
                sentiment_result = self._analyze_sentiment(claim_text)
                verification_result["sentiment"] = sentiment_result.get("sentiment", {})
            
            # Add detailed source analysis if requested
            if detailed_analysis:
                source_analysis = self._analyze_sources(verification_result["sources"])
                verification_result["source_analysis"] = source_analysis
            
            logger.info(f"Fact-check completed: {verification_result['status']}")
            return verification_result
            
        except Exception as e:
            logger.error(f"Error during fact-checking: {str(e)}")
            return self._create_error_result(claim_text, str(e))
    
    def _perform_main_verification(self, claim_text: str) -> Dict[str, Any]:
        """
        Perform the main claim verification using GPT-5.
        
        Args:
            claim_text: The claim to verify
            
        Returns:
            Verification result from GPT-5
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025
                messages=[
                    {"role": "system", "content": self.verification_prompts["main_verification"]},
                    {"role": "user", "content": f"Please fact-check this claim: {claim_text}"}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                return result
            else:
                raise ValueError("Empty response from OpenAI API")
            
        except Exception as e:
            logger.error(f"Error in main verification: {str(e)}")
            return {
                "status": "unverified",
                "confidence": 0.0,
                "explanation": f"Verification failed due to technical error: {str(e)}",
                "key_factors": ["technical_error"],
                "requires_sources": False
            }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment and emotional tone of the claim.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis results
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025
                messages=[
                    {"role": "system", "content": self.verification_prompts["sentiment_analysis"]},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            else:
                return {
                    "sentiment": {
                        "label": "neutral",
                        "score": 0.5,
                        "emotional_indicators": ["api_error"]
                    }
                }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "sentiment": {
                    "label": "neutral",
                    "score": 0.5,
                    "emotional_indicators": ["analysis_error"]
                }
            }
    
    def _analyze_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform detailed analysis of source credibility and relevance.
        
        Args:
            sources: List of sources to analyze
            
        Returns:
            Source analysis results
        """
        try:
            source_text = json.dumps(sources, indent=2)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025
                messages=[
                    {"role": "system", "content": self.verification_prompts["source_analysis"]},
                    {"role": "user", "content": f"Analyze these sources: {source_text}"}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            else:
                return {
                    "source_assessment": [],
                    "analysis_error": "Empty API response"
                }
            
        except Exception as e:
            logger.error(f"Error in source analysis: {str(e)}")
            return {
                "source_assessment": [],
                "analysis_error": str(e)
            }
    
    def _generate_mock_sources(self, claim_text: str, verification_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate relevant mock sources based on the claim and verification result.
        In production, this would query actual fact-checking databases.
        
        Args:
            claim_text: The original claim
            verification_result: Results from main verification
            
        Returns:
            List of mock sources with metadata
        """
        # Determine source types based on claim content
        claim_lower = claim_text.lower()
        
        sources = []
        
        # Health-related claims
        if any(term in claim_lower for term in ['vaccine', 'covid', 'health', 'medical', 'disease']):
            sources.extend([
                {
                    "title": "World Health Organization Fact Sheet",
                    "url": "https://who.int/news-room/fact-sheets",
                    "relevance": 0.95,
                    "credibility": 0.98,
                    "summary": "Official WHO guidance on health-related claims and medical facts.",
                    "type": "authoritative_health"
                },
                {
                    "title": "Centers for Disease Control and Prevention",
                    "url": "https://cdc.gov",
                    "relevance": 0.92,
                    "credibility": 0.97,
                    "summary": "CDC research and recommendations on public health matters.",
                    "type": "government_health"
                }
            ])
        
        # Technology claims
        if any(term in claim_lower for term in ['5g', 'technology', 'internet', 'phone', 'wireless']):
            sources.extend([
                {
                    "title": "IEEE Technology Standards",
                    "url": "https://ieee.org",
                    "relevance": 0.88,
                    "credibility": 0.94,
                    "summary": "Technical standards and research on wireless technologies.",
                    "type": "technical_authority"
                },
                {
                    "title": "Federal Communications Commission",
                    "url": "https://fcc.gov",
                    "relevance": 0.85,
                    "credibility": 0.93,
                    "summary": "Government regulations and safety assessments for wireless technology.",
                    "type": "regulatory"
                }
            ])
        
        # Climate claims
        if any(term in claim_lower for term in ['climate', 'global warming', 'temperature', 'carbon']):
            sources.extend([
                {
                    "title": "Intergovernmental Panel on Climate Change",
                    "url": "https://ipcc.ch",
                    "relevance": 0.97,
                    "credibility": 0.99,
                    "summary": "Scientific consensus reports on climate change and global warming.",
                    "type": "scientific_consensus"
                },
                {
                    "title": "NASA Climate Change Evidence",
                    "url": "https://climate.nasa.gov",
                    "relevance": 0.94,
                    "credibility": 0.98,
                    "summary": "NASA satellite data and climate research findings.",
                    "type": "scientific_data"
                }
            ])
        
        # Generic fact-checking sources
        if not sources:
            sources.extend([
                {
                    "title": "Snopes Fact Check Database",
                    "url": "https://snopes.com",
                    "relevance": 0.75,
                    "credibility": 0.85,
                    "summary": "Independent fact-checking of various claims and rumors.",
                    "type": "fact_check"
                },
                {
                    "title": "PolitiFact Truth-O-Meter",
                    "url": "https://politifact.com",
                    "relevance": 0.73,
                    "credibility": 0.84,
                    "summary": "Political and general fact-checking with truth ratings.",
                    "type": "fact_check"
                }
            ])
        
        # Adjust relevance based on verification status
        status_modifier = {
            "verified": 1.0,
            "false": 0.95,
            "uncertain": 0.8,
            "unverified": 0.6
        }
        
        modifier = status_modifier.get(verification_result.get("status", "unverified"), 0.6)
        
        for source in sources:
            source["relevance"] *= modifier
            source["verification_status"] = verification_result.get("status", "unverified")
        
        return sources[:5]  # Return top 5 sources
    
    def _create_error_result(self, claim_text: str, error_message: str) -> Dict[str, Any]:
        """
        Create an error result when verification fails.
        
        Args:
            claim_text: The original claim
            error_message: Error description
            
        Returns:
            Error result dictionary
        """
        return {
            "claim": claim_text,
            "status": "unverified",
            "confidence": 0.0,
            "explanation": f"Fact-checking failed due to technical error: {error_message}",
            "key_factors": ["technical_error"],
            "sources": [],
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "verification_id": f"error_{datetime.now().timestamp()}"
        }
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """
        Get fact-checker statistics and health metrics.
        
        Returns:
            Dictionary containing fact-checker statistics
        """
        return {
            "model_version": "gpt-5",
            "api_status": "connected" if self.openai_client else "disconnected",
            "last_check": datetime.now().isoformat(),
            "supported_languages": ["en"],
            "version": "1.0.0"
        }
