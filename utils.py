"""
Utility functions for the fact-checking application.
Contains helper functions for data formatting, visualization, and common operations.
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st

def format_confidence_score(confidence: float) -> str:
    """
    Format confidence score as a percentage string.
    
    Args:
        confidence: Confidence score between 0.0 and 1.0
        
    Returns:
        Formatted percentage string
    """
    return f"{confidence * 100:.1f}"

def get_verification_color(status: str) -> str:
    """
    Get color code for verification status.
    
    Args:
        status: Verification status
        
    Returns:
        Color code for the status
    """
    color_map = {
        "verified": "#28a745",    # Green
        "false": "#dc3545",       # Red
        "uncertain": "#ffc107",   # Yellow
        "unverified": "#6c757d"   # Gray
    }
    return color_map.get(status.lower(), "#6c757d")

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."

def clean_claim_text(text: str) -> str:
    """
    Clean and normalize claim text for processing.
    
    Args:
        text: Raw claim text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    return text

def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extract key phrases from text for analysis.
    
    Args:
        text: Text to analyze
        max_phrases: Maximum number of phrases to extract
        
    Returns:
        List of key phrases
    """
    # Simple keyword extraction - in production use proper NLP
    words = text.lower().split()
    
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Filter meaningful words
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Create simple phrases (2-3 word combinations)
    phrases = []
    for i in range(len(meaningful_words) - 1):
        if i < len(meaningful_words) - 2:
            three_word = ' '.join(meaningful_words[i:i+3])
            phrases.append(three_word)
        two_word = ' '.join(meaningful_words[i:i+2])
        phrases.append(two_word)
    
    # Return unique phrases, limited by max_phrases
    unique_phrases = list(dict.fromkeys(phrases))
    return unique_phrases[:max_phrases]

def format_sources_for_display(sources: List[Dict[str, Any]]) -> str:
    """
    Format sources list for user-friendly display.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Formatted sources string
    """
    if not sources:
        return "No sources available"
    
    formatted = []
    for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
        title = source.get('title', 'Unknown Source')
        relevance = source.get('relevance', 0)
        formatted.append(f"{i}. {title} (Relevance: {relevance:.2f})")
    
    return "\n".join(formatted)

def calculate_overall_confidence(verification_result: Dict[str, Any]) -> float:
    """
    Calculate overall confidence score considering multiple factors.
    
    Args:
        verification_result: Complete verification result
        
    Returns:
        Overall confidence score
    """
    base_confidence = verification_result.get('confidence', 0.0)
    
    # Factor in source quality
    sources = verification_result.get('sources', [])
    if sources:
        avg_source_credibility = sum(s.get('credibility', 0) for s in sources) / len(sources)
        source_factor = avg_source_credibility * 0.3  # 30% weight for sources
    else:
        source_factor = 0.0
    
    # Factor in number of sources
    source_count_factor = min(len(sources) / 5.0, 1.0) * 0.2  # 20% weight for source count
    
    # Combine factors
    overall_confidence = base_confidence * 0.5 + source_factor + source_count_factor
    
    return min(overall_confidence, 1.0)  # Cap at 1.0

def create_verification_summary(result: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of verification results.
    
    Args:
        result: Verification result dictionary
        
    Returns:
        Summary string
    """
    status = result.get('status', 'unverified').title()
    confidence = format_confidence_score(result.get('confidence', 0.0))
    source_count = len(result.get('sources', []))
    
    summary = f"Status: {status} (Confidence: {confidence}%)"
    if source_count > 0:
        summary += f" | Sources: {source_count}"
    
    explanation = result.get('explanation', '')
    if explanation:
        summary += f"\n{explanation[:200]}..."
    
    return summary

def validate_claim_input(claim_text: str) -> tuple[bool, str]:
    """
    Validate user input for claim text.
    
    Args:
        claim_text: User input text
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not claim_text or not claim_text.strip():
        return False, "Please enter a claim to fact-check."
    
    if len(claim_text.strip()) < 10:
        return False, "Claim must be at least 10 characters long."
    
    if len(claim_text) > 5000:
        return False, "Claim must be less than 5000 characters."
    
    # Check for potentially harmful content patterns
    harmful_patterns = [
        r'<script',
        r'javascript:',
        r'on\w+\s*=',
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, claim_text, re.IGNORECASE):
            return False, "Input contains potentially harmful content."
    
    return True, ""

def load_sample_sources() -> List[Dict[str, Any]]:
    """
    Load sample fact-checking sources for demonstration.
    
    Returns:
        List of sample sources
    """
    return [
        {
            "id": "snopes_vaccine",
            "title": "Snopes: Vaccine Safety and Efficacy",
            "url": "https://snopes.com/fact-check/vaccines-safe-effective/",
            "credibility": 0.85,
            "type": "fact_check",
            "summary": "Comprehensive fact-check on vaccine safety claims"
        },
        {
            "id": "who_health",
            "title": "WHO Health Guidelines",
            "url": "https://who.int/health-topics/",
            "credibility": 0.98,
            "type": "authoritative",
            "summary": "Official health guidance from World Health Organization"
        },
        {
            "id": "reuters_factcheck",
            "title": "Reuters Fact Check",
            "url": "https://reuters.com/fact-check/",
            "credibility": 0.88,
            "type": "journalism",
            "summary": "Professional journalism fact-checking service"
        }
    ]

def export_verification_results(results: List[Dict[str, Any]]) -> str:
    """
    Export verification results to JSON format.
    
    Args:
        results: List of verification results
        
    Returns:
        JSON string of results
    """
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "total_claims": len(results),
        "results": results
    }
    
    return json.dumps(export_data, indent=2, default=str)

def parse_verification_confidence(confidence_str: str) -> float:
    """
    Parse confidence string back to float value.
    
    Args:
        confidence_str: Confidence as string (e.g., "85.5%")
        
    Returns:
        Confidence as float between 0.0 and 1.0
    """
    try:
        # Remove % sign and convert to float
        numeric_str = confidence_str.replace('%', '').strip()
        confidence = float(numeric_str) / 100.0
        return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
    except (ValueError, AttributeError):
        return 0.0

@st.cache_data
def get_cached_verification_result(claim_hash: str) -> Optional[Dict[str, Any]]:
    """
    Get cached verification result if available.
    
    Args:
        claim_hash: Hash of the claim text
        
    Returns:
        Cached result or None
    """
    # This would connect to a cache in production
    # For now, return None to always perform fresh verification
    return None

def generate_claim_hash(claim_text: str) -> str:
    """
    Generate a hash for claim text for caching purposes.
    
    Args:
        claim_text: The claim text
        
    Returns:
        Hash string
    """
    import hashlib
    return hashlib.md5(claim_text.encode()).hexdigest()
