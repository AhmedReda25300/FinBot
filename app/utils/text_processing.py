"""
Text processing utilities
"""
import hashlib
from typing import List
from app.config import get_settings

settings = get_settings()


def split_text(text: str, max_tokens: int = None, overlap_ratio: float = None) -> List[str]:
    """
    Split text into smaller chunks if it exceeds max_tokens, with overlap.
    
    Args:
        text: Text to split
        max_tokens: Maximum number of tokens per chunk
        overlap_ratio: Ratio of overlap between chunks
        
    Returns:
        List of text chunks
    """
    if max_tokens is None:
        max_tokens = settings.MAX_TOKENS
    if overlap_ratio is None:
        overlap_ratio = settings.OVERLAP_RATIO
        
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    overlap = int(max_tokens * overlap_ratio)
    
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap if overlap > 0 else end

    return chunks


def generate_unique_id(content: str) -> str:
    """
    Generate a unique ID based on content hash.
    
    Args:
        content: Content to hash
        
    Returns:
        12-character unique ID
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()