"""
Similarity calculation utilities
"""
import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
    return float(np.dot(vec1_norm, vec2_norm))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(vec1 - vec2))


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vec: Vector to normalize
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm