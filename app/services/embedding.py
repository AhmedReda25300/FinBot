"""
Embedding service for generating text embeddings
"""
import numpy as np
import google.generativeai as genai
from typing import List

from app.config import get_settings
from openai import OpenAI

settings = get_settings()
client = OpenAI()

# Configure Google AI
if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY)


class EmbeddingService:
    """Service for generating embeddings using Google's Gemini model"""
    
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        self.model = settings.EMBEDDING_MODEL
    
    def get_embedding(self, text: str, task_type: str = "retrieval_document") -> np.ndarray:
        """
        Get embedding for the given text.
        
        Args:
            text: Text to embed
            task_type: Type of task (retrieval_document or retrieval_query)
            
        Returns:
            Embedding vector as numpy array
        """
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type=task_type
        )
        # result = client.embeddings.create(
        #     input=text,
        #     model=self.model
        # )
        # embedding = result.data[0].embedding
        embedding = result['embedding']
        return np.array(embedding, dtype=np.float32)
    
    def get_embeddings_batch(self, texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            task_type: Type of task
            
        Returns:
            Array of embeddings
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text, task_type)
            embeddings.append(embedding)
        return np.vstack(embeddings)


# Global instance
embedding_service = EmbeddingService()