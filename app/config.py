"""
Configuration settings for the application
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    APP_NAME: str = "Vector Store API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # MongoDB Settings
    MONGO_URI: str = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
    DB_NAME: str = os.getenv('DB_NAME', 'vector_store')
    COLLECTION_NAME: str = 'embeddings'
    
    # Google AI Settings
    GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY', '')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    # EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.5-flash"
    
    # Text Processing Settings
    MAX_TOKENS: int = 2048
    OVERLAP_RATIO: float = 0.1
    
    # Search Settings
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()