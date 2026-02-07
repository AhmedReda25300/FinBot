"""
API dependencies
"""
from motor.motor_asyncio import AsyncIOMotorCollection
from app.database import get_collection
from app.services.embedding import embedding_service
from app.services.vector_store import vector_store_service
from app.services.llm import llm_service

def get_db_collection() -> AsyncIOMotorCollection:
    """Dependency to get MongoDB collection"""
    return get_collection()


def get_embedding_service():
    """Dependency to get embedding service"""
    return embedding_service


def get_vector_store_service():
    """Dependency to get vector store service"""
    return vector_store_service

def get_llm_service():
    """Dependency to get LLM service"""
    return llm_service