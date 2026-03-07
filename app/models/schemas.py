"""
Pydantic models for request and response validation
"""
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field


class GuidelineDocument(BaseModel):
    """Schema for guideline documents"""
    document_title: str
    chunk_title: str
    content: str
    category: str
    source_type: str
    page_number: Optional[Union[int, float, str, List[Union[int, float, str]]]] = None
    page: Optional[Union[int, float, str, List[Union[int, float, str]]]] = None


class DecisionDocument(BaseModel):
    """Schema for decision documents"""
    تفصيل_القرار: Dict[str, Any]
    Source_Filename: str
    category: str
    source_type: str


class SearchRequest(BaseModel):
    """Schema for search requests"""
    query: str = Field(..., description="Search query text")
    category: Optional[str] = Field(None, description="Filter by category")
    source_type: Optional[str] = Field(None, description="Filter by source type")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")


class SearchResult(BaseModel):
    """Schema for search results"""
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    text: str = Field(..., description="Document text")
    embedding_source: str = Field(..., description="Source of the embedding")


class ChatRequest(BaseModel):
    """Schema for chat/question requests"""
    question: str = Field(..., description="User question")
    category: Optional[str] = Field(None, description="Filter by category")
    source_type: Optional[Union[str, List[str]]] = Field(
        None, description="Filter by source type (string or list of strings)"
    )
    num_chunks: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    custom_system_prompt: Optional[str] = Field(None, description="Custom system prompt for LLM")


class ChatResponse(BaseModel):
    """Schema for chat/question responses"""
    answer: str = Field(..., description="LLM generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source chunks used")
    question: str = Field(..., description="Original question")


class EmbedResponse(BaseModel):
    """Schema for embed response"""
    status: str
    message: str
    chunks_count: int


class DeleteResponse(BaseModel):
    """Schema for delete response"""
    status: str
    message: str
    deleted_count: int


class StatsResponse(BaseModel):
    """Schema for statistics response"""
    total_documents: int
    by_category: Dict[str, int]
    by_source_type: Dict[str, int]


class ChunkData(BaseModel):
    """Internal schema for chunk data"""
    unique_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    embedding_source: str