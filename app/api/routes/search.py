"""
Search endpoints
"""
from typing import List
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorCollection
from app.models.schemas import SearchRequest, SearchResult
from app.api.dependencies import get_db_collection, get_embedding_service
from app.services.embedding import EmbeddingService
from app.utils.similarity import cosine_similarity

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=List[SearchResult])
async def search_documents(
    request: SearchRequest,
    collection: AsyncIOMotorCollection = Depends(get_db_collection),
    embedding_svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Search for similar documents using vector similarity.
    Optionally filter by category and source_type.
    
    Args:
        request: Search request with query and filters
        
    Returns:
        List of search results sorted by similarity score
    """
    try:
        # Get query embedding
        # query_embedding = embedding_svc.get_embedding(
        #     request.query, 
        #     task_type="retrieval_query"
        # )
        query_embedding = embedding_svc.get_embedding(request.query)
        # Build MongoDB filter
        mongo_filter = {}
        if request.category:
            mongo_filter['metadata.category'] = request.category
        if request.source_type:
            mongo_filter['metadata.source_type'] = request.source_type
        
        # Retrieve all matching documents
        cursor = collection.find(mongo_filter)
        documents = await cursor.to_list(length=None)
        
        if not documents:
            return []
        
        # Calculate similarities
        results = []
        for doc in documents:
            doc_embedding = np.array(doc['embedding'], dtype=np.float32)
            similarity = cosine_similarity(query_embedding, doc_embedding)
            
            results.append(SearchResult(
                score=similarity,
                metadata=doc['metadata'],
                text=doc['text'],
                embedding_source=doc['embedding_source']
            ))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:request.top_k]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))