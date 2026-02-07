"""
Embedding endpoints
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorCollection
from app.models.schemas import GuidelineDocument, DecisionDocument, EmbedResponse
from app.api.dependencies import get_db_collection, get_vector_store_service
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/embed", tags=["Embedding"])


@router.post("", response_model=EmbedResponse)
async def embed_documents(
    documents: List[Dict[str, Any]],
    collection: AsyncIOMotorCollection = Depends(get_db_collection),
    vector_service: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Embed and save documents to MongoDB.
    Automatically detects if document is guideline or decision based on structure.
    
    Args:
        documents: List of documents to embed (can be guidelines or decisions)
        
    Returns:
        EmbedResponse with status and count
    """
    try:
        all_chunks = []
        
        for doc in documents:
            # Check if it's a decision document
            if 'تفصيل_القرار' in doc:
                decision = DecisionDocument(**doc)
                chunks = await vector_service.process_decision(decision)
            else:
                # It's a guideline document
                guideline = GuidelineDocument(**doc)
                chunks = await vector_service.process_guideline(guideline)
            
            all_chunks.extend(chunks)
        
        # Insert into MongoDB
        if all_chunks:
            await collection.insert_many(all_chunks)
        
        return EmbedResponse(
            status="success",
            message=f"Successfully embedded and saved {len(all_chunks)} chunks",
            chunks_count=len(all_chunks)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))