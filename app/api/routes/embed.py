"""
Embedding endpoints
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import BulkWriteError # Import this to handle duplicates

from app.models.schemas import GuidelineDocument, DecisionDocument, EmbedResponse
from app.api.dependencies import get_db_collection, get_vector_store_service
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/embed", tags=["Embedding"])


async def _process_document(
    doc: Dict[str, Any],
    vector_service: VectorStoreService
) -> List[Dict[str, Any]]:
    if 'تفصيل_القرار' in doc:
        decision = DecisionDocument(**doc)
        return await vector_service.process_decision(decision)

    guideline = GuidelineDocument(**doc)
    return await vector_service.process_guideline(guideline)

@router.post("", response_model=EmbedResponse)
async def embed_documents(
    documents: List[Dict[str, Any]],
    collection: AsyncIOMotorCollection = Depends(get_db_collection),
    vector_service: VectorStoreService = Depends(get_vector_store_service)
):
    try:
        all_chunks = []
        
        for doc in documents:
            chunks = await _process_document(doc, vector_service)
            
            # Ensure embedding_text exists
            for chunk in chunks:
                if 'embedding_text' not in chunk:
                     chunk['embedding_text'] = chunk.get('text', '')
            
            all_chunks.extend(chunks)
        
        # Insert into MongoDB with duplicate handling.
        inserted_count = 0
        duplicates_count = 0
        
        if all_chunks:
            try:
                # ordered=False: If one fails (duplicate), continue inserting the others.
                result = await collection.insert_many(all_chunks, ordered=False)
                inserted_count = len(result.inserted_ids)
            except BulkWriteError as e:
                # Calculate how many actually succeeded vs failed
                inserted_count = e.details['nInserted']
                duplicates_count = len(e.details['writeErrors'])
                
                # Optional: Print warning for debugging
                print(f"Warning: {duplicates_count} chunks were duplicates and skipped.")

        return EmbedResponse(
            status="success" if duplicates_count == 0 else "partial_success",
            message=f"Saved {inserted_count} chunks. {duplicates_count} duplicates skipped.",
            chunks_count=inserted_count
        )
    
    except Exception as e:
        print(f"Embedding Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")