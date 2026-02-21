"""
Embedding endpoints
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorCollection
import tiktoken
from pymongo.errors import BulkWriteError # Import this to handle duplicates

from app.models.schemas import GuidelineDocument, DecisionDocument, EmbedResponse
from app.api.dependencies import get_db_collection, get_vector_store_service
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/embed", tags=["Embedding"])

MAX_TOKEN_LIMIT = 8000 

def truncate_text_to_tokens(text: str, max_tokens: int = MAX_TOKEN_LIMIT) -> str:
    if not text:
        return ""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            return encoding.decode(tokens[:max_tokens])
        return text
    except Exception:
        char_limit = max_tokens * 2 
        if len(text) > char_limit:
            return text[:char_limit]
        return text

@router.post("", response_model=EmbedResponse)
async def embed_documents(
    documents: List[Dict[str, Any]],
    collection: AsyncIOMotorCollection = Depends(get_db_collection),
    vector_service: VectorStoreService = Depends(get_vector_store_service)
):
    try:
        all_chunks = []
        
        for doc in documents:
            # 1. Truncate Text to prevent OpenAI 400 Errors
            if 'تفصيل_القرار' in doc and isinstance(doc['تفصيل_القرار'], str):
                doc['تفصيل_القرار'] = truncate_text_to_tokens(doc['تفصيل_القرار'])
            
            for key in ['text', 'content', 'description']:
                if key in doc and isinstance(doc[key], str):
                    doc[key] = truncate_text_to_tokens(doc[key])

            # 2. Process Vectors
            if 'تفصيل_القرار' in doc:
                decision = DecisionDocument(**doc)
                chunks = await vector_service.process_decision(decision)
            else:
                guideline = GuidelineDocument(**doc)
                chunks = await vector_service.process_guideline(guideline)
            
            # 3. Ensure embedding_text exists
            for chunk in chunks:
                if 'embedding_text' not in chunk:
                     chunk['embedding_text'] = chunk.get('text', '')
            
            all_chunks.extend(chunks)
        
        # 4. Insert into MongoDB with Duplicate Handling
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