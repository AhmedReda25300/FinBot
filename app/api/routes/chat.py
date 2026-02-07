"""
Chat/Question answering endpoints
"""
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorCollection
from app.models.schemas import ChatRequest, ChatResponse
from app.api.dependencies import get_db_collection, get_embedding_service, get_llm_service
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.utils.similarity import cosine_similarity

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    collection: AsyncIOMotorCollection = Depends(get_db_collection),
    embedding_svc: EmbeddingService = Depends(get_embedding_service),
    llm_svc: LLMService = Depends(get_llm_service)
):
    """
    Ask a question and get an AI-generated answer based on retrieved documents.
    
    Args:
        request: Chat request with question and filters
        
    Returns:
        ChatResponse with answer and source documents
    """
    try:
        # Get query embedding
        query_embedding = embedding_svc.get_embedding(
            request.question, 
            task_type="retrieval_query"
        )
        
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
            return ChatResponse(
                answer="عذراً، لم أتمكن من العثور على معلومات ذات صلة بسؤالك في قاعدة البيانات الحالية. يرجى التأكد من المعايير المستخدمة أو طرح السؤال بطريقة مختلفة.",
                sources=[],
                question=request.question
            )
        
        # Calculate similarities
        results = []
        for doc in documents:
            doc_embedding = np.array(doc['embedding'], dtype=np.float32)
            similarity = cosine_similarity(query_embedding, doc_embedding)
            
            results.append({
                'score': similarity,
                'text': doc['text'],
                'metadata': doc['metadata'],
                'embedding_source': doc['embedding_source']
            })
        
        # Sort by similarity and get top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = results[:request.num_chunks]
        
        # Generate answer using LLM
        answer = llm_svc.generate_answer(
            question=request.question,
            context_chunks=top_chunks,
            temperature=request.temperature
        )
        
        # Prepare sources for response
        sources = []
        for chunk in top_chunks:
            source_info = {
                'score': chunk['score'],
                'text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'metadata': chunk['metadata']
            }
            sources.append(source_info)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            question=request.question
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask_stream")
async def ask_question_stream(
    request: ChatRequest,
    collection: AsyncIOMotorCollection = Depends(get_db_collection),
    embedding_svc: EmbeddingService = Depends(get_embedding_service),
    llm_svc: LLMService = Depends(get_llm_service)
):
    """
    Ask a question with streaming response (for future implementation).
    Currently returns the same as /ask but can be enhanced for streaming.
    
    Args:
        request: Chat request with question and filters
        
    Returns:
        ChatResponse with answer and source documents
    """
    # For now, redirect to regular ask
    # In future, implement streaming using SSE or WebSockets
    return await ask_question(request, collection, embedding_svc, llm_svc)