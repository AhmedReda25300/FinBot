"""
Chat/Question answering endpoints with custom system prompt support
"""
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorCollection
from app.models.schemas import ChatRequest, ChatResponse
from app.api.dependencies import get_db_collection, get_embedding_service, get_llm_service
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.utils.similarity import cosine_similarity
import json

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
        request: Chat request with question, filters, and optional custom system prompt
        
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
            if isinstance(request.source_type, list):
                mongo_filter['metadata.source_type'] = {'$in': request.source_type}
            else:
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
                'score': float(similarity),
                'text': doc['text'],
                'metadata': doc['metadata'],
                'embedding_source': doc['embedding_source']
            })
        
        # Sort by similarity and get top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = results[:request.num_chunks]
        
        # Generate answer using LLM with custom prompt if provided
        if request.custom_system_prompt:
            # Use custom prompt
            answer = await _generate_with_custom_prompt(
                llm_svc=llm_svc,
                question=request.question,
                context_chunks=top_chunks,
                temperature=request.temperature,
                model_name=request.model_name,
                custom_prompt=request.custom_system_prompt
            )
        else:
            # Use default prompt
            answer = llm_svc.generate_answer(
                question=request.question,
                context_chunks=top_chunks,
                temperature=request.temperature,
                model_name=request.model_name
            )
        
        # Prepare sources for response
        sources = []
        for chunk in top_chunks:
            source_info = {
                'score': float(chunk['score']),
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
    Ask a question with streaming response.
    
    Args:
        request: Chat request with question, filters, and optional custom system prompt
        
    Returns:
        Streaming response with answer chunks
    """
    print("Received chat request:", request)
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
            if isinstance(request.source_type, list):
                mongo_filter['metadata.source_type'] = {'$in': request.source_type}
            else:
                mongo_filter['metadata.source_type'] = request.source_type
        
        # Retrieve all matching documents
        cursor = collection.find(mongo_filter)
        documents = await cursor.to_list(length=None)
        
        if not documents:
            async def no_results_stream():
                error_response = {
                    "type": "error",
                    "content": "عذراً، لم أتمكن من العثور على معلومات ذات صلة بسؤالك."
                }
                yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
            
            return StreamingResponse(
                no_results_stream(),
                media_type="text/event-stream"
            )
        
        # Calculate similarities
        results = []
        for doc in documents:
            doc_embedding = np.array(doc['embedding'], dtype=np.float32)
            similarity = cosine_similarity(query_embedding, doc_embedding)
            
            results.append({
                'score': float(similarity),
                'text': doc['text'],
                'metadata': doc['metadata'],
                'embedding_source': doc['embedding_source']
            })
        
        # Sort by similarity and get top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = results[:request.num_chunks]
        
        # Prepare sources for response
        sources = []
        for chunk in top_chunks:
            source_info = {
                'score': float(chunk['score']),
                'text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'metadata': chunk['metadata']
            }
            sources.append(source_info)
        
        # Stream generator
        async def generate_stream():
            try:
                # First, send the sources
                sources_event = {
                    "type": "sources",
                    "content": sources
                }
                yield f"data: {json.dumps(sources_event, ensure_ascii=False)}\n\n"
                
                # Then stream the answer
                if request.custom_system_prompt:
                    # Use custom prompt
                    async for chunk in _generate_stream_with_custom_prompt(
                        llm_svc=llm_svc,
                        question=request.question,
                        context_chunks=top_chunks,
                        temperature=request.temperature,
                        model_name=request.model_name,
                        custom_prompt=request.custom_system_prompt
                    ):
                        answer_event = {
                            "type": "answer",
                            "content": chunk
                        }
                        yield f"data: {json.dumps(answer_event, ensure_ascii=False)}\n\n"
                else:
                    # Use default prompt
                    async for chunk in llm_svc.generate_answer_stream(
                        question=request.question,
                        context_chunks=top_chunks,
                        temperature=request.temperature,
                        model_name=request.model_name
                    ):
                        answer_event = {
                            "type": "answer",
                            "content": chunk
                        }
                        yield f"data: {json.dumps(answer_event, ensure_ascii=False)}\n\n"
                
                # Send done signal
                done_event = {
                    "type": "done",
                    "content": ""
                }
                yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                error_event = {
                    "type": "error",
                    "content": f"خطأ في توليد الإجابة: {str(e)}"
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for custom prompt handling

async def _generate_with_custom_prompt(
    llm_svc: LLMService,
    question: str,
    context_chunks: list,
    temperature: float,
    model_name: str,
    custom_prompt: str
) -> str:
    """
    Generate answer with custom system prompt (non-streaming).
    
    Args:
        llm_svc: LLM service instance
        question: User's question
        context_chunks: Retrieved context chunks
        temperature: LLM temperature
        custom_prompt: Custom system prompt
        
    Returns:
        Generated answer string
    """
    # Build context
    context = llm_svc._build_context(context_chunks)
    
    # Create messages with custom prompt
    messages = [
        {"role": "system", "content": custom_prompt},
        {
            "role": "user", 
            "content": f"السؤال:\n{question}\n\nالمصادر المتاحة:\n{context}\n\nقم بتحليل المصادر المتاحة والإجابة على السؤال وفقاً للإرشادات المحددة."
        }
    ]
    
    # Generate response
    response = await llm_svc.client.chat.completions.create(
        model=llm_svc.resolve_model_name(model_name),
        messages=messages,
        temperature=temperature,
        max_tokens=4096,
    )
    
    return response.choices[0].message.content


async def _generate_stream_with_custom_prompt(
    llm_svc: LLMService,
    question: str,
    context_chunks: list,
    temperature: float,
    model_name: str,
    custom_prompt: str
):
    """
    Generate streaming answer with custom system prompt.
    
    Args:
        llm_svc: LLM service instance
        question: User's question
        context_chunks: Retrieved context chunks
        temperature: LLM temperature
        custom_prompt: Custom system prompt
        
    Yields:
        Answer chunks
    """
    # Build context
    context = llm_svc._build_context(context_chunks)
    
    # Create messages with custom prompt
    messages = [
        {"role": "system", "content": custom_prompt},
        {
            "role": "user", 
            "content": f"السؤال:\n{question}\n\nالمصادر المتاحة:\n{context}\n\nقم بتحليل المصادر المتاحة والإجابة على السؤال وفقاً للإرشادات المحددة."
        }
    ]
    
    # Generate streaming response
    stream = await llm_svc.client.chat.completions.create(
        model=llm_svc.resolve_model_name(model_name),
        messages=messages,
        temperature=temperature,
        max_tokens=4096,
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content