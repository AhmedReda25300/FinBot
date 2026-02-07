"""
LLM service for question answering using retrieved context
"""
import google.generativeai as genai
from typing import List, Dict, Any
from app.config import get_settings

settings = get_settings()

# Configure Google AI
if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY)


class LLMService:
    """Service for generating answers using Google's Gemini model"""
    
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        self.model_name = settings.LLM_MODEL
    
    def generate_answer(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> str:
        """
        Generate an answer using the LLM with retrieved context.
        
        Args:
            question: User's question
            context_chunks: Retrieved context chunks with metadata
            temperature: LLM temperature for response generation
            
        Returns:
            Generated answer as string
        """
        # Build context from chunks
        context = self._build_context(context_chunks)
        
        # Create prompt
        prompt = self._create_prompt(question, context)
        
        # Generate response
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2048,
            )
        )
        
        return response.text
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            chunks: List of chunks with text and metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for idx, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # Format based on source type
            source_type = metadata.get('source_type', '')
            
            if 'decision' in source_type.lower():
                # Decision document
                decision_id = metadata.get('decision_id', 'Unknown')
                بند_name = metadata.get('بند_name', '')
                
                context_part = f"""
[مصدر {idx} - قرار رقم: {decision_id}]
البند: {بند_name}
المحتوى:
{text}
"""
            else:
                # Guideline document
                doc_title = metadata.get('document_title', 'Unknown')
                chunk_title = metadata.get('chunk_title', '')
                
                context_part = f"""
[مصدر {idx} - {doc_title}]
القسم: {chunk_title}
المحتوى:
{text}
"""
            
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create the prompt for the LLM.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""أنت مساعد ذكي متخصص في الأنظمة الضريبية والقرارات القضائية في المملكة العربية السعودية.
مهمتك هي الإجابة على أسئلة المستخدمين بناءً على المعلومات المقدمة من الوثائق والقرارات الرسمية.

تعليمات مهمة:
1. استخدم فقط المعلومات الموجودة في السياق المقدم للإجابة
2. إذا لم تجد إجابة كافية في السياق، قل ذلك بوضوح
3. اذكر مصادر المعلومات عند الإجابة (رقم القرار أو اسم الوثيقة)
4. كن دقيقاً ومهنياً في إجاباتك
5. إذا كانت الإجابة تتطلب استشارة قانونية متخصصة، أوصِ بذلك
6. استخدم اللغة العربية الفصحى الواضحة

السياق المتاح:
{context}

سؤال المستخدم:
{question}

الإجابة:"""
        
        return prompt


# Global instance
llm_service = LLMService()