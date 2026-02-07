"""
Vector store service for document processing and storage
"""
from datetime import datetime
from typing import List, Dict, Any
from app.models.schemas import GuidelineDocument, DecisionDocument
from app.services.embedding import embedding_service
from app.utils.text_processing import split_text, generate_unique_id


class VectorStoreService:
    """Service for processing and storing documents"""
    
    async def process_guideline(self, doc: GuidelineDocument) -> List[Dict[str, Any]]:
        """
        Process a guideline document and return chunks with embeddings.
        
        Args:
            doc: Guideline document
            
        Returns:
            List of chunks with embeddings and metadata
        """
        chunks = []
        text_chunks = split_text(doc.content)
        
        for chunk_idx, text_chunk in enumerate(text_chunks):
            embedding = embedding_service.get_embedding(text_chunk)
            unique_id = generate_unique_id(
                f"{doc.document_title}_{doc.chunk_title}_{chunk_idx}"
            )
            
            chunk_data = {
                'unique_id': unique_id,
                'text': text_chunk,
                'embedding': embedding.tolist(),
                'metadata': {
                    'document_title': doc.document_title,
                    'chunk_title': doc.chunk_title,
                    'category': doc.category,
                    'source_type': doc.source_type,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(text_chunks)
                },
                'embedding_source': 'guideline',
                'created_at': datetime.utcnow()
            }
            chunks.append(chunk_data)
        
        return chunks
    
    async def process_decision(self, doc: DecisionDocument) -> List[Dict[str, Any]]:
        """
        Process a decision document and return chunks with embeddings.
        Each بند gets its own embedding.
        
        Args:
            doc: Decision document
            
        Returns:
            List of chunks with embeddings and metadata
        """
        chunks = []
        decision_data = doc.تفصيل_القرار
        decision_id = decision_data.get("رقم_القرار_النهائي", "")
        
        # Process each بند separately
        بنود = decision_data.get("البنود_محل_الدعوى", [])
        
        for بند_idx, بند in enumerate(بنود):
            # Combine fields for embedding
            embedding_text = self._create_decision_embedding_text(بند)
            
            if not embedding_text:
                continue
            
            embedding = embedding_service.get_embedding(embedding_text)
            unique_id = generate_unique_id(f"{decision_id}_{بند_idx}")
            
            chunk_data = {
                'unique_id': unique_id,
                'text': embedding_text,
                'embedding': embedding.tolist(),
                'metadata': {
                    'decision_id': decision_id,
                    'decision_data': doc.تفصيل_القرار,
                    'source_filename': doc.Source_Filename,
                    'category': doc.category,
                    'source_type': doc.source_type,
                    'بند_name': بند.get('اسم_البند', ''),
                    'بند_index': بند_idx,
                    'total_بنود': len(بنود)
                },
                'embedding_source': f'بند_{بند_idx}',
                'created_at': datetime.utcnow()
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _create_decision_embedding_text(self, بند: Dict[str, Any]) -> str:
        """
        Create embedding text from بند fields.
        
        Args:
            بند: بند dictionary
            
        Returns:
            Combined text for embedding
        """
        embedding_text = (
            f"اسم البند: {بند.get('اسم_البند', '')}\n\n"
            f"نبذة مختصرة عن الاعتراض: {بند.get('نبذة_مختصرة_عن_الاعتراض', '')}\n\n"
            f"وجهة نظر المكلف بالتفصيل: {بند.get('وجهة_نظر_المكلف_بالتفصيل', '')}\n\n"
            f"وجهة نظر الهيئة بالتفصيل: {بند.get('وجهة_نظر_الهيئة_بالتفصيل', '')}\n\n"
            f"الرأي النهائي للجنة الاستئناف ومبرراته: {بند.get('الرأي_النهائي_لجنة_الاستئناف_ومبرراته', '')}"
        ).strip()
        
        return embedding_text


# Global instance
vector_store_service = VectorStoreService()