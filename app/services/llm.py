from openai import AsyncOpenAI
from typing import List, Dict, Any, AsyncIterator
from app.config import get_settings
import json

settings = get_settings()

# Configure OpenAI
client = None
if settings.OPENAI_API_KEY:
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

class LLMService:
    """Service for generating answers using OpenAI's GPT-4 model"""
    
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = client
        self.model_name = settings.LLM_MODEL
    
    def generate_answer(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> str:
        context = self._build_context(context_chunks)
        messages = self._create_messages(question, context)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    async def generate_answer_stream(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> AsyncIterator[str]:
        context = self._build_context(context_chunks)
        messages = self._create_messages(question, context)
        
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Builds context focusing on 'chunk_title' and conditional page info.
        """
        source_type_mapping = {
            "Laws": "نظام / لائحة",
            "Decisions": "قرار لجنة فصل",
            "Guidelines": "دليل إرشادي",
            "Publications": "منشورات",
            "Regulations": "لوائح"
        }

        grouped_docs = {}

        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            score = chunk.get('score', 0)
            
            raw_source_type = metadata.get('source_type', 'Unknown')
            display_source_type = source_type_mapping.get(raw_source_type, raw_source_type)
            
            # --- 1. Identify Document & Keys ---
            doc_key = ""
            doc_title = ""
            excerpt_text = ""
            section_title = ""
            source_filename = metadata.get('Source_Filename', metadata.get('source_filename', ''))
            
            # Normalize page metadata: supports scalar values and lists (e.g. [1, 2]).
            page_val = metadata.get('page_number', metadata.get('page'))
            if page_val is None:
                page_val = chunk.get('page_number', chunk.get('page'))
            page_info = self._format_page_info(page_val)

            # --- 2. Handle Decisions vs Standard Docs ---
            if raw_source_type == 'Decisions' or 'تفصيل_القرار' in metadata:
                # Decision Logic
                decision_details = metadata.get('تفصيل_القرار', {})
                decision_id = decision_details.get('رقم_القرار_النهائي', metadata.get('Source_Filename', 'Unknown'))
                
                doc_title = f"قرار رقم: {decision_id}"
                canonical_decision_key = source_filename or str(decision_id)
                doc_key = f"DEC_{str(canonical_decision_key).strip().lower()}"
                
                # Handle Dispute Items as the "Text"
                dispute_items = decision_details.get('البنود_محل_الدعوى', [])
                if dispute_items:
                    items_text = []
                    for item in dispute_items:
                        items_text.append(
                            f"- بند النزاع: {item.get('اسم_البند', '')}\n"
                            f"- الملخص: {item.get('نبذة_مختصرة_عن_الاعتراض', '')}\n"
                            f"- حكم اللجنة النهائية: {item.get('رأي_اللجنة_النهائية_ومبرراته', '')}"
                        )
                    excerpt_text = "\n".join(items_text)
                    section_title = "البنود محل الدعوى"
                else:
                    excerpt_text = chunk.get('text', chunk.get('content', ''))
                    section_title = metadata.get('chunk_title', 'نص القرار')

            else:
                # Standard Document Logic (Guidelines/Laws) - MATCHING YOUR JSON
                doc_title = metadata.get('document_title', source_filename or 'Unknown Document')
                canonical_source_key = source_filename or doc_title
                doc_key = f"DOC_{str(canonical_source_key).strip().lower()}"
                
                # Extract the chunk title explicitly
                section_title = metadata.get('chunk_title', 'نص عام')
                excerpt_text = chunk.get('content', chunk.get('text', ''))

            # --- 3. Grouping ---
            if doc_key not in grouped_docs:
                grouped_docs[doc_key] = {
                    "title": doc_title,
                    "type": display_source_type,
                    "max_score": score,
                    "excerpts": [],
                    "_excerpt_keys": set()
                }

            # Prefer a richer title if current title is unknown/generic.
            if grouped_docs[doc_key]["title"] in ("Unknown Document", "Unknown") and doc_title not in ("Unknown Document", "Unknown"):
                grouped_docs[doc_key]["title"] = doc_title
            
            if score > grouped_docs[doc_key]["max_score"]:
                grouped_docs[doc_key]["max_score"] = score

            # Add explicit metadata to the excerpt list
            excerpt_key = (section_title, excerpt_text, page_info)
            if excerpt_key not in grouped_docs[doc_key]["_excerpt_keys"]:
                grouped_docs[doc_key]["_excerpt_keys"].add(excerpt_key)
                grouped_docs[doc_key]["excerpts"].append({
                    "section": section_title,
                    "text": excerpt_text,
                    "page": page_info  # Passes None if not found
                })

        # --- 4. Format String for LLM ---
        context_parts = []
        sorted_docs = sorted(grouped_docs.values(), key=lambda x: x['max_score'], reverse=True)

        for idx, doc in enumerate(sorted_docs, 1):
            doc.pop("_excerpt_keys", None)
            excerpts_str = ""
            for i, exc in enumerate(doc['excerpts'], 1):

                page_line = f"- رقم الصفحة: {exc['page']}\n" if exc['page'] else ""

                # Construct the block
                excerpts_str += f"""
                >>> اقتباس رقم {i}:
                - عنوان الفقرة (Chunk Title): {exc['section']}
                {page_line}
                - النص:
                {exc['text']}
                --------------------------------------------------
                """

            context_part = f"""
            [الملف رقم {idx}]
            نوع الملف: {doc['type']}
            اسم الملف: {doc['title']}
            
            النصوص المقتبسة:
            {excerpts_str}
            """
            context_parts.append(context_part)
        
        return "\n==================================================\n".join(context_parts)

    def _format_page_info(self, page_value: Any) -> str:
        """Returns a normalized page string or None when page info is missing."""
        if page_value is None:
            return None

        if isinstance(page_value, list):
            normalized_pages = []
            for value in page_value:
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    normalized_pages.append(text)
            return ", ".join(normalized_pages) if normalized_pages else None

        text = str(page_value).strip()
        return text if text else None
    
    def _create_messages(self, question: str, context: str) -> List[Dict[str, str]]:
        """
        System Prompt updated to handle conditional page numbers.
        """
        system_prompt = """
أنت مساعد قانوني ذكي متخصص في الأنظمة الضريبية السعودية.

إرشادات صارمة للإجابة:
1. **الاعتماد على المصادر**: أجب بناءً فقط على المعلومات الواردة في "المصادر المتاحة".

2. **تجميع المصدر الواحد**: إذا ظهرت عدة اقتباسات من نفس الملف/المصدر، تعامل معها كمصدر واحد فقط في الإجابة (لا تكرر نفس اسم الملف كمصدر مستقل).

3. **التوثيق الدقيق (Citations)**:
   - يجب ذكر **عنوان الفقرة (Chunk Title)** دائماً عند الاقتباس.
    - يجب أن يظهر **عنوان الفقرة في بداية المقطع المشروح** بهذا الشكل: `##### 🧩 [عنوان الفقرة]`.
    - **رقم الصفحة**: إذا كان "رقم الصفحة" متوفراً في المصدر، اذكره **داخل سطر الشرح نفسه** وليس في قسم منفصل (مثال: `... وفقاً للنص (رقم الصفحة: 6)` أو `... (رقم الصفحة: 5، 6)`).
    - إذا كان رقم الصفحة غير متوفر، لا تذكر الصفحة نهائياً.

4. **تنسيق الإجابة (Markdown)**:

   # [عنوان الإجابة]
   
   ---
   
   ## 📋 الملخص التنفيذي
   [إجابة مباشرة]
   
   ---
   
   ## 📚 التفاصيل من المصادر
   
   ### 1. [اسم الملف]
   **النوع:** [نوع الملف]
   
   #### 📄 المحتوى ذو الصلة:
    ##### 🧩 [عنوان الفقرة (Chunk Title)]
    [شرح المحتوى بشكل مباشر، مع إدراج رقم الصفحة داخل سطر الشرح إذا كان متوفراً مثل: (رقم الصفحة: 6)]
   
   ---
   
   ### 2. [اسم الملف الثاني...]
   ...
   
   ---
   
   ## 💡 الخلاصة
   [نصيحة نهائية]
"""

        user_prompt = f"""السؤال:
{question}

المصادر المتاحة:
{context}
"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

# Global instance
llm_service = LLMService()