from openai import AsyncOpenAI
from typing import List, Dict, Any, AsyncIterator
from app.config import get_settings
import json
import google.generativeai as genai

settings = get_settings()

# Configure OpenAI
client = None
if settings.OPENAI_API_KEY:
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Configure Google Generative AI
if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY)

class LLMService:
    """Service for generating answers using OpenAI's GPT-4 model"""
    
    def __init__(self):
        if not settings.OPENAI_API_KEY and not settings.GOOGLE_API_KEY:
            raise ValueError("Neither OPENAI_API_KEY nor GOOGLE_API_KEY found in environment variables")
        self.client = client
        self.model_name = settings.LLM_MODEL
        self.allowed_models = {"gpt-4.1", "gpt-5-nano", "gpt-5-mini", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o-mini", "gemini-2.5-flash"}

    def resolve_model_name(self, model_name: str = None) -> str:
        """Return a safe model name, falling back to configured default when invalid/missing."""
        candidate = (model_name or "").strip()
        if candidate in self.allowed_models:
            return candidate
        return self.model_name
    
    async def generate_answer(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        model_name: str = None
    ) -> str:
        context = self._build_context(context_chunks)
        messages = self._create_messages(question, context)
        return await self.generate_answer_from_messages(
            messages=messages,
            temperature=temperature,
            model_name=model_name,
        )

    async def generate_answer_stream(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        model_name: str = None
    ) -> AsyncIterator[str]:
        context = self._build_context(context_chunks)
        messages = self._create_messages(question, context)
        async for chunk in self.generate_answer_stream_from_messages(
            messages=messages,
            temperature=temperature,
            model_name=model_name,
        ):
            yield chunk

    async def generate_answer_from_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        model_name: str = None,
        max_tokens: int = 4096,
    ) -> str:
        resolved_model = self.resolve_model_name(model_name)
        self._ensure_provider_available(resolved_model)

        if self._is_gemini_model(resolved_model):
            prompt = self._messages_to_gemini_prompt(messages)
            model = genai.GenerativeModel(resolved_model)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            text = getattr(response, "text", None)
            return text or ""

        response = await self.client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def generate_answer_stream_from_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        model_name: str = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        resolved_model = self.resolve_model_name(model_name)
        self._ensure_provider_available(resolved_model)

        if self._is_gemini_model(resolved_model):
            prompt = self._messages_to_gemini_prompt(messages)
            model = genai.GenerativeModel(resolved_model)
            stream = model.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                stream=True,
            )
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
            return

        stream = await self.client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def _is_gemini_model(self, model_name: str) -> bool:
        return str(model_name).strip().startswith("gemini-")

    def _ensure_provider_available(self, model_name: str) -> None:
        if self._is_gemini_model(model_name):
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required to use Gemini models")
            return

        if self.client is None:
            raise ValueError("OPENAI_API_KEY is required to use OpenAI GPT models")

    def _messages_to_gemini_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not content:
                continue

            if role == "system":
                prompt_parts.append(f"[System Instructions]\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"[Assistant]\n{content}")
            else:
                prompt_parts.append(f"[User]\n{content}")

        return "\n\n".join(prompt_parts)
    
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
            source_filename = metadata.get(
                'Source_Filename',
                metadata.get('source_filename', chunk.get('Source_Filename', chunk.get('source_filename', '')))
            )
            
            # Normalize page metadata: supports scalar values and lists (e.g. [1, 2]).
            page_val = metadata.get('page_number', metadata.get('page'))
            if page_val is None:
                page_val = chunk.get('page_number', chunk.get('page'))
            page_info = self._format_page_info(page_val)

            # --- 2. Handle Decisions vs Standard Docs ---
            if raw_source_type == 'Decisions' or 'تفصيل_القرار' in metadata or 'تفصيل_القرار' in chunk:
                # Decision Logic
                decision_details = self._normalize_decision_details(metadata, chunk)
                decision_id = self._extract_decision_id(decision_details, metadata, chunk)
                
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

    def _normalize_decision_details(self, metadata: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns decision details as a dict even when the payload is stored as JSON string.
        """
        raw_details = metadata.get('تفصيل_القرار')
        if raw_details is None:
            raw_details = chunk.get('تفصيل_القرار')

        if isinstance(raw_details, dict):
            return raw_details

        if isinstance(raw_details, str):
            text = raw_details.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}

        return {}

    def _extract_decision_id(
        self,
        decision_details: Dict[str, Any],
        metadata: Dict[str, Any],
        chunk: Dict[str, Any]
    ) -> str:
        """
        Extracts the best available decision number from multiple expected locations.
        """
        candidates = [
            decision_details.get('رقم_القرار_النهائي'),
            metadata.get('رقم_القرار_النهائي'),
            chunk.get('رقم_القرار_النهائي'),
            metadata.get('decision_id'),
            chunk.get('decision_id'),
            metadata.get('Source_Filename'),
            metadata.get('source_filename'),
            chunk.get('Source_Filename'),
            chunk.get('source_filename'),
        ]

        for value in candidates:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text

        return "Unknown"
    
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

4. **ترتيب المصادر إلزامي**:
    - اعرض المصادر بنفس ترتيب ظهورها في "المصادر المتاحة" (الملف رقم 1 ثم 2 ثم 3...).
    - لكل مصدر: اكتب الإجابة/الشرح الخاص به مباشرة تحته قبل الانتقال للمصدر التالي.
    - لا تخلط شرح مصدر مع مصدر آخر.
    - لا تبدأ بالمصدر الثاني قبل إكمال الأول.

5. **تنسيق الإجابة (Markdown)**:

   # [عنوان الإجابة]
   
   ---
   
   ## 📋 الملخص التنفيذي
   [إجابة مباشرة]
   
   ---
   
   ## 📚 التفاصيل من المصادر
   
    ### المصدر الأول: [اسم الملف]
   **النوع:** [نوع الملف]
   
   #### 📄 المحتوى ذو الصلة:
    ##### 🧩 [عنوان الفقرة (Chunk Title)]
    [شرح المحتوى بشكل مباشر، مع إدراج رقم الصفحة داخل سطر الشرح إذا كان متوفراً مثل: (رقم الصفحة: 6)]
   
   ---
   
    ### المصدر الثاني: [اسم الملف الثاني]
    **النوع:** [نوع الملف]

    #### 📄 المحتوى ذو الصلة:
     ##### 🧩 [عنوان الفقرة (Chunk Title)]
     [شرح المحتوى بشكل مباشر، مع إدراج رقم الصفحة داخل سطر الشرح إذا كان متوفراً]

    ---

    ### المصدر الثالث: [...]
    ... بنفس النمط
   
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