"""
Web Search Service
------------------
Acts as a pre-retrieval enrichment layer:
  1. Collects distinct document titles from MongoDB (filtered by source_type/category).
  2. Uses GPT-4o-mini to compose a focused internet search query.
  3. Searches the web via Tavily.
  4. Summarises the results to 2–3 sentences with GPT-4o-mini.
  5. Returns the summary as an enriched retrieval query for the RAG pipeline.
"""
from __future__ import annotations

from typing import List, Optional

from openai import AsyncOpenAI

from app.config import get_settings

settings = get_settings()


class WebSearchService:
    """Enriches a user question using live internet search before vector retrieval."""

    def __init__(self) -> None:
        self._openai: Optional[AsyncOpenAI] = None
        self._tavily = None

        if settings.OPENAI_API_KEY:
            self._openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        if settings.TAVILY_API_KEY:
            try:
                from tavily import AsyncTavilyClient
                self._tavily = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
            except ImportError:
                print("Warning: tavily-python is not installed. Web search will be disabled.")

    @property
    def is_available(self) -> bool:
        return self._openai is not None and self._tavily is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enrich_query(
        self,
        user_question: str,
        doc_titles: List[str],
        source_types: List[str],
        categories: List[str],
    ) -> str:
        """
        Returns an enriched 2-3 sentence summary built from a live web search.
        Falls back to the original question when the service is unavailable or
        when the search returns no useful results.
        """
        if not self.is_available:
            return user_question

        search_query = await self._generate_search_query(
            user_question, doc_titles, source_types, categories
        )

        raw_results = await self._search_web(search_query)
        if not raw_results:
            return user_question

        summary = await self._summarize(user_question, raw_results)
        return summary if summary else user_question

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_search_query(
        self,
        question: str,
        doc_titles: List[str],
        source_types: List[str],
        categories: List[str],
    ) -> str:
        """Ask GPT-4o-mini to compose a focused internet search query."""
        titles_str = "\n".join(f"- {t}" for t in doc_titles[:40])
        types_str = "، ".join(sorted(set(source_types))) if source_types else "غير محدد"
        cats_str = "، ".join(sorted(set(categories))) if categories else "غير محدد"

        prompt = (
            "أنت خبير في الأنظمة الضريبية السعودية.\n\n"
            f"لديك قاعدة معرفية تحتوي على:\n"
            f"• أنواع المصادر: {types_str}\n"
            f"• الفئات: {cats_str}\n"
            f"• عناوين الوثائق المتاحة:\n{titles_str}\n\n"
            f"السؤال: {question}\n\n"
            "اكتب **استعلام بحث إنترنت واحد موجز** (جملة واحدة، عربية أو إنجليزية) "
            "يجلب معلومات تكمّل قاعدة المعرفة وتساعد على الإجابة الصحيحة.\n"
            "أجب بالاستعلام فقط، بدون أي شرح."
        )

        response = await self._openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()

    async def _search_web(self, query: str) -> str:
        """Run a Tavily search and return concatenated snippets."""
        try:
            result = await self._tavily.search(
                query,
                max_results=3,
                search_depth="basic",
                include_answer=False,
            )
            snippets = [
                r.get("content", "").strip()
                for r in result.get("results", [])
                if r.get("content", "").strip()
            ]
            return "\n\n".join(snippets)
        except Exception as exc:
            print(f"[WebSearchService] Search error: {exc}")
            return ""

    async def _summarize(self, question: str, search_results: str) -> str:
        """Condense web search results into 2-3 Arabic sentences."""
        prompt = (
            "لديك نتائج بحث من الإنترنت حول سؤال ضريبي سعودي.\n\n"
            f"السؤال الأصلي:\n{question}\n\n"
            f"نتائج البحث:\n{search_results}\n\n"
            "لخّص النتائج في ٢–٣ جمل باللغة العربية تُركّز على المعلومات "
            "الأكثر صلة بالسؤال الأصلي. "
            "سيُستخدم هذا الملخص كاستعلام مُحسَّن للبحث في قاعدة بيانات وثائق ضريبية."
        )

        response = await self._openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()


# Global singleton
web_search_service = WebSearchService()
