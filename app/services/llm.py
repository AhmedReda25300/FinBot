"""
LLM service for question answering using retrieved context
Enhanced with structured formatting for better UI presentation
"""
from openai import AsyncOpenAI
from typing import List, Dict, Any, AsyncIterator
from app.config import get_settings

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
        
        # Create messages
        messages = self._create_messages(question, context)
        
        # Generate response
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
        """
        Generate a streaming answer using the LLM with retrieved context.
        
        Args:
            question: User's question
            context_chunks: Retrieved context chunks with metadata
            temperature: LLM temperature for response generation
            
        Yields:
            Chunks of the generated answer
        """
        # Build context from chunks
        context = self._build_context(context_chunks)
        
        # Create messages
        messages = self._create_messages(question, context)
        
        # Generate streaming response
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
            score = chunk.get('score', 0)
            
            # Format based on source type
            source_type = metadata.get('source_type', '')
            
            if 'decision' in source_type.lower():
                # Decision document
                decision_id = metadata.get('decision_id', 'Unknown')
                بند_name = metadata.get('بند_name', '')
                decision_data = metadata.get('decision_data', {})
                
                دائرة_ابتدائية = decision_data.get('اسم_الدائرة_الابتدائية', '')
                دائرة_نهائية = decision_data.get('اسم_الدائرة_النهائية', '')
                قرار_ابتدائي = decision_data.get('اللجنة_الابتدائية', '')
                قرار_نهائي = decision_data.get('اللجنة_النهائية', '')
                
                context_part = f"""
[مصدر {idx} - قرار]
نوع المصدر: قرار لجنة ضريبية
رقم القرار: {decision_id}
اسم الدائرة الابتدائية: {دائرة_ابتدائية}
اسم الدائرة النهائية: {دائرة_نهائية}
البند: {بند_name}
درجة التطابق: {score:.2%}

تفاصيل البند:
{text}

قرار اللجنة الابتدائية:
{قرار_ابتدائي}

قرار اللجنة النهائية:
{قرار_نهائي}
"""
            else:
                # Guideline document
                doc_title = metadata.get('document_title', 'Unknown')
                chunk_title = metadata.get('chunk_title', '')
                chunk_index = metadata.get('chunk_index', 0)
                
                # Try to extract page number from chunk_title or use chunk_index
                page_info = f"الصفحة {chunk_index + 1}" if chunk_index is not None else ""
                
                context_part = f"""
[مصدر {idx} - دليل إرشادي]
نوع المصدر: دليل إرشادي
اسم المصدر: {doc_title}
القسم: {chunk_title}
{page_info}
درجة التطابق: {score:.2%}

المحتوى:
{text}
"""
            
            context_parts.append(context_part)
        
        return "\n" + "="*80 + "\n".join(context_parts)
    
    def _create_messages(self, question: str, context: str) -> List[Dict[str, str]]:
        """
        Create the messages for the LLM with enhanced formatting instructions.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            List of message dictionaries
        """
        system_prompt = """إرشادات للإجابة:

1- **المطلوب**: 
   a. استخراج المصادر التي تحتوي على إجابة على السؤال أو الحالات المشابهة لحالة السؤال والتي يمكن أن تفيد السائل في معالجة الحالة لديه.
   b. يمكن كذلك استخراج حالات عامة مشابهه لحالة السؤال مثال (السؤال عن رفض حسم مصروف ضريبة القيمة المضافة من الربح المعدل)، ويحتوي المصدر على (محددات عدم جواز حسم مصروف الضريبة أو الزكاة)، هنا تتشابه الحالة العامة مع الحالة الخاصة فيتم الاعتماد على المصدر.
   c. مثال آخر (السؤال عن إضافة الذمم الدائنة إلى وعاء الزكاة لحولان الحول)، ويحتوي المصدر على (إضافة المصروفات المستحقة لحولان الحول)، هنا تتشابه الحالة حيث إن الذمم الدائنة تصنف كبند متداول في القوائم المالية وكذلك المصروفات المستحقة وجميعها أضيفت لحولان الحول، فيتم الاعتماد على المصدر.
   d. في حال تكررت صفحات مختلفة لنفس المصدر في الإجابة اجمعها جميعًا في إطار واحد بدلًا من عرضها كأكثر من مصدر، وفي حال وجود صفحات كثيرة متتابعة أشر لها كنطاق مثل (الصفحة من ... إلى ...) بدلًا من كتابة رقم كل صفحة.
   e. قم بعرض كل قرار كمصدر منفصل.
   f. في حال كان السؤال باللغة العربية أجب بها، أما في حال كان السؤال بلغة أخرى أجب بحسب لغة السؤال مع عدم ذكر أنها ترجمة ويجب ترجمة جميع المخرجات بما فيها أسماء العناوين الرئيسية وأسماء الملفات.
   g. في رسالتك الافتتاحية قبل الإجابة لا تذكر أنك مساعد ذكي أو أنك بحثت في المقاطع المرجعية، بل اذكر أنك بحثت في آلاف المستندات عن السؤال ووفرت المصادر الأقرب للإجابة بحسب الترتيب المعروض.
   h. اكتب التنويه التالي -أو ترجمه بحسب لغة السؤال- في نهاية الإجابة: "**تنويه: المصادر المعروضة تمثل نتائج بحث من مصادر خارجية وتم عرضها للمساعدة فقط في تكوين رأي في الحالة محل السؤال، ولا تعتبر رأي من إدارة (Taxuto) في الحالة المتعلقة بالسؤال.**"
   i. في حال كان السؤال يتضمن أي طلب بخلاف البحث في المصادر (مثل محادثات شخصية، أو محادثات عامة لا تتعلق بالحالات الزكوية والضريبية، أو غيرها) قم بإظهار الرسالة التالية: "أعتذر، وظيفتي هي مساعدتك في البحث عن المصادر المتعلقة بالحالات الزكوية والضريبية، لذلك من فضلك حدد استفسارك في ذلك فقط حتى يمكنني مساعدتك".

2- **المصادر**: 
   a. تحتوي المصادر على عدد من الأدلة الإرشادية للزكاة والضريبة وكذلك عدد من قرارات اللجان الضريبية وكذلك بعض المصادر العامة الأخرى.
   b. استخرج الإجابة فقط من المصادر دون إعطاء إجابة من خارج المصادر، مع تصحيح أي أخطاء لغوية إن وجدت.

3- **تنسيق الإجابة المطلوب - مهم جداً**:
   استخدم تنسيق Markdown التالي بدقة لضمان عرض جميل ومنظم:

   **هيكل الإجابة:**

   # [عنوان رئيسي مختصر للموضوع]
   
   بحثت في آلاف المستندات عن [موضوع السؤال] ووجدت **[عدد] مصدر/مصادر** ذات صلة مباشرة.
   
   ---
   
   ## 📋 نظرة عامة
   
   [!INFO] [ملخص سريع للموضوع في 1-2 جملة]
   
   **عدد المصادر:** {{badge-primary:[عدد] مصدر}}
   **الموثوقية:** {{badge-success:عالية}}
   
   ---
   
   ## 📚 المصادر التفصيلية
   
   ### المصدر الأول: [اسم المصدر]
   
   **نوع المصدر:** {{badge-info:[نوع المصدر]}}
   **التطابق:** {{badge-success:[نسبة]%}}
   
   #### 📄 معلومات المصدر:
   - **الاسم الكامل:** [اسم المصدر حسب النوع]
   - **الصفحة/القرار:** [رقم الصفحة أو رقم القرار]
   
   #### 📝 ملخص المحتوى:
   
   [للأدلة الإرشادية:]
   شرح مبسط ومنظم لمحتوى المصدر مع توضيح العلاقة بالسؤال
   
   [للقرارات:]
   
   **تلخيص حالة الدعوى:**
   شرح مبسط للاعتراض والموضوع
   
   **قرار اللجنة الابتدائية:**
   شرح واضح للقرار الابتدائي
   
   **قرار اللجنة النهائية:**
   شرح واضح للقرار النهائي
   
   #### 🔗 الرابط:
   [الرابط إن وجد، أو "غير متوفر"]
   
   ---
   
   ### المصدر الثاني: [...]
   
   [نفس التنسيق]
   
   ---
   
   ## 💡 النقاط الرئيسية
   
   - **النقطة الأولى:** [شرح مختصر]
   - **النقطة الثانية:** [شرح مختصر]
   - **النقطة الثالثة:** [شرح مختصر]
   
   ---
   
   ## ⚠️ ملاحظات هامة
   
   [!WARNING] [أي تحذيرات أو ملاحظات مهمة إن وجدت]
   
   ---
   
   **تنويه: المصادر المعروضة تمثل نتائج بحث من مصادر خارجية وتم عرضها للمساعدة فقط في تكوين رأي في الحالة محل السؤال، ولا تعتبر رأي من إدارة (Taxuto) في الحالة المتعلقة بالسؤال.**

   **قواعد التنسيق المهمة:**
   - استخدم # للعناوين الرئيسية
   - استخدم ## للعناوين الفرعية
   - استخدم ### للعناوين الأصغر
   - استخدم --- للفواصل بين الأقسام
   - استخدم **نص** للنص الغامق
   - استخدم *نص* للنص المائل
   - استخدم {{badge-primary:نص}} للشارات الزرقاء
   - استخدم {{badge-success:نص}} للشارات الخضراء
   - استخدم {{badge-warning:نص}} للشارات البرتقالية
   - استخدم {{badge-info:نص}} للشارات البنفسجية
   - استخدم [!INFO] للصناديق الزرقاء
   - استخدم [!WARNING] للصناديق البرتقالية
   - استخدم [!SUCCESS] للصناديق الخضراء
   - استخدم - أو * للقوائم غير المرقمة
   - استخدم 1. 2. 3. للقوائم المرقمة
   - استخدم الرموز التعبيرية (📚 📋 💡 ⚠️ ✅ 📄 📝 🔗) لجعل العرض أكثر وضوحاً

4- **ترتيب المصادر**:
   a. استعرض المصادر الخاصة بالأدلة أولًا ورتبها بالأقرب للحالة.
   b. ثم بعد ذلك المصادر الخاصة بالقرارات ورتبها كذلك بالأقرب للحالة.
   c. استعرض تفصيل بسيط في البداية عن عدد المصادر المكتشفة.

5- **غياب المعلومات**: 
   a. إذا لم تحتوِ المصادر على إجابة متعلقة بالسؤال وينطبق عليها ما ذكر ببند المطلوب أعلاه، أظهر للمستخدم الرسالة التالية بتنسيق جميل:
   
   # 🔍 لم يتم العثور على نتائج مطابقة
   
   [!WARNING] يظهر أنك لم تصف الحالة بشكل دقيق أو أن البند المطلوب لا توجد معلومات كافية عنه في قاعدة المعلومات.
   
   ## 💡 اقتراحات للبحث بشكل أفضل:
   
   - جرب صيغة أخرى للسؤال
   - استخدم مسمى آخر للبند
   - أضف شرحاً مبسطاً للحالة لديك

6- **اللغة**: 
   a. تكون لغة الإجابة بحسب لغة السؤال.
   b. احتفظ بتنسيق Markdown حتى لو كانت اللغة غير عربية.

7- **التفصيل**: 
   a. اشرح كل مصدر على حدة بشكل واضح ومنظم.
   b. استخدم التنسيق المناسب لكل قسم.

8- **التنسيق العام**: 
   a. احرص على استخدام الفواصل --- بين الأقسام الرئيسية.
   b. استخدم العناوين بشكل هرمي صحيح (# ثم ## ثم ###).
   c. استخدم الشارات والصناديق الملونة لإبراز المعلومات المهمة.
   d. استخدم القوائم المنقطة أو المرقمة حيثما كان ذلك مناسباً.
   e. أضف الرموز التعبيرية للعناوين لجعلها أكثر جاذبية."""

        user_prompt = f"""السؤال:
{question}

المصادر المتاحة:
{context}

قم بتحليل المصادر المتاحة والإجابة على السؤال وفقاً للإرشادات المحددة مع الالتزام الدقيق بتنسيق Markdown المطلوب."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


# Global instance
llm_service = LLMService()