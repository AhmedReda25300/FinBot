import streamlit as st
import pickle
import google.generativeai as genai
import faiss
import numpy as np
from dotenv import load_dotenv
import os
from datetime import datetime
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="مساعد المستندات المالية والزكوية",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for RTL support and dark theme
st.markdown("""
<style>
    /* Dark theme */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }
    
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    [data-testid="stHeader"] {
        background-color: #0e1117;
    }
    
    /* Main header */
    .main-header {
        text-align: center;
        color: #4CAF50;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 15px;
        margin-bottom: 30px;
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        direction: rtl;
        text-align: right;
        line-height: 1.6;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        margin-right: 20%;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }
    
    /* Input styling */
    .stChatInput {
        direction: rtl;
    }
    
    .stChatInput textarea {
        direction: rtl;
        text-align: right;
        background-color: #262730;
        color: white;
        border: 2px solid #4CAF50;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.4);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e1e1e;
        color: white;
        direction: rtl;
        border-radius: 5px;
    }
    
    .streamlit-expanderContent {
        background-color: #262730;
        direction: rtl;
        text-align: right;
    }
    
    /* Text area in sidebar */
    [data-testid="stSidebar"] textarea {
        direction: rtl;
        text-align: right;
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #4CAF50;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #27ae60;
        color: white;
        direction: rtl;
        text-align: right;
    }
    
    /* Info boxes */
    .element-container div[data-testid="stMarkdownContainer"] p {
        direction: rtl;
        text-align: right;
    }
    
    /* Error messages */
    .error-message {
        background-color: #e74c3c;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    """Load the FAISS indices and chunks data"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("مطلوب مفتاح API. تأكد من وجود GOOGLE_API_KEY في ملف .env")
        return None, None, None, None, None, None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    guides_chunks = []
    guides_index = None
    decisions_chunks = []
    decisions_index = None
    
    # Load guides
    try:
        guides_index = faiss.read_index('guides.faiss')
        with open('guides_chunks.pkl', 'rb') as f:
            guides_chunks = pickle.load(f)
    except FileNotFoundError:
        st.warning("لم يتم العثور على فهرس الأدلة أو المقاطع.")
    
    # Load decisions
    try:
        decisions_index = faiss.read_index('decisions copy.faiss')
        with open('decisions_chunks_new.pkl', 'rb') as f:
            decisions_chunks = pickle.load(f)
    except FileNotFoundError:
        st.warning("لم يتم العثور على فهرس القرارات أو المقاطع.")
    
    return model, guides_index, guides_chunks, decisions_index, decisions_chunks, api_key

def get_embedding(text, api_key, task_type="retrieval_query"):
    """Get embedding for the given text using Google embeddings model."""
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="gemini-embedding-001",
        content=text,
        task_type=task_type
    )
    embedding = np.array(result['embedding'], dtype=np.float32)
    faiss.normalize_L2(embedding.reshape(1, -1))
    return embedding

def safe_get_metadata_value(metadata, key, default=''):
    """Safely get metadata value and convert to string"""
    value = metadata.get(key, default)
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return str(value)
    return str(value).strip()

def enrich_decision_chunk(chunk, all_chunks):
    """
    Enrich a decision chunk with all metadata fields from related chunks
    of the same decision number.
    """
    metadata = chunk.get('metadata', {})
    decision_num = safe_get_metadata_value(metadata, 'رقم القرار')
    
    if not decision_num or decision_num == 'غير محدد':
        return chunk
    
    # Fields we want to ensure are complete
    required_fields = [
        'رقم القرار',
        'رقم الدعوى / الاستْناف',
        'العام الضريبي المرتبط بالقرار',
        'تفصيل القرار',
        'اسباب القرار',
        'البنود محل الاعتراض',
        'منطوق القرار',
        'Source_Filename'
    ]
    
    # Collect all metadata from chunks with the same decision number
    enriched_metadata = {}
    for field in required_fields:
        enriched_metadata[field] = safe_get_metadata_value(metadata, field)
    
    # Iterate through all chunks to find matching decisions
    for other_chunk in all_chunks:
        other_metadata = other_chunk.get('metadata', {})
        other_decision = safe_get_metadata_value(other_metadata, 'رقم القرار')
        
        if other_decision == decision_num:
            # Merge metadata, prioritizing non-empty values
            for field in required_fields:
                current_value = enriched_metadata.get(field, '')
                new_value = safe_get_metadata_value(other_metadata, field)
                
                # Only update if current value is empty
                if not current_value and new_value:
                    enriched_metadata[field] = new_value
                # For text fields, concatenate if both have values and they're different
                elif field in ['اسباب القرار', 'البنود محل الاعتراض', 'منطوق القرار', 'تفصيل القرار']:
                    if new_value and current_value and new_value not in current_value:
                        enriched_metadata[field] = f"{current_value}\n\n{new_value}"
    
    # Update the chunk with enriched metadata
    chunk_copy = chunk.copy()
    chunk_copy['metadata'] = enriched_metadata
    
    return chunk_copy

def retrieve_chunks(query_embedding, index, chunks, top_k=12):
    """Retrieve top_k most similar unique chunks to the query."""
    if not index or not chunks:
        return []

    try:
        # Request more chunks than needed to account for potential duplicates
        search_k = min(top_k * 3, len(chunks))
        distances, indices = index.search(query_embedding.reshape(1, -1), search_k)
        
        # Track unique chunks by decision number (or content hash for guides)
        seen_decisions = set()
        unique_chunks = []
        
        for i, idx in enumerate(indices[0]):
            chunk = chunks[idx].copy()  # Create a copy to avoid modifying original
            chunk['similarity_score'] = float(distances[0][i])
            
            # Check if this is a decision chunk
            metadata = chunk.get('metadata', {})
            decision_num = safe_get_metadata_value(metadata, 'رقم القرار')
            
            if decision_num and decision_num != 'غير محدد':
                # For decisions: deduplicate by decision number
                if decision_num not in seen_decisions:
                    seen_decisions.add(decision_num)
                    
                    # Ensure all metadata fields are present in the chunk
                    chunk = enrich_decision_chunk(chunk, chunks)
                    unique_chunks.append(chunk)
            else:
                # For guides: always add
                unique_chunks.append(chunk)
            
            # Stop when we have enough unique chunks
            if len(unique_chunks) >= top_k:
                break
        
        return unique_chunks
        
    except Exception as e:
        st.error(f"حدث خطأ أثناء استرجاع المقاطع: {e}")
        import traceback
        st.error(traceback.format_exc())
        return []

def format_decision_chunk(chunk, index):
    """Format a single decision chunk with all metadata fields"""
    metadata = chunk.get('metadata', {})
    source = safe_get_metadata_value(metadata, 'Source_Filename', safe_get_metadata_value(chunk, 'filename', 'غير محدد'))
    embedding_source = safe_get_metadata_value(chunk, 'embedding_source', 'غير محدد')
    
    # Build comprehensive context with ALL fields
    context_parts = [f"المقطع المرجعي {index} من القرارات - الملف: {source}"]
    
    # Add all metadata fields with proper handling of empty values
    decision_num = safe_get_metadata_value(metadata, 'رقم القرار')
    if decision_num and decision_num != 'غير محدد':
        context_parts.append(f"رقم القرار: {decision_num}")
    file_name = safe_get_metadata_value(metadata, 'Source_Filename')
    if file_name and file_name != 'غير محدد':
        context_parts.append(f"اسم القرار: {file_name}")
    case_num = safe_get_metadata_value(metadata, 'رقم الدعوى / الاستْناف')
    if case_num and case_num != 'غير محدد':
        context_parts.append(f"رقم الدعوى / الاستْناف: {case_num}")
    
    tax_year = safe_get_metadata_value(metadata, 'العام الضريبي المرتبط بالقرار')
    if tax_year and tax_year != 'غير محدد':
        context_parts.append(f"العام الضريبي المرتبط بالقرار: {tax_year}")
    
    decision_detail = safe_get_metadata_value(metadata, 'تفصيل القرار')
    if decision_detail and decision_detail != 'غير محدد':
        context_parts.append(f"تفصيل القرار: {decision_detail}")
    
    # Add the three main content fields with clear labels
    reasons = safe_get_metadata_value(metadata, 'اسباب القرار')
    if reasons and reasons != 'غير محدد':
        context_parts.append(f"\nاسباب القرار:\n{reasons}")
    
    objections = safe_get_metadata_value(metadata, 'البنود محل الاعتراض')
    if objections and objections != 'غير محدد':
        context_parts.append(f"\nالبنود محل الاعتراض:\n{objections}")
    
    verdict = safe_get_metadata_value(metadata, 'منطوق القرار')
    if verdict and verdict != 'غير محدد':
        context_parts.append(f"\nمنطوق القرار:\n{verdict}")
    
    return '\n'.join(context_parts)

def format_guide_chunk(chunk, index):
    """Format a single guide chunk"""
    metadata = chunk.get('metadata', {})
    source = safe_get_metadata_value(metadata, 'filename', safe_get_metadata_value(chunk, 'filename', 'غير محدد'))
    
    context_parts = [
        f"المقطع المرجعي {index} من الأدلة الإرشادية - الملف: {source}",
        f"المحتوى: {chunk.get('text', '')}"
    ]
    
    return '\n'.join(context_parts)

def format_context(retrieved_chunks, source_type):
    """Format retrieved chunks into context string with sources."""
    if not retrieved_chunks:
        return ""
    
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        if source_type == "القرارات":
            context_parts.append(format_decision_chunk(chunk, i))
        else:
            context_parts.append(format_guide_chunk(chunk, i))
    
    return '\n\n'.join(context_parts)

def display_retrieved_chunks(chunks, source_type):
    """Display retrieved chunks in the sidebar with complete information"""
    if not chunks:
        st.sidebar.info(f"لم يتم العثور على مقاطع من {source_type}")
        return
    
    st.sidebar.subheader(f"المقاطع المسترجعة من {source_type} ({len(chunks)} مقاطع فريدة)")
    
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get('metadata', {})
        
        if source_type == "القرارات":
            # For decisions, show decision number and all available info
            decision_num = safe_get_metadata_value(metadata, 'رقم القرار', 'غير محدد')
            embedding_src = safe_get_metadata_value(chunk, 'embedding_source', 'غير محدد')
            source = safe_get_metadata_value(metadata, 'Source_Filename', safe_get_metadata_value(chunk, 'filename', 'غير محدد'))
            score = chunk.get('similarity_score', 0.0)
            tax_year = safe_get_metadata_value(metadata, 'العام الضريبي المرتبط بالقرار', 'غير محدد')
            
            with st.sidebar.expander(f"مقطع {i} - قرار {decision_num} ({tax_year}) - درجة: {score:.3f}"):
                st.write(f"**رقم القرار:** {decision_num}")
                st.write(f"**رقم الدعوى:** {safe_get_metadata_value(metadata, 'رقم الدعوى / الاستْناف', 'غير محدد')}")
                st.write(f"**العام الضريبي:** {tax_year}")
                st.write(f"**حقل البحث:** {embedding_src}")
                st.write(f"**المصدر:** {source}")
                
                # Show content preview
                st.write("---")
                text_preview = chunk.get('text', '')
                if text_preview:
                    st.text(text_preview[:300] + "..." if len(text_preview) > 300 else text_preview)
                
                # Show if metadata is complete
                required_fields = ['رقم القرار', 'اسباب القرار', 'البنود محل الاعتراض', 'منطوق القرار']
                complete_fields = sum(1 for field in required_fields 
                                     if safe_get_metadata_value(metadata, field) and 
                                     safe_get_metadata_value(metadata, field) != 'غير محدد')
                st.write(f"**اكتمال البيانات:** {complete_fields}/{len(required_fields)} حقول")
        else:
            # For guides
            source = safe_get_metadata_value(metadata, 'filename', safe_get_metadata_value(chunk, 'filename', 'غير محدد'))
            score = chunk.get('similarity_score', 0.0)
            
            with st.sidebar.expander(f"مقطع {i} - {source} (درجة: {score:.3f})"):
                text_preview = chunk.get('text', '')
                st.text(text_preview[:300] + "..." if len(text_preview) > 300 else text_preview)

def summarize_decisions_batch(decisions_batch, query, model, batch_num):
    """Summarize a batch of decisions related to the query"""
    if not decisions_batch:
        return ""
    
    # Format the decisions batch
    formatted_decisions = []
    for i, chunk in enumerate(decisions_batch, 1):
        # print 1st chunk
        if i == 1:
            print('1st decision chunk:', chunk)
        formatted_decisions.append(format_decision_chunk(chunk, i))
    decisions_text = '\n\n'.join(formatted_decisions)
    
    # Create summarization prompt
    summarization_prompt = f"""أنت مساعد متخصص في تحليل القرارات الضريبية والزكوية.

مهمتك: استخرج وقم بتلخيص المعلومات ذات الصلة بسؤال المستخدم من القرارات التالية.

السؤال: {query}

القرارات المتاحة:
{decisions_text}

التعليمات:
1. استخرج فقط المعلومات المرتبطة مباشرة بالسؤال
2. احتفظ بجميع التفاصيل المهمة (رقم القرار، أسباب القرار، البنود محل الاعتراض، منطوق القرار و تفصيل القرار واسم الملف )
3. إذا لم تكن هناك معلومات ذات صلة، اذكر ذلك بوضوح
4. قدم الملخص بشكل منظم وواضح
5. لا تضيف معلومات غير موجودة في القرارات

الملخص المستخرج:"""
    
    try:
        response = model.generate_content(summarization_prompt)
        return f"=== نتائج المجموعة {batch_num} ===\n{response.text}\n"
    except Exception as e:
        return f"=== خطأ في معالجة المجموعة {batch_num} ===\n{str(e)}\n"

async def summarize_decisions_async(decision_chunks, query, model):
    """Split decisions into batches and summarize them asynchronously"""
    if not decision_chunks:
        return ""
    
    # Split into 2 halves
    mid_point = len(decision_chunks) // 2
    batch1 = decision_chunks[:mid_point]
    batch2 = decision_chunks[mid_point:]
    
    # Use ThreadPoolExecutor for async processing
    with ThreadPoolExecutor(max_workers=2) as executor:
        loop = asyncio.get_event_loop()
        
        # Submit both batches for processing
        future1 = loop.run_in_executor(
            executor,
            summarize_decisions_batch,
            batch1, query, model, 1
        )
        future2 = loop.run_in_executor(
            executor,
            summarize_decisions_batch,
            batch2, query, model, 2
        )
        
        # Wait for both to complete
        summary1, summary2 = await asyncio.gather(future1, future2)
    
    # Combine summaries
    combined_summary = f"{summary1}\n{summary2}"
    return combined_summary

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 مساعد المستندات المالية والزكوية</h1>', unsafe_allow_html=True)
    
    # Load models and data
    with st.spinner("جاري تحميل النماذج والبيانات..."):
        model, guides_index, guides_chunks, decisions_index, decisions_chunks, api_key = load_models_and_data()
    
    if model is None:
        st.error("فشل في تحميل النماذج. تحقق من مفتاح API والملفات المطلوبة.")
        return
    
    # Sidebar information
    st.sidebar.header("معلومات النظام")
    st.sidebar.info(f"📚 الأدلة المحملة: {len(guides_chunks)} مقطع")
    st.sidebar.info(f"⚖️ القرارات المحملة: {len(decisions_chunks)} مقطع")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    
    if "top_k_guides" not in st.session_state:
        st.session_state.top_k_guides = 4
    
    if "top_k_decisions" not in st.session_state:
        st.session_state.top_k_decisions = 12
    
    # Settings
    st.sidebar.header("إعدادات")
    show_sources = st.sidebar.checkbox("عرض المصادر", value=st.session_state.show_sources)
    st.session_state.show_sources = show_sources
    
    # Separate sliders for each source type
    st.sidebar.subheader("⚙️ إعدادات الاسترجاع")
    
    top_k_guides = st.sidebar.slider(
        "📚 عدد المقاطع من الأدلة الإرشادية", 
        min_value=1, 
        max_value=10, 
        value=st.session_state.top_k_guides,
        help="عدد المقاطع التي سيتم استرجاعها من الأدلة الإرشادية"
    )
    st.session_state.top_k_guides = top_k_guides
    
    top_k_decisions = st.sidebar.slider(
        "⚖️ عدد المقاطع من القرارات", 
        min_value=1, 
        max_value=20, 
        value=st.session_state.top_k_decisions,
        help="عدد المقاطع التي سيتم استرجاعها من القرارات"
    )
    st.session_state.top_k_decisions = top_k_decisions
    
    # Display total chunks
    total_chunks = top_k_guides + top_k_decisions
    st.sidebar.info(f"📊 إجمالي المقاطع المسترجعة: {total_chunks}")
    st.sidebar.markdown("---")
    
    # System prompt section
    st.sidebar.header("تخصيص التعليمات")
    
    # Initialize default prompt in session state
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = """✨ إرشادات للإجابة
        1.  *المصادر المرجعية*: استخرج الإجابة فقط من المصادر المرجعية (أدلة أو قرارات أو غيرها) مع تصحيح أي أخطاء كتابية إن وجدت.
        2.  *تنسيق الإجابة المطلوب*:
            يجب أن تكون الإجابة بالتنسيق التالي بدقة:
            المصدر الأول:
            اسم المصدر: [اسم الدليل أو المستند مع رقم الصفحة التي تتضمن المعلومة، أو في حال كان قرار استخرج رقم القرار النهائي من داخل المصدر مع اسم الدائرة المصدرة له وتجاهل رقم الدعوى]
            ملخص ما جاء في المصدر: [شرح ملخص للمعلومات الموجودة]
            رابط المصدر: [الرابط إن وجد، أو "غير متوفر"]
            [وهكذا لجميع المصادر]
        3.  *ترتيب المصادر*: ابدأ بالأدلة ثم القرارات، ورتبهم حسب الأهمية.
        4.  *الشرح التفصيلي*: في قسم "ملخص ما جاء في المصدر" اشرح بالتفصيل:
        في حال كان المصدر عبارة عن دليل إرشادي أشرح كالتالي:
        * شرح تفصيلي لمحتوى المصدر المتعلق بالسؤال، مع عدم ذكر أي مبالغ تتعلق بأمثلة مذكورة في المصدر.    
        في حال كان المصدر عبارة عن قرار في دعوى أشرح كالتالي:
        * قم بسرد تفصيل لقرار (اللجنة الابتدائية) في البداية وأذكر اسم اللجنة من واقع القرار.
        * قم بسرد تفصيل لقرار (اللجنة النهائية) في البداية وأذكر اسم اللجنة من واقع القرار.
        **حاول في السرد أن يكون الشرح محدد فقط بمعلومات من المصادر دون الخروج أو إعطاء نتيجة عامة أو نتيجة غير موجودة في المصادر، مع عدم ذكر أي مبالغ ترتبط بالقرار.
        **قم بعرض مصادر إضافية بنفس تنسيق الشرح التفصيلي تتضمن بنود مشابهه للحالة إن توفرت حتى وإن كانت ليست بنفس المسمى المعروض في السؤال.
        5.  *غياب المعلومات*: إذا لم تحتوِ المقاطع المرجعية على إجابة كافية، اكتب:
            > "يظهر أن البند المطلوب لا توجد عنه معلومات كافية في قاعدة المعلومات، حاول أن تجرب تفصيل آخر أو مسمى آخر للبند"
        6.  *اللغة*: استخدم اللغة العربية فقط.
        7.  *التفصيل*: اشرح كل مصدر على حدة.
        8.  *التنسيق*: قم بعرض التنسيق التالي: 
        الصف الأول: رقم المصدر.
        الصف الثاني: اسم المصدر.
        الصف الثالث: ملخص ما جاء في المصدر بتنسيق جذاب ومرتب.
        الصف الأخير: رابط المصدر"""
    
    # Create expandable section for prompt editing
    with st.sidebar.expander("✏️ تعديل التعليمات النظامية", expanded=False):
        custom_prompt = st.text_area(
            "التعليمات النظامية:",
            value=st.session_state.system_prompt,
            height=300,
            help="قم بتخصيص التعليمات التي يتبعها المساعد عند الإجابة على الأسئلة"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 حفظ", use_container_width=True):
                st.session_state.system_prompt = custom_prompt
                st.success("تم حفظ التعليمات!")
        
        with col2:
            if st.button("🔄 استعادة الافتراضي", use_container_width=True):
                st.session_state.system_prompt = """✨ إرشادات للإجابة
                1.  *المصادر المرجعية*: استخرج الإجابة فقط من المصادر المرجعية (أدلة أو قرارات أو غيرها) مع تصحيح أي أخطاء كتابية إن وجدت.
                2.  *تنسيق الإجابة المطلوب*:
                    يجب أن تكون الإجابة بالتنسيق التالي بدقة:
                    المصدر الأول:
                    اسم المصدر: [اسم الدليل أو المستند مع رقم الصفحة التي تتضمن المعلومة، أو في حال كان قرار استخرج رقم القرار النهائي من داخل المصدر مع اسم الدائرة المصدرة له وتجاهل رقم الدعوى]
                    ملخص ما جاء في المصدر: [شرح ملخص للمعلومات الموجودة]
                    رابط المصدر: [الرابط إن وجد، أو "غير متوفر"]
                    [وهكذا لجميع المصادر]
                3.  *ترتيب المصادر*: ابدأ بالأدلة ثم القرارات، ورتبهم حسب الأهمية.
                4.  *الشرح التفصيلي*: في قسم "ملخص ما جاء في المصدر" اشرح بالتفصيل:
                في حال كان المصدر عبارة عن دليل إرشادي أشرح كالتالي:
                * شرح تفصيلي لمحتوى المصدر المتعلق بالسؤال، مع عدم ذكر أي مبالغ تتعلق بأمثلة مذكورة في المصدر.    
                في حال كان المصدر عبارة عن قرار في دعوى أشرح كالتالي:
                * قم بسرد تفصيل لقرار (اللجنة الابتدائية) في البداية وأذكر اسم اللجنة من واقع القرار.
                * قم بسرد تفصيل لقرار (اللجنة النهائية) في البداية وأذكر اسم اللجنة من واقع القرار.
                **حاول في السرد أن يكون الشرح محدد فقط بمعلومات من المصادر دون الخروج أو إعطاء نتيجة عامة أو نتيجة غير موجودة في المصادر، مع عدم ذكر أي مبالغ ترتبط بالقرار.
                **قم بعرض مصادر إضافية بنفس تنسيق الشرح التفصيلي تتضمن بنود مشابهه للحالة إن توفرت حتى وإن كانت ليست بنفس المسمى المعروض في السؤال.
                5.  *غياب المعلومات*: إذا لم تحتوِ المقاطع المرجعية على إجابة كافية، اكتب:
                    > "يظهر أن البند المطلوب لا توجد عنه معلومات كافية في قاعدة المعلومات، حاول أن تجرب تفصيل آخر أو مسمى آخر للبند"
                6.  *اللغة*: استخدم اللغة العربية فقط.
                7.  *التفصيل*: اشرح كل مصدر على حدة.
                8.  *التنسيق*: قم بعرض التنسيق التالي: 
                الصف الأول: رقم المصدر.
                الصف الثاني: اسم المصدر.
                الصف الثالث: ملخص ما جاء في المصدر بتنسيق جذاب ومرتب.
                الصف الأخير: رابط المصدر"""
                st.rerun()
    
    # Clear chat button
    if st.sidebar.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>أنت:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>المساعد:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    query = st.chat_input("اسأل سؤالاً عن المستندات المالية والزكوية...")
    
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message immediately
        st.markdown(f'<div class="chat-message user-message"><strong>أنت:</strong> {query}</div>', unsafe_allow_html=True)
        
        try:
            with st.spinner("جاري البحث وتحليل المستندات..."):
                # Get query embedding
                q_embedding = get_embedding(query, api_key)
                
                # Retrieve chunks
                guide_retrieved = []
                if guides_index and len(guides_chunks) > 0:
                    guide_retrieved = retrieve_chunks(q_embedding, guides_index, guides_chunks, top_k=st.session_state.top_k_guides)
                
                decision_retrieved = []
                decision_summary = ""
                if decisions_index and len(decisions_chunks) > 0:
                    decision_retrieved = retrieve_chunks(q_embedding, decisions_index, decisions_chunks, top_k=st.session_state.top_k_decisions)
                    print('decision_retrieved', len(decision_retrieved))
                    # call summarize function
                    if decision_retrieved:
                        decision_summary = asyncio.run(summarize_decisions_async(decision_retrieved, query, model))
                        print('decision_summary length:', len(decision_summary))
                
                # Display retrieved chunks in sidebar if enabled
                if show_sources:
                    st.sidebar.markdown("---")
                    st.sidebar.subheader("المقاطع المسترجعة للسؤال الحالي")
                    display_retrieved_chunks(guide_retrieved, "الأدلة الإرشادية")
                    display_retrieved_chunks(decision_retrieved, "القرارات")
                
                # Format context with ALL metadata
                guides_context = format_context(guide_retrieved, "الأدلة الإرشادية")
                
                # Use decision_summary instead of formatting decisions again
                if decision_summary:
                    full_context = f"الأدلة الإرشادية:\n{guides_context}\n\nالقرارات:\n{decision_summary}".strip()
                else:
                    full_context = f"الأدلة الإرشادية:\n{guides_context}".strip()
                
                # Debug: print to console to verify all fields are included
                print("=" * 80)
                print("DECISIONS SUMMARY:")
                print(decision_summary if decision_summary else "No decisions")
                print("=" * 80)
                
                if not full_context:
                    response = "عذراً، لم يتم العثور على مقاطع ذات صلة بسؤالك."
                else:
                    # Create prompt using the customizable system prompt
                    prompt = f"""أنت مساعد ذكي متخصص في فهم وتحليل المستندات المالية والزكوية العربية.
يجب أن تعتمد إجاباتك فقط على المعلومات الموجودة في المقاطع المرجعية أدناه.

{st.session_state.system_prompt}

المقاطع المرجعية:
{full_context}

السؤال: {query}

الإجابة:"""
                    
                    # Generate response
                    response = model.generate_content(prompt).text
            
            # Display assistant response
            st.markdown(f'<div class="chat-message assistant-message"><strong>المساعد:</strong> {response}</div>', unsafe_allow_html=True)
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            import traceback
            error_message = f"حدث خطأ أثناء معالجة السؤال: {str(e)}\n\n{traceback.format_exc()}"
            st.markdown(f'<div class="error-message">{error_message}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": f"حدث خطأ: {str(e)}"})
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="footer">'
        '🤖 مساعد المستندات المالية والزكوية - مدعوم بالذكاء الاصطناعي'
        '</div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()