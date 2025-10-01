import streamlit as st
import pickle
import google.generativeai as genai
import faiss
import numpy as np
from dotenv import load_dotenv
import os
from datetime import datetime
import time

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
    model = genai.GenerativeModel('gemini-2.5-pro')
    
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
        decisions_index = faiss.read_index('decisions.faiss')
        with open('decisions_chunks.pkl', 'rb') as f:
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

def retrieve_chunks(query_embedding, index, chunks, top_k=8):
    """Retrieve top_k most similar unique chunks to the query."""
    if not index or not chunks:
        return []

    try:
        distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        # Add similarity scores to the chunks for display
        for i, chunk in enumerate(retrieved_chunks):
            chunk['similarity_score'] = distances[0][i]
        return retrieved_chunks
    except Exception as e:
        st.error(f"حدث خطأ أثناء استرجاع المقاطع: {e}")
        return []

def format_decision_chunk(chunk, index):
    """Format a single decision chunk with all metadata fields"""
    metadata = chunk.get('metadata', {})
    source = metadata.get('Source_Filename', chunk.get('filename', 'غير محدد'))
    embedding_source = chunk.get('embedding_source', 'غير محدد')
    
    # Build comprehensive context with ALL fields
    context_parts = [f"المقطع المرجعي {index} من القرارات - الملف: {source}"]
    
    # Add all metadata fields
    if 'رقم القرار' in metadata:
        context_parts.append(f"رقم القرار: {metadata['رقم القرار']}")
    
    if 'رقم الدعوى / الاستْناف' in metadata:
        context_parts.append(f"رقم الدعوى / الاستْناف: {metadata['رقم الدعوى / الاستْناف']}")
    
    if 'العام الضريبي المرتبط بالقرار' in metadata:
        context_parts.append(f"العام الضريبي المرتبط بالقرار: {metadata['العام الضريبي المرتبط بالقرار']}")
    
    if 'رقم الدعوه' in metadata:
        context_parts.append(f"رقم الدعوه: {metadata['رقم الدعوه']}")
    
    # Add the three main content fields with clear labels
    if 'اسباب القرار' in metadata:
        context_parts.append(f"\nاسباب القرار:\n{metadata['اسباب القرار']}")
    
    if 'البنود محل الاعتراض' in metadata:
        context_parts.append(f"\nالبنود محل الاعتراض:\n{metadata['البنود محل الاعتراض']}")
    
    if 'منطوق القرار' in metadata:
        context_parts.append(f"\nمنطوق القرار:\n{metadata['منطوق القرار']}")
    
    # Add note that all fields are included
    context_parts.append(f"\n[تم العثور على هذا القرار من خلال البحث في: {embedding_source}، والمقطع يتضمن جميع الحقول: اسباب القرار، البنود محل الاعتراض، منطوق القرار]")
    
    return '\n'.join(context_parts)

def format_guide_chunk(chunk, index):
    """Format a single guide chunk"""
    metadata = chunk.get('metadata', {})
    source = metadata.get('filename', chunk.get('filename', 'غير محدد'))
    
    context_parts = [
        f"المقطع المرجعي {index} من الأدلة الإرشادية - الملف: {source}",
        f"المحتوى: {chunk['text']}"
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
    """Display retrieved chunks in the sidebar"""
    if not chunks:
        st.sidebar.info(f"لم يتم العثور على مقاطع من {source_type}")
        return
    
    st.sidebar.subheader(f"المقاطع المسترجعة من {source_type}")
    
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get('metadata', {})
        
        if source_type == "القرارات":
            # For decisions, show decision number and embedding source
            decision_num = metadata.get('رقم القرار', 'غير محدد')
            embedding_src = chunk.get('embedding_source', 'غير محدد')
            source = metadata.get('Source_Filename', chunk.get('filename', 'غير محدد'))
            score = chunk.get('similarity_score', 0.0)
            
            with st.sidebar.expander(f"مقطع {i} - قرار {decision_num} (من: {embedding_src}) - درجة: {score:.3f}"):
                # Show all metadata fields
                st.write(f"**رقم القرار:** {metadata.get('رقم القرار', 'غير محدد')}")
                st.write(f"**العام الضريبي:** {metadata.get('العام الضريبي المرتبط بالقرار', 'غير محدد')}")
                st.write(f"**حقل البحث:** {embedding_src}")
                st.write("---")
                st.text(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
        else:
            # For guides
            source = metadata.get('filename', chunk.get('filename', 'غير محدد'))
            score = chunk.get('similarity_score', 0.0)
            
            with st.sidebar.expander(f"مقطع {i} - {source} (درجة التشابه: {score:.3f})"):
                st.text(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])

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
        st.session_state.top_k_decisions = 8
    
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
        max_value=15, 
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
                if decisions_index and len(decisions_chunks) > 0:
                    decision_retrieved = retrieve_chunks(q_embedding, decisions_index, decisions_chunks, top_k=st.session_state.top_k_decisions)
                
                # Display retrieved chunks in sidebar if enabled
                if show_sources:
                    st.sidebar.markdown("---")
                    st.sidebar.subheader("المقاطع المسترجعة للسؤال الحالي")
                    display_retrieved_chunks(guide_retrieved, "الأدلة الإرشادية")
                    display_retrieved_chunks(decision_retrieved, "القرارات")
                
                # Format context with ALL metadata
                guides_context = format_context(guide_retrieved, "الأدلة الإرشادية")
                decisions_context = format_context(decision_retrieved, "القرارات")
                full_context = f"الأدلة الإرشادية:\n{guides_context}\n\nالقرارات:\n{decisions_context}".strip()
                
                # Debug: print to console to verify all fields are included
                print("=" * 80)
                print("DECISIONS CONTEXT WITH ALL METADATA:")
                print(decisions_context)
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
            error_message = f"حدث خطأ أثناء معالجة السؤال: {str(e)}"
            st.markdown(f'<div class="error-message">{error_message}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
    
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