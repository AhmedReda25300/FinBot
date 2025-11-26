import streamlit as st
import pickle
import google.generativeai as genai
import faiss
import numpy as np
from dotenv import load_dotenv
import os

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
    """Load the FAISS indices and chunks data for all document types"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("مطلوب مفتاح API. تأكد من وجود GOOGLE_API_KEY في ملف .env")
        return None, None, {}, {}
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    indices = {}
    chunks = {}

    file_map = {
        "publications": ("publications_new_1112.faiss", "publications_chunks_new_1112.pkl"),
        "laws": ("laws_new_1112.faiss", "laws_chunks_new_1112.pkl"),
        "guidelines": ("guidelines_new_1112.faiss", "guidelines_chunks_new_1112.pkl"),
        "decisions": ("decisions_new_1112.faiss", "decisions_chunks_new_1112.pkl")
    }
    
    for doc_type, (faiss_file, pkl_file) in file_map.items():
        try:
            indices[doc_type] = faiss.read_index(faiss_file)
            with open(pkl_file, 'rb') as f:
                chunks[doc_type] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"لم يتم العثور على ملفات {doc_type}.")
            indices[doc_type] = None
            chunks[doc_type] = []
            
    return model, api_key, indices, chunks

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

def retrieve_chunks(query_embedding, index, chunks, top_k=5):
    """Retrieve top_k most similar unique chunks to the query."""
    if not index or not chunks or top_k == 0:
        return []

    try:
        search_k = min(top_k * 2, len(chunks)) # Retrieve more to filter duplicates
        distances, indices = index.search(query_embedding.reshape(1, -1), search_k)
        
        retrieved_chunks = []
        seen_content = set()
        
        for i, idx in enumerate(indices[0]):
            chunk = chunks[idx].copy()
            # Similarity score: 1 is perfect match, 0 is distant. Assumes normalized vectors.
            chunk['similarity_score'] = (2 - distances[0][i]**2) / 2

            # Use content for deduplication to avoid showing very similar chunks
            content_key = chunk.get('content', chunk.get('text', ''))
            if content_key in seen_content:
                continue
            
            seen_content.add(content_key)
            retrieved_chunks.append(chunk)
            
            if len(retrieved_chunks) >= top_k:
                break
        
        return retrieved_chunks
    except Exception as e:
        st.error(f"حدث خطأ أثناء استرجاع المقاطع: {e}")
        return []

def format_decision_chunk(chunk, index):
    """Format a single decision chunk with all metadata fields"""
    try:
        metadata = chunk.get('metadata', {})
        source = metadata.get('Source_Filename', chunk.get('filename', 'غير محدد'))
        decision_details = metadata.get('تفصيل_القرار', {})
        
        context_parts = [f"المقطع المرجعي {index} من القرارات - الملف: {source}"]
        
        if decision_details.get('رقم_القرار_النهائي'):
            context_parts.append(f"رقم القرار النهائي: {decision_details['رقم_القرار_النهائي']}")
        if decision_details.get('اسم_الدائرة_النهائية'):
            context_parts.append(f"الدائرة النهائية: {decision_details['اسم_الدائرة_النهائية']}")
        
        items = decision_details.get('البنود_محل_الدعوى', [])
        if items:
            context_parts.append("\nالبنود محل الدعوى:")
            for item in items:
                context_parts.append(f"- {item.get('اسم_البند', 'بند غير محدد')}: {item.get('نبذة_مختصرة_عن_الاعتراض', '')}")
        
        return '\n'.join(context_parts)
    except Exception as e:
        return f"[خطأ في معالجة القرار {index}: {str(e)}]"

def format_document_chunk(chunk, index, source_type):
    """Format a single chunk from publications, laws, or guidelines"""
    try:
        metadata = chunk.get('metadata', {})
        doc_title = metadata.get('document_title', chunk.get('document_title', 'غير محدد'))
        chunk_title = metadata.get('chunk_title', chunk.get('chunk_title', ''))
        content = chunk.get('content', chunk.get('text', '[محتوى المقطع غير متوفر]'))
        
        context_parts = [f"المقطع المرجعي {index} من {source_type} - المستند: {doc_title}"]
        if chunk_title:
            context_parts.append(f"عنوان المقطع: {chunk_title}")
        context_parts.append(f"المحتوى: {content}")
        
        return '\n'.join(filter(None, context_parts))
    except Exception as e:
        return f"[خطأ في معالجة المقطع {index} من {source_type}: {str(e)}]"

def format_context(retrieved_chunks, source_type):
    """Format retrieved chunks into a context string."""
    if not retrieved_chunks:
        return ""
    
    context_parts = []
    is_decision = source_type == "القرارات"
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        if is_decision:
            context_parts.append(format_decision_chunk(chunk, i))
        else:
            context_parts.append(format_document_chunk(chunk, i, source_type))
    
    return '\n\n'.join(context_parts)

def display_retrieved_chunks(chunks, source_type):
    """Display retrieved chunks in the sidebar"""
    if not chunks:
        st.sidebar.info(f"لم يتم العثور على مقاطع من {source_type}")
        return
    
    st.sidebar.subheader(f"المقاطع المسترجعة من {source_type}")
    
    for i, chunk in enumerate(chunks, 1):
        try:
            score = chunk.get('similarity_score', 0.0)
            metadata = chunk.get('metadata', {})
            is_decision = source_type == "القرارات"

            title = 'مستند غير محدد'
            content_preview = ''

            if is_decision:
                decision_details = metadata.get('تفصيل_القرار', {})
                decision_num = decision_details.get('رقم_القرار_النهائي', 'غير محدد')
                title = f"قرار {decision_num}"
                content_preview = chunk.get('text', '')
            else: # Publications, laws, guidelines
                doc_title = metadata.get('document_title', chunk.get('document_title', 'مستند غير محدد'))
                chunk_title = metadata.get('chunk_title', chunk.get('chunk_title', ''))
                title = f"{doc_title} - {chunk_title}" if chunk_title and doc_title != 'مستند غير محدد' else doc_title
                content_preview = chunk.get('content', chunk.get('text', ''))

            expander_title = title if title != 'مستند غير محدد' else f"مقطع {i} من {source_type}"
            with st.sidebar.expander(f"{expander_title[:60]}... (التشابه: {score:.3f})"):
                st.markdown(f"**المصدر:** {title}")
                st.markdown(f"**نوع المستند:** {source_type}")
                st.markdown(f"**درجة التشابه:** {score:.3f}")
                st.markdown("---")
                if content_preview:
                    st.text(content_preview[:400] + "..." if len(content_preview) > 400 else content_preview)
                else:
                    st.text("[معاينة النص غير متوفرة]")
        except Exception as e:
            st.sidebar.error(f"خطأ في عرض المقطع {i}: {str(e)}")

def main():
    st.markdown('<h1 class="main-header">🤖 مساعد المستندات المالية والزكوية</h1>', unsafe_allow_html=True)
    
    with st.spinner("جاري تحميل النماذج والبيانات..."):
        model, api_key, indices, chunks_data = load_models_and_data()
    
    if model is None:
        st.error("فشل في تحميل النماذج. تحقق من مفتاح API والملفات المطلوبة.")
        return
    
    st.sidebar.header("معلومات النظام")
    st.sidebar.info(f"📚 المنشورات: {len(chunks_data.get('publications', []))} مقطع")
    st.sidebar.info(f"⚖️ الأنظمة: {len(chunks_data.get('laws', []))} مقطع")
    st.sidebar.info(f"📑 الأدلة الإرشادية: {len(chunks_data.get('guidelines', []))} مقطع")
    st.sidebar.info(f"👨‍⚖️ القرارات: {len(chunks_data.get('decisions', []))} مقطع")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    if "show_sources" not in st.session_state: st.session_state.show_sources = True
    
    defaults = {
        "top_k_publications": 3, "top_k_laws": 3,
        "top_k_guidelines": 4, "top_k_decisions": 5
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

    st.sidebar.header("إعدادات")
    st.session_state.show_sources = st.sidebar.checkbox("عرض المصادر المسترجعة", value=st.session_state.show_sources)
    st.sidebar.subheader("⚙️ إعدادات الاسترجاع")

    st.session_state.top_k_publications = st.sidebar.slider("📚 عدد المقاطع من المنشورات", 0, 15, st.session_state.top_k_publications)
    st.session_state.top_k_laws = st.sidebar.slider("⚖️ عدد المقاطع من الأنظمة", 0, 15, st.session_state.top_k_laws)
    st.session_state.top_k_guidelines = st.sidebar.slider("📑 عدد المقاطع من الأدلة الإرشادية", 0, 15, st.session_state.top_k_guidelines)
    st.session_state.top_k_decisions = st.sidebar.slider("👨‍⚖️ عدد المقاطع من القرارات", 0, 15, st.session_state.top_k_decisions)
    
    total_chunks = sum(st.session_state[k] for k in defaults)
    st.sidebar.info(f"📊 إجمالي المقاطع المسترجعة: {total_chunks}")
    st.sidebar.markdown("---")

    # (Your existing system prompt and expander logic should be here)
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = """إرشادات للإجابة: -
1-*المطلوب*: 
... (ضع التعليمات الطويلة الكاملة هنا) ..."""

    with st.sidebar.expander("✏️ تعديل التعليمات النظامية", expanded=False):
        # (Your prompt editing UI logic here)
        pass


    if st.sidebar.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        role_name = "أنت" if message["role"] == "user" else "المساعد"
        st.markdown(f'<div class="chat-message {role_class}"><strong>{role_name}:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    if query := st.chat_input("اسأل سؤالاً عن المستندات المالية والزكوية..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.markdown(f'<div class="chat-message user-message"><strong>أنت:</strong> {query}</div>', unsafe_allow_html=True)
        
        try:
            with st.spinner("جاري البحث وتحليل المستندات..."):
                q_embedding = get_embedding(query, api_key)
                
                retrieved_data = {
                    "المنشورات": retrieve_chunks(q_embedding, indices['publications'], chunks_data['publications'], st.session_state.top_k_publications),
                    "الأنظمة": retrieve_chunks(q_embedding, indices['laws'], chunks_data['laws'], st.session_state.top_k_laws),
                    "الأدلة الإرشادية": retrieve_chunks(q_embedding, indices['guidelines'], chunks_data['guidelines'], st.session_state.top_k_guidelines),
                    "القرارات": retrieve_chunks(q_embedding, indices['decisions'], chunks_data['decisions'], st.session_state.top_k_decisions)
                }

                if st.session_state.show_sources:
                    st.sidebar.markdown("---")
                    st.sidebar.subheader("المقاطع المسترجعة للسؤال الحالي")
                    for source_type, chunks in retrieved_data.items():
                        display_retrieved_chunks(chunks, source_type)
                
                contexts = [format_context(chunks, name) for name, chunks in retrieved_data.items()]
                full_context = "\n\n".join(filter(None, contexts))
                
                if not full_context.strip():
                    response = "عذراً، لم يتم العثور على معلومات ذات صلة بسؤالك في المستندات المتاحة."
                else:
                    prompt = f"{st.session_state.system_prompt}\n\nالمقاطع المرجعية:\n{full_context}\n\nالسؤال: {query}\n\nالإجابة:"
                    response = model.generate_content(prompt).text
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(f'<div class="chat-message assistant-message"><strong>المساعد:</strong> {response}</div>', unsafe_allow_html=True)
            
        except Exception as e:
            error_message = f"حدث خطأ غير متوقع: {str(e)}"
            st.markdown(f'<div class="error-message">{error_message}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    st.markdown("---")
    st.markdown('<div class="footer">🤖 مساعد المستندات المالية والزكوية - مدعوم بالذكاء الاصطناعي</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
