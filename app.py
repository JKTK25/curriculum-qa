import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader
import docx
import os


# -----------------------------
# Helper functions
# -----------------------------
def extract_text_from_uploaded_files(uploaded_files):
    documents = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            documents.append(text)

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents.append(text)

        else:  # TXT
            documents.append(file.read().decode("utf-8"))

    return documents


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.create_documents(docs)


def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="EduChat - AI Learning Assistant", page_icon="🎓", layout="wide")

# Advanced CSS for professional styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    padding: 2.5rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    animation: fadeInDown 0.8s ease-out;
}

.upload-section {
    background: linear-gradient(145deg, #f8f9fa, #e9ecef);
    padding: 2rem;
    border-radius: 15px;
    border-left: 5px solid #667eea;
    margin-bottom: 1.5rem;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}

.upload-section:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

.chat-container {
    background: linear-gradient(145deg, #ffffff, #f8f9fa);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    min-height: 450px;
    max-height: 550px;
    overflow-y: auto;
    border: 1px solid rgba(102, 126, 234, 0.1);
}

.sidebar-section {
    background: linear-gradient(145deg, #f8f9fa, #ffffff);
    padding: 1rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.message-timestamp {
    font-size: 0.75em;
    opacity: 0.6;
    margin-top: 5px;
}

.typing-indicator {
    display: flex;
    align-items: center;
    padding: 10px 15px;
    background-color: #f1f3f4;
    border-radius: 20px;
    margin: 10px 0;
    max-width: 100px;
}

.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #667eea;
    margin: 0 2px;
    animation: typing 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(1) { animation-delay: -0.32s; }
.typing-dot:nth-child(2) { animation-delay: -0.16s; }

@keyframes typing {
    0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
    40% { transform: scale(1); opacity: 1; }
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(30px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}

.student-message {
    animation: slideInRight 0.3s ease-out;
}

.ai-message {
    animation: slideInLeft 0.3s ease-out;
}

.input-container {
    background: white;
    border-radius: 25px;
    padding: 5px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    border: 2px solid transparent;
    transition: all 0.3s ease;
}

.input-container:focus-within {
    border-color: #667eea;
    box-shadow: 0 4px 25px rgba(102, 126, 234, 0.2);
}

.stButton > button {
    border-radius: 20px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
}

.status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #28a745;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
    100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎓 EduChat - AI Learning Assistant</h1>
    <p>Your intelligent companion for exploring curriculum content and enhancing learning</p>
</div>
""", unsafe_allow_html=True)

# Set API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["DEESEEK_API_KEY"]

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📁 Upload Learning Materials")
st.markdown("*Upload your curriculum documents to start learning with AI assistance*")
uploaded_files = st.file_uploader(
    "Choose your files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Supported formats: PDF, Word documents, and text files"
)
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

# Enhanced sidebar for chat management
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h3 style="color: #667eea; margin-bottom: 0.5rem;">💬 Chat Management</h3>
        <div><span class="status-indicator"></span><small>AI Assistant Online</small></div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("✨ Start New Conversation", use_container_width=True, type="primary"):
        if st.session_state.chat_history:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M")
            st.session_state.chat_sessions.append({
                "history": st.session_state.chat_history.copy(),
                "title": st.session_state.chat_history[0][0][:25] + "..." if st.session_state.chat_history else "Empty Chat",
                "timestamp": timestamp,
                "message_count": len(st.session_state.chat_history)
            })
        st.session_state.chat_history = []
        st.session_state.input_key += 1
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📚 Learning History")
    
    if st.session_state.chat_sessions:
        st.markdown(f"*{len(st.session_state.chat_sessions)} previous conversations*")
        for i, session in enumerate(reversed(st.session_state.chat_sessions)):
            session_idx = len(st.session_state.chat_sessions) - 1 - i
            if st.button(
                f"📝 {session['title']}", 
                key=f"session_{session_idx}", 
                use_container_width=True,
                help=f"Messages: {session.get('message_count', 0)} | Time: {session.get('timestamp', 'Unknown')}"
            ):
                st.session_state.chat_history = session['history'].copy()
                st.rerun()
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #6c757d;">
            <p>🌱 <em>Start your first conversation!</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    docs = extract_text_from_uploaded_files(uploaded_files)
    chunks = split_documents(docs)
    vector_store = build_vector_store(chunks)

    st.success("✅ Documents processed successfully!")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        base_url="https://api.deepseek.com"
    )

    # Create a custom retrieval chain using available components
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create a custom retrieval chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    st.session_state.qa_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Chat interface
st.markdown("### 💬 Learning Conversation")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history with educational styling
if st.session_state.chat_history:
    for question, answer in st.session_state.chat_history:
        # Student question
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin: 15px 0; align-items: flex-start;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 18px; border-radius: 20px 20px 5px 20px; max-width: 75%; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);">
                    <div style="font-weight: 500; margin-bottom: 4px; font-size: 0.9em; opacity: 0.9;">👨‍🎓 Student</div>
                    <div>{question}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # AI tutor response
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin: 15px 0; align-items: flex-start;">
                <div style="background-color: #f8f9fa; color: #2c3e50; padding: 12px 18px; border-radius: 20px 20px 20px 5px; max-width: 75%; border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="font-weight: 500; margin-bottom: 4px; font-size: 0.9em; color: #667eea;">🤖 AI Tutor</div>
                    <div style="line-height: 1.6;">{answer}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.markdown(
        """
        <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(145deg, #f8f9fa, #ffffff); border-radius: 20px; margin: 2rem 0;">
            <div style="font-size: 4em; margin-bottom: 1rem; animation: fadeInDown 1s ease-out;">🎓</div>
            <h2 style="color: #667eea; margin-bottom: 1rem; font-weight: 600;">Welcome to EduChat!</h2>
            <p style="font-size: 1.1em; color: #6c757d; margin-bottom: 1.5rem; line-height: 1.6;">Your intelligent learning companion is ready to help you explore and understand your curriculum materials.</p>
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 1rem 2rem; border-radius: 15px; display: inline-block; margin-bottom: 1rem;">
                <strong>🚀 Getting Started:</strong>
            </div>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap;">
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 200px;">
                    <div style="font-size: 2em; margin-bottom: 0.5rem;">📁</div>
                    <strong>1. Upload</strong><br>
                    <small>Add your documents</small>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 200px;">
                    <div style="font-size: 2em; margin-bottom: 0.5rem;">❓</div>
                    <strong>2. Ask</strong><br>
                    <small>Type your questions</small>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 200px;">
                    <div style="font-size: 2em; margin-bottom: 0.5rem;">🎆</div>
                    <strong>3. Learn</strong><br>
                    <small>Get AI-powered insights</small>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Enhanced input area with professional styling
st.markdown("""
<div style="background: linear-gradient(145deg, #f8f9fa, #ffffff); padding: 1.5rem; border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
    <div style="text-align: center; margin-bottom: 1rem;">
        <h4 style="color: #667eea; margin: 0;">💭 Ask Your Question</h4>
        <small style="color: #6c757d;">Get instant AI-powered explanations from your documents</small>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])
with col1:
    user_query = st.text_input(
        "Question", 
        placeholder="💡 What would you like to learn about? Ask me anything from your uploaded materials...", 
        label_visibility="collapsed",
        key=f"input_{st.session_state.input_key}",
        help="💬 Type your question and click Ask to get AI-powered explanations"
    )
with col2:
    send_button = st.button("🚀 Ask", use_container_width=True, type="primary")

if send_button and user_query:
    if st.session_state.qa_chain is None:
        st.error("Please upload documents first!")
    else:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke(user_query)
        
        st.session_state.chat_history.append((user_query, response))
        st.session_state.input_key += 1
        st.rerun()
