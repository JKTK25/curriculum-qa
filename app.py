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

# Custom CSS for professional styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.upload-section {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin-bottom: 1rem;
}
.chat-container {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    min-height: 400px;
    max-height: 500px;
    overflow-y: auto;
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
        <div style="text-align: center; padding: 3rem; color: #6c757d;">
            <h3>🌟 Welcome to Your Learning Journey!</h3>
            <p>Upload your curriculum materials above and start asking questions to enhance your understanding.</p>
            <p><em>I'm here to help you learn, explain concepts, and answer your questions!</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Input area with educational styling
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    user_query = st.text_input(
        "Ask your question", 
        placeholder="What would you like to learn about? Ask me anything from your uploaded materials...", 
        label_visibility="collapsed",
        key=f"input_{st.session_state.input_key}",
        help="Type your question and press Send to get AI-powered explanations"
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
