import os
import csv
import tempfile
import streamlit as st
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import (
    PyMuPDFLoader, Docx2txtLoader, TextLoader, JSONLoader
)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import openai

# ----------- UI CONFIG -----------
st.set_page_config(page_title="📚 Curriculum Chatbot", layout="centered")
st.markdown("""
    <style>
        .block-container {padding-top: 2rem;}
        .stChatMessage.user {text-align: right;}
        .stChatMessage.user .stMarkdown {
            background-color: #DCF8C6; padding: 0.8rem 1rem;
            border-radius: 10px; display: inline-block;
            max-width: 80%;
        }
        .stChatMessage.assistant .stMarkdown {
            background-color: #F1F0F0; padding: 0.8rem 1rem;
            border-radius: 10px; display: inline-block;
            max-width: 80%;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Curriculum Chatbot")

# ----------- SIDEBAR -----------
with st.sidebar:
    st.subheader("🔐 API & Upload")
    api_key = st.text_input("DeepSeek API Key", type="password")
    uploaded_files = st.file_uploader("📁 Upload curriculum files", accept_multiple_files=True)
    if st.button("🔄 Reset Chat"):
        st.session_state.clear()
        st.experimental_rerun()

# ----------- SESSION MEMORY -----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------- SETUP -----------
if api_key and uploaded_files:
    if "qa_chain" not in st.session_state:
        with st.spinner("🧠 Initializing chatbot..."):
            openai.api_base = "https://api.deepseek.com/v1"
            openai.api_key = api_key
            llm = ChatOpenAI(
                model="deepseek-chat",
                openai_api_key=api_key,
                openai_api_base="https://api.deepseek.com/v1",
                temperature=0.3
            )

            temp_dir = tempfile.mkdtemp()
            all_docs = []
            for file in uploaded_files:
                path = os.path.join(temp_dir, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                ext = os.path.splitext(path)[1].lower()
                try:
                    if ext == ".pdf":
                        loader = PyMuPDFLoader(path)
                    elif ext == ".docx":
                        loader = Docx2txtLoader(path)
                    elif ext == ".txt":
                        loader = TextLoader(path)
                    elif ext == ".jsonl":
                        loader = JSONLoader(path, text_key="text")
                    else:
                        continue
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as e:
                    st.warning(f"⚠️ Could not load {file.name}: {e}")

            if not all_docs:
                st.error("❌ No documents loaded.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(all_docs)

            SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embeddings)

            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

            # ✅ Classic memory setup (no ChatMessageHistory)
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory
            )

            st.session_state.qa_chain = chain

# ----------- CHAT LOOP -----------
if "qa_chain" in st.session_state:
    user_input = st.chat_input("💬 Ask a question about your curriculum")
    if user_input:
        with st.spinner("🤖 Thinking..."):
            try:
                result = st.session_state.qa_chain({"question": user_input})
                answer = result["answer"]
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", answer))

                # Save log
                if not os.path.exists("qa_log.csv"):
                    with open("qa_log.csv", "w", newline='', encoding="utf-8") as f:
                        csv.writer(f).writerow(["Question", "Answer"])
                with open("qa_log.csv", "a", newline='', encoding="utf-8") as f:
                    csv.writer(f).writerow([user_input, answer])
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

    # Display messages
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)
else:
    st.info("⬅️ Enter API key and upload files to begin.")
