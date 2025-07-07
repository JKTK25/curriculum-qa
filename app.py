import os
import csv
import tempfile
import streamlit as st
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import (
    PyMuPDFLoader, Docx2txtLoader, TextLoader, JSONLoader
)
from langchain_community.vectorstores import FAISS  # ✅ updated
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import openai

# ---------------- App Config ----------------
st.set_page_config(page_title="📚 Curriculum Chatbot", layout="wide")
st.title("📚 Curriculum Chatbot with Memory")
st.markdown("Upload your curriculum files and ask questions — now with chat-style memory!")

# ---------------- Optional Password Gate ----------------
REQUIRE_PASSWORD = False  # Set to True to enable password gate
APP_PASSWORD = "letmein"  # You can change this password

if REQUIRE_PASSWORD:
    pw = st.text_input("🔒 Enter App Password", type="password")
    if pw != APP_PASSWORD:
        st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("🔐 DeepSeek API Key")
    api_key = st.text_input("Paste your key here", type="password")

    uploaded_files = st.file_uploader("📁 Upload curriculum files", accept_multiple_files=True)
    if st.button("🔄 Reset Conversation"):
        st.session_state.clear()
        st.experimental_rerun()

# ---------------- Session State ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- Start Processing ----------------
if api_key and uploaded_files:
    with st.spinner("🧠 Setting up chatbot..."):

        # Set DeepSeek via OpenAI-style API
        openai.api_base = "https://api.deepseek.com/v1"
        openai.api_key = api_key
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.3
        )

        # Save uploaded files to temp dir
        temp_dir = tempfile.mkdtemp()
        all_files = []
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            all_files.append(file_path)

        # Load documents
        supported_exts = {".pdf", ".docx", ".txt", ".jsonl"}
        all_docs = []
        for file_path in tqdm(all_files, disable=True):
            ext = os.path.splitext(file_path)[1].lower()
            try:
                if ext == ".pdf":
                    loader = PyMuPDFLoader(file_path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                elif ext == ".txt":
                    loader = TextLoader(file_path)
                elif ext == ".jsonl":
                    loader = JSONLoader(file_path, text_key="text")
                else:
                    continue
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                st.warning(f"⚠️ Could not load {file_path}: {e}")

        if not all_docs:
            st.error("❌ No documents were successfully loaded.")
            st.stop()

        # Split and embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(all_docs)

        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Memory + Retrieval Chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False
        )

        st.session_state.qa_chain = qa_chain

    st.success("✅ Chatbot is ready! Ask your curriculum questions below.")

# ---------------- Chat UI ----------------
if "qa_chain" in st.session_state:
    user_input = st.chat_input("💬 Ask something about your curriculum")
    if user_input:
        with st.spinner("💡 Thinking..."):
            try:
                result = st.session_state.qa_chain({"question": user_input})
                answer = result["answer"]

                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", answer))

                # Save to CSV
                if not os.path.exists("qa_log.csv"):
                    with open("qa_log.csv", "w", newline='', encoding="utf-8") as f:
                        csv.writer(f).writerow(["Question", "Answer"])
                with open("qa_log.csv", "a", newline='', encoding="utf-8") as f:
                    csv.writer(f).writerow([user_input, answer])

            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Show full conversation
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(text)
        else:
            st.chat_message("assistant").markdown(text)
else:
    st.info("⬅️ Enter your API key and upload documents to begin.")
