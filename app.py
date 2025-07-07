import os
import csv
import shutil
import tempfile
import streamlit as st
from tqdm import tqdm

from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import openai

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📚 Curriculum QA Tool")
st.title("📚 Curriculum QA with DeepSeek + LangChain")
st.markdown("Upload your curriculum files and ask questions. Supports PDF, DOCX, TXT, and JSONL.")

# ---------------- Input API Key ----------------
api_key = st.text_input("🔑 Enter your DeepSeek API Key", type="password")
uploaded_files = st.file_uploader("📁 Upload your files", accept_multiple_files=True)

# ---------------- Optional Reset Button ----------------
if st.button("🔄 Reset App"):
    st.cache_data.clear()
    st.rerun()

# ---------------- Run Processing ----------------
if api_key and uploaded_files:
    with st.spinner("🔧 Initializing QA Tool..."):
        openai.api_base = "https://api.deepseek.com/v1"
        openai.api_key = api_key
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.3
        )

        # Save files to temp dir
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

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("✅ Curriculum QA initialized. Ask your questions below!")

    # ---------------- Q&A Section ----------------
    question = st.text_input("💬 Ask a question about your curriculum")
    if question:
        try:
            with st.spinner("Thinking..."):
                answer = qa_chain.run(question)
            st.markdown(f"**💡 Answer:** {answer}")

            # Save to CSV
            if not os.path.exists("qa_log.csv"):
                with open("qa_log.csv", "w", newline='', encoding="utf-8") as f:
                    csv.writer(f).writerow(["Question", "Answer"])
            with open("qa_log.csv", "a", newline='', encoding="utf-8") as f:
                csv.writer(f).writerow([question, answer])

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
else:
    st.info("🔐 Please provide your DeepSeek API key and upload at least one document.")
