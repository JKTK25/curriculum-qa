import os
import csv
import tempfile
import streamlit as st
from tqdm import tqdm
import io
import shutil
import openai

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

# ------------- CONFIG -------------
st.set_page_config(page_title="📚 Curriculum Chatbot", layout="centered")
DEMO_MODE = st.secrets.get("DEMO_MODE", False)
API_KEY = os.getenv("DEESEEK_API_KEY") or st.secrets.get("DEESEEK_API_KEY")

if not API_KEY:
    st.error("❌ Missing API key. Please set DEESEEK_API_KEY in Streamlit secrets.")
    st.stop()

# ------------- STYLING -------------
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

# ------------- SIDEBAR -------------
with st.sidebar:
    st.subheader("⚙️ Configuration")
    subject = st.selectbox("📚 Select Subject", ["General", "Chemistry", "Biology", "Physics", "Math"])
    index_path = f"faiss_index_{subject.lower()}"

    if not DEMO_MODE:
        uploaded_files = st.file_uploader("📁 Upload curriculum files", accept_multiple_files=True)
    else:
        uploaded_files = []
        st.info("🟢 Demo mode: Using prebuilt index.")

    if st.button("🔄 Reset Chat"):
        st.session_state.clear()
        st.experimental_rerun()

    if st.button("🗑 Clear Saved Index"):
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            st.success("✅ FAISS index cleared.")
            st.session_state.clear()
            st.experimental_rerun()

# ------------- SESSION STATE -------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------- LLM SETUP -------------
if "qa_chain" not in st.session_state:
    with st.spinner("🧠 Initializing chatbot..."):
        openai.api_base = "https://api.deepseek.com/v1"
        openai.api_key = API_KEY
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=API_KEY,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.3
        )

        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load or build index
        if os.path.exists(index_path):
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            st.success(f"📦 Loaded index for {subject}")
        elif uploaded_files:
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

            chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(all_docs)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(index_path)
            st.success(f"✅ Index built and saved for {subject}")
        else:
            st.info("📁 Please upload files or use a prebuilt index.")
            st.stop()

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        st.session_state.qa_chain = chain

# ------------- CHAT DISPLAY -------------
if "qa_chain" in st.session_state:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    user_input = st.chat_input("💬 Ask a question about your curriculum")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    result = st.session_state.qa_chain({"question": user_input})
                    answer = result["answer"]

                    # Clean up AI responses
                    for phrase in [
                        "from the provided context", "based on the context provided",
                        "according to the information provided", "from what I can gather"
                    ]:
                        answer = answer.replace(phrase, "").strip()

                    st.markdown(answer)

                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("assistant", answer))

                    # Log to CSV
                    if not os.path.exists("qa_log.csv"):
                        with open("qa_log.csv", "w", newline='', encoding="utf-8") as f:
                            csv.writer(f).writerow(["Question", "Answer"])
                    with open("qa_log.csv", "a", newline='', encoding="utf-8") as f:
                        csv.writer(f).writerow([user_input, answer])
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

    # ------------- DOWNLOAD HISTORY -------------
    if st.session_state.chat_history:
        st.divider()
        st.subheader("📥 Download Chat History")
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["Role", "Message"])
        writer.writerows(st.session_state.chat_history)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        txt_buffer = io.StringIO()
        for role, msg in st.session_state.chat_history:
            speaker = "You" if role == "user" else "Bot"
            txt_buffer.write(f"{speaker}: {msg}\n{'-'*50}\n")
        txt_bytes = txt_buffer.getvalue().encode("utf-8")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download as CSV", csv_bytes, "chat_history.csv", "text/csv")
        with col2:
            st.download_button("⬇️ Download as TXT", txt_bytes, "chat_history.txt", "text/plain")
