import os
import csv
import tempfile
import streamlit as st
from tqdm import tqdm
import io
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

# ----------- SECURE API LOAD -----------
api_key = os.getenv("DEESEEK_API_KEY")
if not api_key:
    st.error("❌ API key not found. Please set DEESEEK_API_KEY in Streamlit secrets.")
    st.stop()

# ----------- SIDEBAR -----------
with st.sidebar:
    st.subheader("📁 Upload Curriculum Files")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, TXT, or JSONL files", accept_multiple_files=True)
    if st.button("🔄 Reset Chat"):
        st.session_state.clear()
        st.experimental_rerun()

# ----------- SESSION MEMORY -----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------- SETUP LLM & CHAIN -----------
if uploaded_files:
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

# ----------- CHAT INTERFACE -----------
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

                    # Remove unwanted boilerplate phrases
                    for phrase in [
                        "from the provided context", "based on the context provided",
                        "according to the information provided", "from what I can gather"
                    ]:
                        answer = answer.replace(phrase, "").strip()

                    st.markdown(answer)

                    # Save history
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

    # ----------- DOWNLOAD CHAT -----------
    if st.session_state.chat_history:
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(["Role", "Message"])
        csv_writer.writerows(st.session_state.chat_history)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        text_buffer = io.StringIO()
        for role, msg in st.session_state.chat_history:
            speaker = "You" if role == "user" else "Bot"
            text_buffer.write(f"{speaker}: {msg}\n")
            text_buffer.write("-" * 50 + "\n")
        text_bytes = text_buffer.getvalue().encode("utf-8")

        st.divider()
        st.subheader("📥 Download Your Conversation")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download as CSV", data=csv_bytes, file_name="chat_history.csv", mime="text/csv")
        with col2:
            st.download_button("⬇️ Download as Text", data=text_bytes, file_name="chat_history.txt", mime="text/plain")

else:
    st.info("⬅️ Upload curriculum files to begin.")
