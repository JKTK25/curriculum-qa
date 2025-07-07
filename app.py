import os
import csv
import io
import openai
import streamlit as st
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# ---------------- CONFIG ----------------
st.set_page_config(page_title="📚 school AI", layout="centered")
API_KEY = os.getenv("DEESEEK_API_KEY") or st.secrets.get("DEESEEK_API_KEY")
if not API_KEY:
    st.error("❌ Missing API key. Please set DEESEEK_API_KEY in Streamlit secrets.")
    st.stop()

# ---------------- STYLING ----------------
st.image("https://static.mycareersfuture.gov.sg/images/company/logos/b9b623bfe890ac230ac57629e84742ba/lark-technologies.png", width=100)
st.title("📚 School AI")

st.markdown("""
    <style>
        .block-container {padding-top: 2rem;}
        .stChatMessage.user {text-align: right;}
        .stChatMessage.user .stMarkdown {
            background-color: #DCF8C6; padding: 0.8rem 1rem;
            border-radius: 10px; display: inline-block; max-width: 80%;
        }
        .stChatMessage.assistant .stMarkdown {
            background-color: #F1F0F0; padding: 0.8rem 1rem;
            border-radius: 10px; display: inline-block; max-width: 80%;
        }
        .download-footer {
            position: fixed; bottom: 5px; right: 5px;
            background-color: #f0f2f6; padding: 6px 8px;
            border-radius: 6px; font-size: 10px;
        }
        .quick-queries { margin-top: 1rem; font-size: 0.85rem; }
        .stButton>button { padding: 0.2rem 0.4rem; font-size: 0.75rem; }
    </style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---------------- LOAD VECTOR STORE ----------------
@st.cache_resource

def load_vectorstore():
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    HF_REPO_ID = "JK-TK/curriculum-faiss-index"
    LOCAL_INDEX_DIR = "faiss_index_general"
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)

    hf_hub_download(repo_id=HF_REPO_ID, filename="index.faiss", repo_type="dataset", local_dir=LOCAL_INDEX_DIR)
    hf_hub_download(repo_id=HF_REPO_ID, filename="index.pkl", repo_type="dataset", local_dir=LOCAL_INDEX_DIR)

    return FAISS.load_local(LOCAL_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# ---------------- LOAD LLM ----------------
@st.cache_resource

def load_llm():
    openai.api_base = "https://api.deepseek.com/v1"
    openai.api_key = API_KEY
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=API_KEY,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0.3
    )

# ---------------- INIT QA CHAIN ----------------
if st.session_state.qa_chain is None:
    with st.spinner("🧐 Initializing chatbot..."):
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = load_llm()

        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=3)
        chat_prompt = ChatPromptTemplate.from_template("English.\n\nContext:\n{context}\n\nQuestion: {question}")

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": chat_prompt}
        )
        st.session_state.qa_chain = chain

# ---------------- CHAT UI ----------------
if st.session_state.qa_chain:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    if not st.session_state.chat_history:
        st.markdown("<div class='quick-queries'>", unsafe_allow_html=True)
        st.markdown("### 🔎 Quick queries:")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🧬 What is biology?", key="bio"):
                st.session_state.user_input = "What is biology?"
        with col2:
            if st.button("⚗️ What is chemistry?", key="chem"):
                st.session_state.user_input = "What is chemistry?"
        with col3:
            if st.button("➕ Solve: 2x + 10 = 20", key="math"):
                st.session_state.user_input = "Solve: 2x + 10 = 20"
        st.markdown("</div>", unsafe_allow_html=True)

    user_input = st.chat_input("💬 Ask a question about your curriculum")
    if user_input:
        st.session_state.user_input = user_input

    if "user_input" in st.session_state:
        query = st.session_state.pop("user_input")
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    result = st.session_state.qa_chain({"question": query})
                    answer = result["answer"]

                    for phrase in [
                        "from the provided context", "based on the context provided",
                        "according to the information provided", "from what I can gather"
                    ]:
                        answer = answer.replace(phrase, "").strip()

                    st.markdown(answer)
                    st.session_state.chat_history.append(("user", query))
                    st.session_state.chat_history.append(("assistant", answer))

                    with open("qa_log.csv", "a", newline='', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if f.tell() == 0:
                            writer.writerow(["Question", "Answer"])
                        writer.writerow([query, answer])
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

# ---------------- DOWNLOAD FOOTER ----------------
if st.session_state.chat_history:
    txt_buffer = io.StringIO()
    for role, msg in st.session_state.chat_history:
        speaker = "You" if role == "user" else "Bot"
        txt_buffer.write(f"{speaker}: {msg}\n{'-'*50}\n")
    txt_bytes = txt_buffer.getvalue().encode("utf-8")

    st.markdown("<div class='download-footer'>", unsafe_allow_html=True)
    st.download_button("⬇ Chat TXT", txt_bytes, "chat_history.txt", "text/plain", key="download_txt")
    st.markdown("</div>", unsafe_allow_html=True)
