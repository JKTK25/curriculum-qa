import os
import csv
import io
import uuid
import streamlit as st
from huggingface_hub import hf_hub_download

# ✅ Latest LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# ✅ Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- CONFIG -----------------
st.set_page_config(page_title="📚 School AI", layout="centered")

API_KEY = os.getenv("DEESEEK_API_KEY") or st.secrets.get("DEESEEK_API_KEY")

if not API_KEY:
    st.error("❌ Missing API key. Add DEESEEK_API_KEY to Streamlit Secrets.")
    st.stop()

# ---------------- FIREBASE INIT ----------------
if "firebase_app" not in st.session_state:
    try:
        if "FIREBASE" in st.secrets:
            cred = credentials.Certificate(dict(st.secrets["FIREBASE"]))
        else:
            cred = credentials.Certificate("firebase_key.json")  # local fallback

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        st.session_state.firebase_app = True
        db = firestore.client()
    except Exception as e:
        st.error(f"❌ Firebase error: {e}")
        st.session_state.firebase_app = False
        db = None
else:
    db = firestore.client()


# ---------------- HELPERS ----------------
def log_chat(user_id, question, answer):
    if db:
        try:
            db.collection("chat_logs").document(user_id).collection("history").add(
                {
                    "question": question,
                    "answer": answer,
                    "timestamp": firestore.SERVER_TIMESTAMP
                }
            )
        except:
            pass


# ---------------- CACHE RESOURCES ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_vectorstore(local_dir="faiss_index_general"):
    HF_REPO_ID = "JK-TK/curriculum-faiss-index"
    os.makedirs(local_dir, exist_ok=True)

    if not os.path.exists(f"{local_dir}/index.faiss"):
        hf_hub_download(repo_id=HF_REPO_ID, filename="index.faiss", repo_type="dataset", local_dir=local_dir)
        hf_hub_download(repo_id=HF_REPO_ID, filename="index.pkl", repo_type="dataset", local_dir=local_dir)

    embeddings = load_embeddings()
    return FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=API_KEY,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0.3
    )


@st.cache_resource
def load_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


@st.cache_resource
def load_qa_chain():
    vectorstore = load_vectorstore()
    llm = load_llm()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_template(
        "Answer clearly and in simple English.\n\nContext:\n{context}\n\nQuestion: {question}"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=load_memory(),
        combine_docs_chain_kwargs={"prompt": prompt}
    )


# ---------------- UI ----------------

st.title("📚 School AI – Curriculum Chatbot")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

qa_chain = load_qa_chain()

# show previous messages
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

query = st.chat_input("💬 Ask anything from your curriculum...")

if query:
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain({"question": query})
            answer = result["answer"]
            st.markdown(answer)

            st.session_state.chat_history.append(("assistant", answer))
            log_chat(st.session_state.user_id, query, answer)

# Allow TXT download
if st.session_state.chat_history:
    buffer = io.StringIO()
    for role, msg in st.session_state.chat_history:
        speaker = "You" if role == "user" else "Bot"
        buffer.write(f"{speaker}:\n{msg}\n\n")

    st.download_button(
        "⬇️ Download Chat",
        buffer.getvalue(),
        file_name="chat_history.txt"
    )
