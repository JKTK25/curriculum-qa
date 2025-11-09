import os
import csv
import io
import uuid
import streamlit as st
from huggingface_hub import hf_hub_download

# LangChain imports (✅ latest architecture)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="📚 School AI", layout="centered")

API_KEY = os.getenv("DEESEEK_API_KEY") or st.secrets.get("DEESEEK_API_KEY")
if not API_KEY:
    st.error("❌ Missing API key. Add DEESEEK_API_KEY in Streamlit Secrets.")
    st.stop()

# ---------------------------------------------------------
# FIREBASE INIT
# ---------------------------------------------------------
if "firebase_app" not in st.session_state:
    try:
        cred = credentials.Certificate(dict(st.secrets["FIREBASE"]))
        firebase_admin.initialize_app(cred)
        st.session_state.firebase_app = True
        db = firestore.client()
    except Exception as e:
        st.error(f"⚠️ Firebase error: {e}")
        st.session_state.firebase_app = False
        db = None
else:
    db = firestore.client()


def log_chat(user, question, answer):
    """ Save chat to Firebase """
    if db:
        try:
            db.collection("chat_logs").document(user).collection("history").add(
                {"question": question, "answer": answer, "timestamp": firestore.SERVER_TIMESTAMP}
            )
        except Exception:
            pass


# ---------------------------------------------------------
# CACHED RESOURCES
# ---------------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_vectorstore():
    HF_REPO = "JK-TK/curriculum-faiss-index"
    local_dir = "faiss_index_general"

    os.makedirs(local_dir, exist_ok=True)

    if not os.path.exists(f"{local_dir}/index.faiss"):
        hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename="index.faiss", local_dir=local_dir)
        hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename="index.pkl", local_dir=local_dir)

    return FAISS.load_local(local_dir, load_embeddings(), allow_dangerous_deserialization=True)


@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0.3,
        openai_api_key=API_KEY,
        openai_api_base="https://api.deepseek.com/v1",
    )


@st.cache_resource
def load_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful academic tutor.

Use only the context below — do NOT make up information.

CONTEXT:
{context}

QUESTION:
{question}
"""
    )

    docs_chain = create_stuff_documents_chain(
        llm=load_llm(),
        prompt=prompt
    )

    return create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=docs_chain,
    )


# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
st.image("https://static.mycareersfuture.gov.sg/images/company/logos/b9b623bfe890ac230ac57629e84742ba/lark-technologies.png", width=100)
st.title("📚 School AI Chatbot")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

qa_chain = load_qa_chain()

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)


# ---------------------------------------------------------
# INPUT BOX
# ---------------------------------------------------------
prompt = st.chat_input("Ask something related to the curriculum...")

if prompt:
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"question": prompt})
            answer = result["answer"]

            answer = answer.replace("According to the context", "").strip()
            st.markdown(answer)

            st.session_state.chat_history.append(("assistant", answer))
            log_chat(st.session_state.user_id, prompt, answer)


# ---------------------------------------------------------
# CHAT DOWNLOAD TXT
# ---------------------------------------------------------
if st.session_state.chat_history:
    history = "\n".join([f"{r.upper()}: {m}" for r, m in st.session_state.chat_history])
    st.download_button(
        "⬇ Download Chat",
        history.encode("utf-8"),
        "chat.txt",
        mime="text/plain"
    )
