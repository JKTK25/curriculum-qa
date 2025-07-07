import os 
import csv 
import io 
import openai 
import uuid 
import time 
import streamlit as st

from huggingface_hub import hf_hub_download 
from langchain_community.vectorstores import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain 
from langchain.memory import ConversationBufferMemory 
from langchain_openai import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate 
from langchain.callbacks.base import BaseCallbackHandler
import firebase_admin 
from firebase_admin import credentials, firestore

----- Streamlittream Handler -----

class StreamlitCallbackHandler(BaseCallbackHandler): def init(self, container): self.container = container self.text = ""

def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text + "▌")
    time.sleep(0.01)

------------- CONFIG -------------

st.set_page_config(page_title="📚 School AI", layout="centered") API_KEY = os.getenv("DEESEEK_API_KEY") or st.secrets.get("DEESEEK_API_KEY") if not API_KEY: st.error("❌ Missing API key. Please set DEESEEK_API_KEY in Streamlit secrets.") st.stop()

------------- FIREBASE INIT -------------

if "firebase_app" not in st.session_state: if "FIREBASE" in st.secrets: cred = credentials.Certificate(dict(st.secrets["FIREBASE"])) else: cred = credentials.Certificate("firebase_key.json") if not firebase_admin._apps: firebase_admin.initialize_app(cred) st.session_state.firebase_app = True

db = firestore.client() if "user_id" not in st.session_state: st.session_state.user_id = str(uuid.uuid4())

def log_chat(user_id, question, answer): db.collection("chat_logs").document(user_id).collection("history").add({ "question": question, "answer": answer, "timestamp": firestore.SERVER_TIMESTAMP })

def load_chat_history_from_firebase(user_id): chat_ref = db.collection("chat_logs").document(user_id).collection("history").order_by("timestamp") chat_docs = chat_ref.stream() history = [] for doc in chat_docs: data = doc.to_dict() history.append(("user", data["question"])) history.append(("assistant", data["answer"])) return history

------------- CACHED COMPONENTS -------------

@st.cache_resource def load_embeddings(): return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource def load_vectorstore(): HF_REPO_ID = "JK-TK/curriculum-faiss-index" LOCAL_INDEX_DIR = "faiss_index_general" os.makedirs(LOCAL_INDEX_DIR, exist_ok=True) if not os.path.exists(f"{LOCAL_INDEX_DIR}/index.faiss"): hf_hub_download(repo_id=HF_REPO_ID, filename="index.faiss", repo_type="dataset", local_dir=LOCAL_INDEX_DIR) hf_hub_download(repo_id=HF_REPO_ID, filename="index.pkl", repo_type="dataset", local_dir=LOCAL_INDEX_DIR) return FAISS.load_local(LOCAL_INDEX_DIR, load_embeddings(), allow_dangerous_deserialization=True)

@st.cache_resource def load_memory(): return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

------------- INIT QA CHAIN -------------

if "chat_history" not in st.session_state: st.session_state.chat_history = load_chat_history_from_firebase(st.session_state.user_id)

------------- HEADER UI -------------

st.image("https://static.mycareersfuture.gov.sg/images/company/logos/b9b623bfe890ac230ac57629e84742ba/lark-technologies.png", width=100) st.title("📚 School AI")

st.markdown("""

<style>
    .block-container { padding-top: 2rem; }
    .stChatMessage.user {
        text-align: right;
    }
    .stChatMessage.user .stMarkdown {
        background-color: #DCF8C6;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        display: inline-block;
        max-width: 80%;
    }
    .stChatMessage.assistant .stMarkdown {
        background-color: #F1F0F0;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        display: inline-block;
        max-width: 80%;
    }
    .download-footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        font-size: 12px;
    }
</style>""", unsafe_allow_html=True)

------------- CHAT DISPLAY -------------

if st.session_state.chat_history: for role, msg in st.session_state.chat_history: with st.chat_message(role): st.markdown(msg)

user_input = st.chat_input("💬 Ask a question about your curriculum") if user_input: st.session_state.chat_history.append(("user", user_input)) with st.chat_message("user"): st.markdown(user_input)

with st.chat_message("assistant"):
    container = st.empty()
    stream_handler = StreamlitCallbackHandler(container)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=API_KEY,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.3,
            streaming=True,
            callbacks=[stream_handler]
        ),
        retriever=load_vectorstore().as_retriever(search_kwargs={"k": 5}),
        memory=load_memory(),
        combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_template(
            "English.\n\nContext:\n{context}\n\nQuestion: {question}"
        )}
    )

    with st.spinner("🤖 Thinking..."):
        result = qa_chain({"question": user_input})
        answer = stream_handler.text
        st.session_state.chat_history.append(("assistant", answer))
        log_chat(st.session_state.user_id, user_input, answer)

------------- DOWNLOAD CHAT -------------

if st.session_state.chat_history: txt_buffer = io.StringIO() for role, msg in st.session_state.chat_history: speaker = "You" if role == "user" else "Bot" txt_buffer.write(f"{speaker}: {msg}\n{'-'*50}\n") txt_bytes = txt_buffer.getvalue().encode("utf-8")

st.markdown("<div class='download-footer'>", unsafe_allow_html=True)
st.download_button("⬇️ Download Chat (TXT)", txt_bytes, "chat_history.txt", "text/plain", key="download_txt")
st.markdown("</div>", unsafe_allow_html=True)

