import os
import csv
import io
import uuid
import streamlit as st
from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import firebase_admin
from firebase_admin import credentials, firestore

# ------------- CONFIG -------------
st.set_page_config(page_title="📚 School AI", layout="centered")
API_KEY = os.getenv("DEESEEK_API_KEY") or st.secrets.get("DEESEEK_API_KEY")

if not API_KEY:
    st.error("❌ Missing API key. Please set DEESEEK_API_KEY in Streamlit secrets.")
    st.stop()

# ------------- FIREBASE INIT -------------
if "firebase_app" not in st.session_state:
    try:
        if "FIREBASE" in st.secrets:
            cred = credentials.Certificate(dict(st.secrets["FIREBASE"]))
        else:
            cred = credentials.Certificate("firebase_key.json")  # Local fallback

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        st.session_state.firebase_app = True
        db = firestore.client()
    except Exception as e:
        st.error(f"❌ Firebase initialization failed: {e}")
        st.session_state.firebase_app = False
        db = None
else:
    db = firestore.client()

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

def log_chat(user_id, question, answer):
    if db:
        try:
            db.collection("chat_logs").document(user_id).collection("history").add({
                "question": question,
                "answer": answer,
                "timestamp": firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            st.error(f"⚠️ Failed to log chat: {e}")

# ------------- CACHED RESOURCES -------------
@st.cache_resource
def load_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"❌ Failed to load embeddings: {e}")
        return None

@st.cache_resource
def load_vectorstore(local_dir="faiss_index_general"):
    try:
        HF_REPO_ID = "JK-TK/curriculum-faiss-index"
        os.makedirs(local_dir, exist_ok=True)

        if not os.path.exists(os.path.join(local_dir, "index.faiss")) or not os.path.exists(os.path.join(local_dir, "index.pkl")):
            hf_hub_download(repo_id=HF_REPO_ID, filename="index.faiss", repo_type="dataset", local_dir=local_dir)
            hf_hub_download(repo_id=HF_REPO_ID, filename="index.pkl", repo_type="dataset", local_dir=local_dir)

        embeddings = load_embeddings()
        if embeddings:
            return FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)
        return None
    except Exception as e:
        st.error(f"❌ Failed to load vector store: {e}")
        return None

@st.cache_resource
def load_llm():
    try:
        return ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=API_KEY,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.3
        )
    except Exception as e:
        st.error(f"❌ Failed to load LLM: {e}")
        return None

@st.cache_resource
def load_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@st.cache_resource
def load_qa_chain():
    try:
        vectorstore = load_vectorstore()
        if not vectorstore:
            return None
            
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        prompt = ChatPromptTemplate.from_template(
            "English.\n\nContext:\n{context}\n\nQuestion: {question}"
        )
        
        llm = load_llm()
        if not llm:
            return None
            
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=load_memory(),
            combine_docs_chain_kwargs={"prompt": prompt}
        )
    except Exception as e:
        st.error(f"❌ Failed to load QA chain: {e}")
        return None

# ------------- HEADER & STYLING -------------
st.image("https://static.mycareersfuture.gov.sg/images/company/logos/b9b623bfe890ac230ac57629e84742ba/lark-technologies.png", width=100)
st.title("📚 School AI")

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
        .download-footer {
            position: fixed;
            bottom: 10px;
            right: 10px;
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 8px;
            font-size: 12px;
        }
        .quick-queries {
            margin-top: 2rem;
        }
        .stButton>button {
            padding: 0.25rem 0.5rem;
            font-size: 0.8rem;
        }
    </style>
""", unsafe_allow_html=True)

# ------------- SESSION STATE -------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    with st.spinner("🤖 Initializing chatbot..."):
        st.session_state.qa_chain = load_qa_chain()
        if st.session_state.qa_chain:
            st.success("✅ Chatbot is ready!")
        else:
            st.error("❌ Failed to initialize chatbot. Please check the logs.")

# ------------- CHAT DISPLAY -------------
if st.session_state.qa_chain:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    if not st.session_state.chat_history:
        st.markdown("<div class='quick-queries'>", unsafe_allow_html=True)
        st.markdown("### 🔎 Quick queries:")
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🧬 What is biology?", key="bio"):
                    st.session_state.user_input = "What is biology?"
            with col2:
                if st.button("\u2697\ufe0f What is chemistry?", key="chem"):
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

                    # Clean up the answer
                    for phrase in [
                        "from the provided context", "based on the context provided",
                        "according to the information provided", "from what I can gather"
                    ]:
                        answer = answer.replace(phrase, "").strip()

                    st.markdown(answer)
                    st.session_state.chat_history.append(("user", query))
                    st.session_state.chat_history.append(("assistant", answer))

                    log_chat(st.session_state.user_id, query, answer)

                    # Log to CSV
                    try:
                        with open("qa_log.csv", "a", newline='', encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if f.tell() == 0:
                                writer.writerow(["Question", "Answer"])
                            writer.writerow([query, answer])
                    except Exception as e:
                        st.error(f"⚠️ Failed to save to CSV: {e}")
                        
                except Exception as e:
                    st.error(f"⚠️ Error processing your question: {e}")

# ------------- DOWNLOAD CHAT HISTORY -------------
if st.session_state.chat_history:
    txt_buffer = io.StringIO()
    for role, msg in st.session_state.chat_history:
        speaker = "You" if role == "user" else "Bot"
        txt_buffer.write(f"{speaker}: {msg}\n{'-'*50}\n")
    txt_bytes = txt_buffer.getvalue().encode("utf-8")

    st.markdown("<div class='download-footer'>", unsafe_allow_html=True)
    st.download_button("⬇️ Download Chat (TXT)", txt_bytes, "chat_history.txt", "text/plain", key="download_txt")
    st.markdown("</div>", unsafe_allow_html=True)
