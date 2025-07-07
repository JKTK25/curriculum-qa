import os
import csv
import io
import openai
import streamlit as st
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# -------- CONFIG --------
st.set_page_config(page_title="📚 Curriculum Chatbot", layout="centered")
API_KEY = os.getenv("DEESEEK_API_KEY") or st.secrets.get("DEESEEK_API_KEY")

if not API_KEY:
    st.error("❌ Missing API key. Please set DEESEEK_API_KEY in Streamlit secrets.")
    st.stop()

# -------- HEADER --------
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/368px-Google_2015_logo.svg.png", width=100)
st.title("📚 Curriculum Chatbot")

# -------- STYLING --------
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
        #download-box {
            position: fixed;
            bottom: 10px;
            right: 20px;
            background-color: #f0f2f6;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

# -------- SESSION STATE --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------- LLM & VECTORSTORE --------
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

        HF_REPO_ID = "JK-TK/curriculum-faiss-index"
        LOCAL_INDEX_DIR = "faiss_index_general"
        os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)

        hf_hub_download(repo_id=HF_REPO_ID, filename="index.faiss", repo_type="dataset", local_dir=LOCAL_INDEX_DIR)
        hf_hub_download(repo_id=HF_REPO_ID, filename="index.pkl", repo_type="dataset", local_dir=LOCAL_INDEX_DIR)

        vectorstore = FAISS.load_local(LOCAL_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retriever, memory=memory
        )

# -------- CHAT DISPLAY --------
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

                    for phrase in [
                        "from the provided context", "based on the context provided",
                        "according to the information provided", "from what I can gather"
                    ]:
                        answer = answer.replace(phrase, "").strip()

                    st.markdown(answer)
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("assistant", answer))

                    with open("qa_log.csv", "a", newline='', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if f.tell() == 0:
                            writer.writerow(["Question", "Answer"])
                        writer.writerow([user_input, answer])
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

# -------- COMMON QUERIES --------
st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
st.markdown("**💡 Common Queries:**")
common_queries = [
    "What is biology?",
    "What is chemistry?",
    "Solve 2x + 10 = 0"
]

cols = st.columns(len(common_queries))
for i, q in enumerate(common_queries):
    if cols[i].button(q, key=f"suggestion_{i}"):
        st.session_state.common_query = q
        st.experimental_rerun()

# -------- HANDLE COMMON QUERY --------
if "common_query" in st.session_state:
    query = st.session_state.pop("common_query")
    st.session_state.chat_history.append(("user", query))
    result = st.session_state.qa_chain({"question": query})
    answer = result["answer"].strip()
    st.session_state.chat_history.append(("assistant", answer))
    st.experimental_rerun()

# -------- FLOATING DOWNLOAD BUTTON --------
if st.session_state.chat_history:
    txt_buffer = io.StringIO()
    for role, msg in st.session_state.chat_history:
        speaker = "You" if role == "user" else "Bot"
        txt_buffer.write(f"{speaker}: {msg}\n{'-'*50}\n")
    txt_bytes = txt_buffer.getvalue().encode("utf-8")

    with st.container():
        st.markdown("""
            <div id="download-box">
                📥 <b>Download Chat</b><br>
        """, unsafe_allow_html=True)
        st.download_button(
            label="⬇ TXT", data=txt_bytes,
            file_name="chat_history.txt", mime="text/plain",
            key="download-txt", help="Download your chat as a text file"
        )
        st.markdown("</div>", unsafe_allow_html=True)
