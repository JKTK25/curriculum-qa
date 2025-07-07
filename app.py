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
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain

# ------------- CONFIG -------------
st.set_page_config(page_title="📚 Curriculum Chatbot", layout="centered")
API_KEY = os.getenv("DEESEEK_API_KEY") or st.secrets.get("DEESEEK_API_KEY")

if not API_KEY:
    st.error("❌ Missing API key. Please set DEESEEK_API_KEY in Streamlit secrets.")
    st.stop()

# ------------- HEADER & STYLING -------------
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/368px-Google_2015_logo.svg.png", width=100)
st.title("📚 Curriculum Chatbot")

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
            right: 10px;
            opacity: 0.7;
        }
        #download-box:hover { opacity: 1; }
        .suggested { font-size: 0.85rem; margin: 0.2rem 0; display: inline-block; background: #f0f0f0; padding: 4px 10px; border-radius: 5px; cursor: pointer; }
    </style>
""", unsafe_allow_html=True)

# ------------- SESSION STATE -------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
        st.success("📦 Loaded FAISS index from Hugging Face Hub.")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chat_prompt = ChatPromptTemplate.from_template(
            "You are a helpful curriculum assistant. Always answer in English.\n\nContext:\n{context}\n\nQuestion: {question}"
        )

        doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=chat_prompt)

        chain = ConversationalRetrievalChain(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            memory=memory
        )

        st.session_state.qa_chain = chain

# ------------- CHAT DISPLAY -------------
if "qa_chain" in st.session_state:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # Suggested queries
    if len(st.session_state.chat_history) == 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("💡 **Try a question:**", unsafe_allow_html=True)
        queries = ["What is biology?", "What is chemistry?", "Solve 2x + 10 = 20"]
        cols = st.columns(len(queries))
        for i, q in enumerate(queries):
            if cols[i].button(q, key=f"suggested-{i}"):
                st.session_state.chat_input = q  # Inject into input box
                st.experimental_rerun()

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

                    # Save Q&A
                    if not os.path.exists("qa_log.csv"):
                        with open("qa_log.csv", "w", newline='', encoding="utf-8") as f:
                            csv.writer(f).writerow(["Question", "Answer"])
                    with open("qa_log.csv", "a", newline='', encoding="utf-8") as f:
                        csv.writer(f).writerow([user_input, answer])
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

# ------------- DOWNLOAD CHAT HISTORY (TEXT ONLY) -------------
if st.session_state.chat_history:
    txt_buffer = io.StringIO()
    for role, msg in st.session_state.chat_history:
        speaker = "You" if role == "user" else "Bot"
        txt_buffer.write(f"{speaker}: {msg}\n{'-'*40}\n")
    txt_bytes = txt_buffer.getvalue().encode("utf-8")

    with st.container():
        st.markdown('<div id="download-box">', unsafe_allow_html=True)
        st.download_button("⬇️ Save Q&A", txt_bytes, "chat_history.txt", "text/plain", help="Save full conversation", key="txt_download", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
