import os
import csv
import io
import shutil
import openai
import streamlit as st
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

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
    </style>
""", unsafe_allow_html=True)

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

        index_path = "faiss_index_general"  # Prebuilt FAISS index path

        if os.path.exists(index_path):
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            st.success("📦 Prebuilt FAISS index loaded.")
        else:
            st.error("❌ Prebuilt FAISS index not found. Please ensure 'faiss_index_general' exists.")
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

    # ------------- DOWNLOAD CHAT HISTORY -------------
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
