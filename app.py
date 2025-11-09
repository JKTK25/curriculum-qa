import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain  # ✅ FIXED IMPORT
from pypdf import PdfReader
import docx


# -----------------------------
# Helper functions
# -----------------------------
def extract_text_from_uploaded_files(uploaded_files):
    documents = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            documents.append(text)

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents.append(text)

        else:  # TXT
            documents.append(file.read().decode("utf-8"))

    return documents


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.create_documents(docs)


def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="📚 Curriculum AI Q&A", page_icon="🤖", layout="wide")
st.title("📚 AI Curriculum Question Answering Assistant")
st.write("Upload curriculum documents (PDF / Word / TXT), then ask anything about them.")

uploaded_files = st.file_uploader(
    "Upload files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    docs = extract_text_from_uploaded_files(uploaded_files)
    chunks = split_documents(docs)
    vector_store = build_vector_store(chunks)

    st.success("✅ Documents processed successfully!")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask a question:")

    if st.button("Ask") and user_query:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(
                {"question": user_query, "chat_history": st.session_state.chat_history}
            )

        st.session_state.chat_history.append((user_query, response["answer"]))

        st.subheader("✅ Answer:")
        st.write(response["answer"])
