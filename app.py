import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pypdf import PdfReader
import docx


# -------------------------------------
# Helper functions
# -------------------------------------
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

        else:  # Plain text files
            documents.append(file.read().decode("utf-8"))

    return documents


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.create_documents(docs)


def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


# -------------------------------------
# Streamlit UI
# -------------------------------------
st.set_page_config(page_title="📚 Curriculum Q&A AI", page_icon="🤖", layout="wide")
st.title("📚 AI Curriculum Question Answering Assistant")

st.write("Upload curriculum documents (PDF / Word / TXT), then ask questions.")

uploaded_files = st.file_uploader(
    "Upload files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# -------------------------------------
# Embedding + Retrieval
# -------------------------------------
if uploaded_files:
    docs = extract_text_from_uploaded_files(uploaded_files)
    chunks = split_documents(docs)
    vector_store = build_vector_store(chunks)

    st.success("✅ Documents uploaded and processed successfully!")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Use OpenAI (GPT-4o-mini) for answering
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    combine_docs_chain = create_stuff_documents_chain(llm)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    question = st.text_input("Ask a question about the curriculum:")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": question})

        st.write("### ✅ Answer:")
        st.write(response["answer"])

