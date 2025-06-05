# streamlit_app.py

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pathlib import Path
import os

# Load API keys
load_dotenv()

# Sidebar
st.sidebar.title("🔧 Settings")
embedding_option = st.sidebar.selectbox("Embedding Model", ["OpenAI", "HuggingFace"])
retriever_option = st.sidebar.selectbox("Retriever Type", ["Similarity", "MMR"])

st.title("📄 RAG-based Document QA")
st.markdown("Upload a PDF or use existing ones to ask questions about their content.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
data_path = Path("data")
data_path.mkdir(exist_ok=True)

# Save uploaded file
if uploaded_file is not None:
    file_path = data_path / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

# Load and chunk documents
all_docs = []
for file in data_path.glob("*.pdf"):
    loader = PyPDFLoader(str(file))
    all_docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

# Embedding model
if embedding_option == "OpenAI":
    embeddings = OpenAIEmbeddings()
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store
if embedding_option == "OpenAI":
    vectorstore = FAISS.from_documents(chunks, embeddings)
else:
    vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="streamlit-chroma")

# Retriever
search_type = "similarity" if retriever_option == "Similarity" else "mmr"
retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Input box for question
query = st.text_input("Ask a question about the document:")

if query:
    result = qa_chain({"query": query})
    st.subheader("Answer")
    st.write(result["result"])

    st.subheader("Sources")
    for doc in result["source_documents"]:
        st.markdown(f"- `{doc.metadata.get('source', 'N/A')}`")

