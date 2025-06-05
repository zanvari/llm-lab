from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# Load PDF
loader = PyPDFLoader("data/contract.pdf")
pages = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

# Embed & Store
embedding_model = OpenAIEmbeddings()
vectordb = FAISS.from_documents(docs, embedding_model)

# Create retriever + QA chain
retriever = vectordb.as_retriever()
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask a question
query = "What are the termination conditions in the contract?"
response = qa_chain.run(query)
print("Answer:", response)

