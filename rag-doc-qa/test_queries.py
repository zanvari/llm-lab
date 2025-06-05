# test_queries.py

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pathlib import Path
import time

# Load API key
load_dotenv()

# Load PDFs
data_path = Path("data")
documents = []
for file in data_path.glob("*.pdf"):
    loader = PyPDFLoader(str(file))
    documents.extend(loader.load())

# Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embed
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Load queries
with open("queries.txt", "r") as f:
    queries = [line.strip() for line in f if line.strip()]

# Run and log results
print("\nRunning test queries:\n")
for query in queries:
    print(f"Q: {query}")
    start = time.time()
    result = qa_chain({"query": query})
    end = time.time()
    print("A:", result['result'])
    print("Sources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "N/A"))
    print(f"Response Time: {end - start:.2f} sec\n")

