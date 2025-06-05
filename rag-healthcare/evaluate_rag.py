# evaluate_rag.py

import json
import csv
from dotenv import load_dotenv
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from difflib import SequenceMatcher

# Load environment variables
load_dotenv()

# --- Load documents and build retriever ---
loader = PyPDFLoader("data/health_insurance_claim_detailed.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embedding_model)
retriever = db.as_retriever()

# --- Initialize LLM ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Load eval set ---
with open("eval_set.json", "r") as f:
    eval_set = json.load(f)

# --- Evaluate and record results ---
results = []
for item in eval_set:
    question = item["question"]
    expected = item["expected_answer"]
    predicted = qa_chain.run(question).strip()
    similarity = SequenceMatcher(None, predicted.lower(), expected.lower()).ratio()
    results.append({
        "question": question,
        "expected_answer": expected,
        "predicted_answer": predicted,
        "similarity": round(similarity, 3)
    })

# --- Save to CSV ---
with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("✅ Evaluation complete. Results saved to results.csv")

