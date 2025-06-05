# classify_batch.py

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# Few-shot examples
fewshot_examples = [
    {"text": "This document is an agreement between two parties regarding software services.", "label": "contract"},
    {"text": "Invoice No: 12345. Amount due: $500. Due Date: July 1, 2024.", "label": "invoice"},
    {"text": "Name: John Smith. Medical diagnosis: Hypertension. Treatment: Medication.", "label": "medical_report"},
    {"text": "Dear Mr. Johnson, Please find attached the proposal for your review.", "label": "letter"},
    {"text": "Claim ID: 78910. Policy Number: XZ-456. Claimed amount: $2000.", "label": "insurance_form"}
]

# Prompt template
prompt_template = """
You are an intelligent document classifier. Classify the following document into one of the categories: contract, invoice, medical_report, letter, insurance_form.

Use the examples below for reference:

{examples}

Document:
"""{doc}"""

Label:
"""

example_blocks = "\n\n".join([
    f"Document:\n{ex['text']}\nLabel: {ex['label']}"
    for ex in fewshot_examples
])

def build_prompt(doc_text):
    return prompt_template.format(examples=example_blocks, doc=doc_text)

# Load model
load_dotenv()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Example test documents
test_docs = [
    "Policyholder: Jane Doe. Total amount claimed: $1200. Date of incident: 01/02/2024.",
    "Invoice #78453. Due: September 30. Total due: $350.",
    "Hello John, I hope you're doing well. I've attached the meeting notes."
]

print("\n🔍 Running Batch Classification\n")

for i, doc in enumerate(test_docs, 1):
    prompt = build_prompt(doc)
    response = llm.predict(prompt).strip()
    print(f"{i}. Document: {doc}\n   → Predicted Label: {response}\n")

