# streamlit_app.py

import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load few-shot examples from JSON
with open("fewshot_examples.json", "r") as f:
    fewshot_examples = json.load(f)

# Prepare prompt template
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

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# Streamlit UI
st.set_page_config(page_title="Few-Shot Document Classifier", layout="centered")
st.title("🧠 Few-Shot Document Classifier")
st.markdown("Classify text into contract, invoice, letter, etc. using GPT-3.5 and few-shot prompts.")

input_text = st.text_area("📄 Paste a document snippet below:", height=200)

if st.button("🔍 Classify"):
    if not input_text.strip():
        st.warning("Please enter a document snippet to classify.")
    else:
        with st.spinner("Classifying with GPT-3.5..."):
            prompt = build_prompt(input_text)
            prediction = llm.predict(prompt).strip()
            st.success(f"**Predicted Label:** `{prediction}`")
            with st.expander("🔍 View Prompt"):
                st.code(prompt)

st.markdown("---")
st.caption("Built with LangChain, Streamlit, and OpenAI.")

