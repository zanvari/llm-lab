import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./outputs"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

label_list = ["finance", "healthcare", "insurance", "legal", "project_management"]  # Update based on actual classes

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze().tolist()
    return [label_list[i] for i, p in enumerate(probs) if p > 0.5]

# Example
text = "This legal document contains budget and project timeline."
print("Predicted labels:", predict(text))
