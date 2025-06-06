import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./outputs"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

label_list = ["finance", "healthcare", "insurance", "legal", "project_management"]

with open("data/batch_input.json") as f:
    samples = json.load(f)

predictions = []
for item in samples:
    inputs = tokenizer(item["text"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze().tolist()
    labels = [label_list[i] for i, p in enumerate(probs) if p > 0.5]
    predictions.append({"id": item["id"], "predicted": labels})

with open("outputs/batch_output.json", "w") as f:
    json.dump(predictions, f, indent=2)

print("Saved predictions to outputs/batch_output.json")
