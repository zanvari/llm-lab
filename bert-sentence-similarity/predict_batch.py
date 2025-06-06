import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./outputs"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

with open("data/batch_input.json") as f:
    pairs = json.load(f)

results = []
for item in pairs:
    inputs = tokenizer(item["sentence1"], item["sentence2"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    score = outputs.logits.item()
    results.append({"id": item["id"], "score": score})

with open("outputs/batch_output.json", "w") as f:
    json.dump(results, f, indent=2)

print("Batch inference complete. Saved to outputs/batch_output.json.")
