import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./outputs"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def compute_similarity(sent1, sent2):
    inputs = tokenizer(sent1, sent2, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.item()

# Example usage
s1 = "A man is walking a dog."
s2 = "A person is walking an animal."
print("Similarity score:", compute_similarity(s1, s2))
