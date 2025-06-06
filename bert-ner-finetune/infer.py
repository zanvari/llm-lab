# infer.py — Run inference on custom sentence using fine-tuned model
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

label_list = ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
model = AutoModelForTokenClassification.from_pretrained("./outputs/bert-ner")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sentence = "Angela Merkel visited Berlin to meet German officials."
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs).logits
predictions = torch.argmax(outputs, dim=2)

print("Token - Label")
for token, pred in zip(tokens, predictions[0][1:-1]):
    print(f"{token} - {label_list[pred]}")
