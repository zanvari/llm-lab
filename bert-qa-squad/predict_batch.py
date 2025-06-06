import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_metric

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained("./bert-qa-squad/outputs/bert-qa")

# Load evaluation metric
squad_metric = load_metric("squad")

# Load examples
with open("bert-qa-squad/data/batch_input.json", "r") as f:
    examples = json.load(f)

predictions = []
references = []
inference_times = []

for ex in examples:
    question = ex["question"]
    context = ex["context"]
    answers = ex.get("answers", {"text": [""], "answer_start": [0]})
    references.append({"id": ex["id"], "answers": answers})

    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end_time = time.time()

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx], skip_special_tokens=True)

    predictions.append({"id": ex["id"], "prediction_text": answer})
    inference_times.append(end_time - start_time)

# Evaluate EM / F1
results = squad_metric.compute(predictions=predictions, references=references)

# Report results
avg_inference_time = sum(inference_times) / len(inference_times)
print("Evaluation Results:")
print(f"  - Exact Match (EM): {results['exact_match']:.2f}")
print(f"  - F1 Score: {results['f1']:.2f}")
print(f"  - Avg. Inference Time: {avg_inference_time:.4f} sec/question")
