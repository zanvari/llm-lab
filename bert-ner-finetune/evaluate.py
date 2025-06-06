# evaluate.py — Evaluate BERT NER model on validation set with detailed metrics
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import numpy as np
from seqeval.metrics import classification_report, f1_score

# Load dataset and tokenizer
dataset = load_dataset("conll2003")
label_list = dataset["train"].features["ner_tags"].feature.names
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()
    label_ids = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(example["ner_tags"][word_idx])
        else:
            label_ids.append(example["ner_tags"][word_idx])
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

tokenized = dataset.map(tokenize_and_align_labels, batched=True)
tokenized.set_format("torch")

# Load fine-tuned model
model = AutoModelForTokenClassification.from_pretrained("./outputs/bert-ner")

# Define Trainer
def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)

    true_labels = [
        [label_list[l] for (p_, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(preds, labels)
    ]
    true_preds = [
        [label_list[p_] for (p_, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(preds, labels)
    ]
    return {"f1": f1_score(true_labels, true_preds)}

args = TrainingArguments(output_dir="./outputs", per_device_eval_batch_size=16)
trainer = Trainer(model=model, tokenizer=tokenizer, args=args)

# Evaluate
predictions, labels, _ = trainer.predict(tokenized["validation"])
metrics = compute_metrics((predictions, labels))

print("F1-score:", metrics["f1"])
print("
Full classification report:")
print(classification_report(
    [[label_list[l] for l in seq if l != -100] for seq in labels],
    [[label_list[p] for (p, l) in zip(pred_seq, label_seq) if l != -100]
     for pred_seq, label_seq in zip(np.argmax(predictions, axis=2), labels)]
))
