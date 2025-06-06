# train.py — Fine-tune BERT for NER on CoNLL-2003
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import f1_score
import numpy as np

label_list = load_dataset("conll2003")["train"].features["ner_tags"].feature.names
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

dataset = load_dataset("conll2003")
dataset = dataset.map(tokenize_and_align_labels, batched=True)
dataset.set_format("torch")

model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label_list[l] for (p_, l) in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    true_preds = [[label_list[p_] for (p_, l) in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    return {"f1": f1_score(true_labels, true_preds)}

training_args = TrainingArguments(
    output_dir="./outputs",
    evaluation_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./outputs/bert-ner")
