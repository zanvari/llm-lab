import json
import time
from pathlib import Path
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import re
from layoutlm_dataset import load_funsd_labels, extract_layoutlm_kv_pairs
from data_preprocessing import process_sample
from sklearn.metrics import precision_score, recall_score, f1_score

def infer_donut(image_path, question, processor, model):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    prompt = f"<s_docvqa><s_question>{question}<s_answer>"
    inputs = processor.tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(pixel_values, decoder_input_ids=inputs.input_ids, max_length=512)
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return re.findall(r'\"(.*?)\"\s*:\s*\"(.*?)\"', decoded)

def evaluate_models(image_dir, label_dir, n=5):
    layoutlm_preds = []
    donut_preds = []
    ground_truths = []

    # Load Donut model
    donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    donut_model.eval()

    image_files = sorted(Path(image_dir).glob("*.png"))[:n]

    for image_path in image_files:
        sample_id = image_path.stem
        label_path = Path(label_dir) / f"{sample_id}.json"

        # Get ground truth
        gt_kv = load_funsd_labels(label_path)

        # Inference — LayoutLM
        layoutlm_kv = extract_layoutlm_kv_pairs(sample_id)

        # Inference — Donut
        start = time.time()
        donut_kv = infer_donut(str(image_path), "What are the key fields and their values?", donut_processor, donut_model)
        end = time.time()
        donut_time = end - start

        # Format predictions
        gt_pairs = [f"{k.strip().lower()}:{v.strip().lower()}" for k, v in gt_kv.items()]
        layoutlm_pairs = [f"{k.strip().lower()}:{v.strip().lower()}" for k, v in layoutlm_kv.items()]
        donut_pairs = [f"{k.strip().lower()}:{v.strip().lower()}" for k, v in donut_kv]

        ground_truths.append(gt_pairs)
        layoutlm_preds.append(layoutlm_pairs)
        donut_preds.append(donut_pairs)

        print(f"🧾 {sample_id}")
        print(f" - Donut Inference Time: {donut_time:.2f} sec")

    def compute_f1(preds, golds):
        y_true, y_pred = [], []
        for gt, pred in zip(golds, preds):
            all_keys = set(gt + pred)
            y_true.extend([1 if x in gt else 0 for x in all_keys])
            y_pred.extend([1 if x in pred else 0 for x in all_keys])
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return precision, recall, f1

    layoutlm_metrics = compute_f1(layoutlm_preds, ground_truths)
    donut_metrics = compute_f1(donut_preds, ground_truths)

    print("\n📊 LayoutLM — Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(*layoutlm_metrics))
    print("📊 Donut     — Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(*donut_metrics))

if __name__ == "__main__":
    evaluate_models(
        image_dir="data/images",
        label_dir="data/annotations",
        n=5
    )
