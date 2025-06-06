import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import re

def infer_donut(image_path, question):
    processor = DonutProcessor.from_pretrained('naver-clova-ix/donut-base-finetuned-docvqa')
    model = VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base-finetuned-docvqa')
    model.eval()

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    task_prompt = f"<s_docvqa><s_question>{question}<s_answer>"
    inputs = processor.tokenizer(task_prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(pixel_values, decoder_input_ids=inputs.input_ids, max_length=512)

    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print("\n🔍 Raw Output:")
    print(decoded)

    print("\n🔑 Extracted Key-Value Pairs:")
    kv_pairs = re.findall(r'\"(.*?)\"\s*:\s*\"(.*?)\"', decoded)
    for key, value in kv_pairs:
        print(f"{key.strip()} --> {value.strip()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to form image")
    parser.add_argument("--question", type=str, default="What are the key fields and their values?", help="Document QA prompt")
    args = parser.parse_args()

    infer_donut(args.image, args.question)
