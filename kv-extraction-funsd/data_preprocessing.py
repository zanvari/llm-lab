import json
import os

# FUNSD labels
label2id = {
    "O": 0,
    "B-KEY": 1,
    "I-KEY": 2,
    "B-VALUE": 3,
    "I-VALUE": 4,
    "B-OTHER": 5,
    "I-OTHER": 6,
}
id2label = {v: k for k, v in label2id.items()}

def normalize_box(box, width=1000, height=1000):
    # Normalize bounding box to 0-1000 scale
    return [
        int(box[0] / width * 1000),
        int(box[1] / height * 1000),
        int(box[2] / width * 1000),
        int(box[3] / height * 1000),
    ]

def parse_funsd_json(json_path, image_size=(1000, 1000)):
    with open(json_path, 'r') as f:
        data = json.load(f)

    words, boxes, labels = [], [], []

    for block in data['form']:
        label = block.get('label', 'other').upper()
        text_label = 'KEY' if label == 'KEY' else 'VALUE' if label == 'VALUE' else 'OTHER'

        if not block['words']:
            continue

        for idx, word in enumerate(block['words']):
            tag = f"{'B' if idx == 0 else 'I'}-{text_label}"
            words.append(word)
            boxes.append(normalize_box(block['box'], *image_size))
            labels.append(label2id.get(tag, 0))

    return {
        'words': words,
        'boxes': boxes,
        'labels': labels
    }
