from torch.utils.data import Dataset
from transformers import LayoutLMTokenizerFast
from data_preprocessing import parse_funsd_json

class FUNSDDataset(Dataset):
    def __init__(self, file_paths, tokenizer_name='microsoft/layoutlm-base-uncased', max_length=512):
        self.tokenizer = LayoutLMTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.samples = [parse_funsd_json(p) for p in file_paths]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item['words'],
            boxes=item['boxes'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=False,
            is_split_into_words=True,
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        labels = item['labels'][:self.max_length] + [0] * (self.max_length - len(item['labels']))
        encoding['labels'] = labels
        return encoding
