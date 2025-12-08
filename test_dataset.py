from train_datasets import MixedEvalDataset, WinograndeDataset, BoolQDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer

# ds = BoolQDataset()

# dl = DataLoader(
#         ds,
#         batch_size=1,
#         shuffle=False,  # Streaming dataset
#         num_workers=0,
#         pin_memory=True
#     )

ds = load_dataset('google/boolq', split='train')

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B')

lengths = []

for example in ds:
    question = example['question']
    passage = example['passage']
    answer = example['answer']

    full_text = f"Question: {question}\nPassage: {passage}\nAnswer: {answer}\n"

    tokens = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
    )

    length = (tokens['attention_mask'] != 0).sum().item()
    lengths.append(length)

l = np.array(lengths)
print("Mean length:", l.mean())
print("Max length:", l.max())
print("Min length:", l.min())
print("99th percentile length:", np.percentile(l, 99))
print("95th percentile length:", np.percentile(l, 95))