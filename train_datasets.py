import torch

import datasets
import transformers

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset as TorchDataset


class FineWebEduDataset(TorchDataset):
    """Dataset class for FineWeb-Edu tokenized data."""

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        streaming: bool = True,
        split: str = "train",
        num_proc: bool = 8
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading dataset {dataset_name}...")

        # FineWeb-Edu's main text field is usually "text"
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            # trust_remote_code=True,  # usually not needed here
        )

        print("Dataset loaded successfully")

        # For demo / small runs, take a subset if streaming
        if streaming:
            # Adjust this as you like for your token budget
            self.dataset = self.dataset.take(500_000)

        print("Done taking from stream (if streaming)")

    def __len__(self):
        # For streaming datasets, __len__ is not well-defined;
        # return a large approximate number.
        return 1_000_000

    def __getitem__(self, idx):
        try:
            # Get the idx-th element in a streaming-safe way
            item = next(iter(self.dataset.skip(idx).take(1)))

            # FineWeb-Edu stores the main content under "text"
            text = item.get("text", "")

            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }

        except Exception as e:
            # Fallback dummy batch on any error
            dummy_ids = torch.full(
                (self.max_length,),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device="cpu",
            )
            return {
                "input_ids": dummy_ids,
                "attention_mask": torch.zeros(
                    self.max_length,
                    dtype=torch.long,
                    device="cpu",
                ),
                "labels": dummy_ids.clone(),
            }

class SlimPajamaDataset(TorchDataset):
    """Dataset class for SlimPajama tokenized data."""
    
    def __init__(
        self, 
        dataset_name: str = "cerebras/SlimPajama-627B",
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B", 
        max_length: int = 2048,
        streaming: bool = True,
        split: str = "train",
        num_proc: int = 8
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Loading dataset {dataset_name}...")
        
        # Load dataset in streaming mode for large datasets
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            # trust_remote_code=True
        )

        print("Dataset loaded successfully")
        
        # For demonstration, we'll take a subset for the 1B token target
        # In practice, you'd want to configure this based on your compute
        if streaming:
            # Take first portion of the dataset
            self.dataset = self.dataset.take(500000)  # Approximate for 1B tokens
        
        print("Done taking from stream")
    
    def __len__(self):
        # For streaming dataset, return a large number
        return 1000000  # Approximate
    
    def __getitem__(self, idx):
        try:
            # Get next item from streaming dataset
            item = next(iter(self.dataset.skip(idx).take(1)))
            text = item['text']
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = tokens['input_ids'].squeeze()
            attention_mask = tokens['attention_mask'].squeeze()
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids.clone()
            }
            
        except Exception as e:
            # Return a dummy batch if there's an error
            dummy_ids = torch.full(
                (self.max_length,),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device="cpu",
            )
            return {
            'input_ids': dummy_ids,
            'attention_mask': torch.zeros(
                self.max_length,
                dtype=torch.long,
                device="cpu",
            ),
            'labels': dummy_ids.clone(),
        }



class BoolQDataset(TorchDataset):
    """BoolQ as prompt + yes/no answer, with loss only on the answer tokens."""

    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        split: str = "train",
        num_proc: int = 8,           # kept for API compatibility, not used
        max_examples: int | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading BoolQ dataset (split: {split})...")
        self.dataset = load_dataset("google/boolq", split=split)

        if max_examples is not None:
            self.dataset = self.dataset.select(range(min(max_examples, len(self.dataset))))

        print(f"BoolQ dataset loaded: {len(self.dataset)} examples")

        self.yes_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
        self.no_id  = self.tokenizer.encode(" no",  add_special_tokens=False)[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        passage = item["passage"]
        question = item["question"]
        answer = bool(item["answer"])  # True/False

        prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer (yes or no):"
        target_answer = " yes" if answer else " no"
        full_text = prompt + target_answer

        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        # Force to CPU in case default tensor device is CUDA
        input_ids = tokens["input_ids"].squeeze(0).to("cpu")
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu")

        # Safety: ensure exact length
        if input_ids.shape[0] != self.max_length:
            if input_ids.shape[0] < self.max_length:
                pad_len = self.max_length - input_ids.shape[0]
                pad_ids = torch.full(
                    (pad_len,),
                    self.tokenizer.pad_token_id,
                    dtype=input_ids.dtype,
                    device="cpu",
                )
                pad_mask = torch.zeros(
                    pad_len,
                    dtype=attention_mask.dtype,
                    device="cpu",
                )
                input_ids = torch.cat([input_ids, pad_ids], dim=0)
                attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
            else:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]

        # Prompt length for masking labels
        prompt_only_tokens = self.tokenizer(
            prompt,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_length = prompt_only_tokens["input_ids"].shape[1]

        labels = input_ids.clone()
        if prompt_length < labels.shape[0]:
            labels[:prompt_length] = -100

        # Final sanity
        assert input_ids.device.type == "cpu"
        assert attention_mask.device.type == "cpu"
        assert labels.device.type == "cpu"
        assert input_ids.shape[0] == self.max_length
        assert attention_mask.shape[0] == self.max_length
        assert labels.shape[0] == self.max_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

class WinograndeDataset(TorchDataset):
    """
    Winogrande as prompt + '1'/'2' answer, with loss only on the answer token.
    This mirrors BoolQ’s prompt+answer formatting.
    """

    def __init__(
        self,
        winogrande_version: str = "winogrande_xl",
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 512,
        split: str = "train",
        num_proc: int = 8,          # kept for API compatibility, not used
        max_examples: int | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading Winogrande dataset ({winogrande_version}, split: {split})...")
        self.dataset = load_dataset("winogrande", winogrande_version, split=split)

        if max_examples is not None:
            self.dataset = self.dataset.select(range(min(max_examples, len(self.dataset))))

        print(f"Winogrande dataset loaded: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        sentence = item["sentence"]   # contains "_"
        option1 = item["option1"]
        option2 = item["option2"]
        answer_idx = int(item["answer"]) - 1  # "1"/"2" -> 0/1

        prompt = (
            f"Sentence: {sentence}\n"
            f"Option1: {option1}\n"
            f"Option2: {option2}\n"
            f"Answer (1 or 2):"
        )
        target_answer = " 1" if answer_idx == 0 else " 2"
        full_text = prompt + target_answer

        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        if input_ids.shape[0] != self.max_length:
            if input_ids.shape[0] < self.max_length:
                pad_len = self.max_length - input_ids.shape[0]
                pad_ids = torch.full(
                    (pad_len,),
                    self.tokenizer.pad_token_id,
                    dtype=input_ids.dtype,
                )
                pad_mask = torch.zeros(
                    pad_len,
                    dtype=attention_mask.dtype,
                )
                input_ids = torch.cat([input_ids, pad_ids], dim=0)
                attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
            else:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]

        prompt_only_tokens = self.tokenizer(
            prompt,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_length = prompt_only_tokens["input_ids"].shape[1]

        labels = input_ids.clone()
        if prompt_length < labels.shape[0]:
            labels[:prompt_length] = -100

        assert input_ids.device.type == "cpu"
        assert attention_mask.device.type == "cpu"
        assert labels.device.type == "cpu"
        assert input_ids.shape[0] == self.max_length
        assert attention_mask.shape[0] == self.max_length
        assert labels.shape[0] == self.max_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MixedEvalDataset(TorchDataset):
    """Mixed dataset combining BoolQ and Winogrande for joint training."""

    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        boolq_split: str = "train",
        winogrande_version: str = "winogrande_xl",
        winogrande_split: str = "train",
        max_boolq_examples: int | None = None,
        max_winogrande_examples: int | None = None,
        mix_ratio: float = 0.5,  # fraction of BoolQ examples
    ):
        self.max_length = max_length

        self.boolq_dataset = BoolQDataset(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split=boolq_split,
            max_examples=max_boolq_examples,
        )

        self.winogrande_dataset = WinograndeDataset(
            winogrande_version=winogrande_version,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split=winogrande_split,
            max_examples=max_winogrande_examples,
        )

        self.mix_ratio = mix_ratio

        total_boolq = len(self.boolq_dataset)
        total_winogrande = len(self.winogrande_dataset)

        target_total = max(total_boolq, total_winogrande)
        self.boolq_samples = int(target_total * mix_ratio)
        self.winogrande_samples = target_total - self.boolq_samples

        print(
            f"Mixed dataset created: "
            f"{self.boolq_samples} BoolQ + {self.winogrande_samples} Winogrande "
            f"= {self.__len__()} total"
        )

    def __len__(self):
        return self.boolq_samples + self.winogrande_samples

    def __getitem__(self, idx):
        # All underlying datasets already return CPU tensors
        if idx < self.boolq_samples:
            boolq_idx = idx % len(self.boolq_dataset)
            return self.boolq_dataset[boolq_idx]
        else:
            win_idx = (idx - self.boolq_samples) % len(self.winogrande_dataset)
            return self.winogrande_dataset[win_idx]