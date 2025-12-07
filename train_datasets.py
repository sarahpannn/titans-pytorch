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