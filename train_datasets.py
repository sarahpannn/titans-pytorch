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
    """Dataset class for BoolQ training data with consistent eval formatting."""
    
    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        split: str = "train",
        num_proc: int = 8,
        max_examples: int = None
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading BoolQ dataset (split: {split})...")
        
        # Load BoolQ dataset
        self.dataset = load_dataset("boolq", split=split)
        
        if max_examples is not None:
            self.dataset = self.dataset.select(range(min(max_examples, len(self.dataset))))
        
        print(f"BoolQ dataset loaded: {len(self.dataset)} examples")
        
        # Precompute yes/no token IDs for consistent training
        self.yes_id = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
        self.no_id = self.tokenizer.encode(" no", add_special_tokens=False)[0]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            
            # Format exactly like evaluation: "Passage: {passage}\nQuestion: {question}\nAnswer (yes or no):"
            passage = item['passage']
            question = item['question']
            answer = bool(item['answer'])  # True/False
            
            # Create the prompt (same as evaluation)
            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer (yes or no):"
            target_answer = " yes" if answer else " no"
            
            # First tokenize prompt to get its length
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True
            )
            prompt_length = prompt_tokens['input_ids'].shape[1]
            
            # Reserve space for the answer (typically just 1-2 tokens)
            max_prompt_length = self.max_length - 5  # Reserve space for answer + EOS
            
            # Tokenize prompt with length limit
            if prompt_length > max_prompt_length:
                prompt_tokens = self.tokenizer(
                    prompt,
                    max_length=max_prompt_length,
                    truncation=True,
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=True
                )
            
            # Tokenize answer separately
            answer_tokens = self.tokenizer(
                target_answer,
                add_special_tokens=False,
                return_tensors="pt"
            )
            
            # Combine prompt + answer
            input_ids = torch.cat([
                prompt_tokens['input_ids'].squeeze(0),
                answer_tokens['input_ids'].squeeze(0)
            ], dim=0)
            
            # Pad to max_length
            if len(input_ids) < self.max_length:
                padding_length = self.max_length - len(input_ids)
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                ], dim=0)
            else:
                input_ids = input_ids[:self.max_length]
            
            # Create attention mask
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            
            # Create labels: mask the prompt part, only train on the answer
            labels = input_ids.clone()
            prompt_actual_length = prompt_tokens['input_ids'].shape[1]
            labels[:prompt_actual_length] = -100  # Don't compute loss on prompt
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error processing BoolQ example {idx}: {e}")
            # Return dummy batch
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
                'labels': torch.full(
                    (self.max_length,),
                    -100,  # Ignore loss
                    dtype=torch.long,
                    device="cpu",
                ),
            }


class WinograndeDataset(TorchDataset):
    """Dataset class for Winogrande training data with consistent eval formatting."""
    
    def __init__(
        self,
        winogrande_version: str = "winogrande_xl",
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 512,  # Winogrande sentences are shorter
        split: str = "train",
        num_proc: int = 8,
        max_examples: int = None
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading Winogrande dataset ({winogrande_version}, split: {split})...")
        
        # Load Winogrande dataset
        self.dataset = load_dataset("winogrande", winogrande_version, split=split)
        
        if max_examples is not None:
            self.dataset = self.dataset.select(range(min(max_examples, len(self.dataset))))
        
        print(f"Winogrande dataset loaded: {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            
            # Extract fields like evaluation
            sentence = item['sentence']  # Has underscore placeholder
            option1 = item['option1']
            option2 = item['option2']
            answer = int(item['answer']) - 1  # Convert "1"/"2" to 0/1
            
            # Create the correct sentence by replacing underscore with correct option
            correct_option = option1 if answer == 0 else option2
            full_sentence = sentence.replace("_", correct_option)
            
            # For training, we'll train the model to complete the sentence correctly
            # This matches the evaluation approach which compares log probabilities
            
            # Tokenize the full correct sentence
            tokens = self.tokenizer(
                full_sentence,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
            
            # For causal language modeling, labels are the same as input_ids
            # Model learns to predict next token throughout the sequence
            labels = input_ids.clone()
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error processing Winogrande example {idx}: {e}")
            # Return dummy batch
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


class MixedEvalDataset(TorchDataset):
    """Mixed dataset combining BoolQ and Winogrande for joint training."""
    
    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        boolq_split: str = "train",
        winogrande_version: str = "winogrande_xl", 
        winogrande_split: str = "train",
        max_boolq_examples: int = None,
        max_winogrande_examples: int = None,
        mix_ratio: float = 0.5,  # Fraction of BoolQ vs Winogrande
    ):
        self.max_length = max_length
        
        # Create individual datasets
        self.boolq_dataset = BoolQDataset(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split=boolq_split,
            max_examples=max_boolq_examples
        )
        
        self.winogrande_dataset = WinograndeDataset(
            winogrande_version=winogrande_version,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split=winogrande_split,
            max_examples=max_winogrande_examples
        )
        
        self.mix_ratio = mix_ratio
        
        # Calculate sizes based on mix ratio
        total_boolq = len(self.boolq_dataset)
        total_winogrande = len(self.winogrande_dataset)
        
        # Create indices for mixing
        target_total = max(total_boolq, total_winogrande)
        self.boolq_samples = int(target_total * mix_ratio)
        self.winogrande_samples = target_total - self.boolq_samples
        
        print(f"Mixed dataset created: {self.boolq_samples} BoolQ + {self.winogrande_samples} Winogrande = {target_total} total")
        
    def __len__(self):
        return self.boolq_samples + self.winogrande_samples
    
    def __getitem__(self, idx):
        # Decide whether to sample from BoolQ or Winogrande
        if idx < self.boolq_samples:
            # Sample from BoolQ (with wrapping if needed)
            boolq_idx = idx % len(self.boolq_dataset)
            return self.boolq_dataset[boolq_idx]
        else:
            # Sample from Winogrande (with wrapping if needed)
            winogrande_idx = (idx - self.boolq_samples) % len(self.winogrande_dataset)
            return self.winogrande_dataset[winogrande_idx]