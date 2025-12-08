import torch

import datasets
import transformers

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset as TorchDataset


class FineWebEduDataset(TorchDataset):
    """Dataset class for FineWeb-Edu tokenized data, BoolQ-style outputs."""

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        streaming: bool = True,
        split: str = "train",
        num_proc: int = 8,   # kept for API compatibility
        max_examples: int | None = None,  # only used when streaming=False
    ):
        self.max_length = max_length
        self.streaming = streaming
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading dataset {dataset_name} (split: {split}, streaming={streaming})...")

        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
        )

        # If not streaming, we can optionally truncate by examples and use real __len__
        if not streaming and max_examples is not None:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )

        if streaming:
            # Take a finite slice of the stream
            self.dataset = self.dataset.take(500_000)

        print("FineWeb-Edu dataset loaded successfully")

    def __len__(self):
        if self.streaming:
            # approximate, like before
            return 1_000_000
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        try:
            if self.streaming:
                # streaming-safe access
                item = next(iter(self.dataset.skip(idx).take(1)))
            else:
                item = self.dataset[idx]

            text = item.get("text", "")

            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Force everything to CPU, matching BoolQ style
            input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
            attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

            labels = input_ids.clone()  # CPU

            assert input_ids.device.type == "cpu"
            assert attention_mask.device.type == "cpu"
            assert labels.device.type == "cpu"

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        except Exception as e:
            # Fallback dummy batch on any error (also CPU)
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
    """Dataset class for SlimPajama tokenized data, BoolQ-style outputs."""

    def __init__(
        self,
        dataset_name: str = "cerebras/SlimPajama-627B",
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        streaming: bool = True,
        split: str = "train",
        num_proc: int = 8,   # kept for API compatibility
        max_examples: int | None = None,  # only for non-streaming
    ):
        self.max_length = max_length
        self.streaming = streaming
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading dataset {dataset_name} (split: {split}, streaming={streaming}).")

        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
        )

        # For non-streaming, allow truncation by max_examples
        if not streaming and max_examples is not None:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )

        if streaming:
            self.dataset = self.dataset.take(500_000)

        print("SlimPajama dataset loaded successfully")

    def __len__(self):
        if self.streaming:
            return 1_000_000
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        try:
            if self.streaming:
                item = next(iter(self.dataset.skip(idx).take(1)))
            else:
                item = self.dataset[idx]

            text = item.get("text", "")

            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Force CPU tensors, just like BoolQ/Winogrande
            input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
            attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

            labels = input_ids.clone()

            assert input_ids.device.type == "cpu"
            assert attention_mask.device.type == "cpu"
            assert labels.device.type == "cpu"

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        except Exception as e:
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


class BoolQDataset(TorchDataset):
    """Dataset class for BoolQ tokenized data, same style as FineWebEdu/SlimPajama."""

    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        split: str = "train",
        num_proc: int = 8,        # kept for API compatibility
        max_examples: int | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading BoolQ dataset (split: {split})...")
        self.dataset = load_dataset("google/boolq", split=split)

        if max_examples is not None:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )

        print(f"BoolQ dataset loaded: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        passage = item["passage"]
        question = item["question"]
        answer = bool(item["answer"])

        text = (
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            f"Answer (yes or no): {'yes' if answer else 'no'}"
        )

        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Force everything to CPU, even if some global default made it CUDA
        input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

        labels = input_ids.clone()  # already CPU

        # Optional debug: uncomment if you want to confirm devices
        # print("BoolQ devices:", input_ids.device, attention_mask.device, labels.device)

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