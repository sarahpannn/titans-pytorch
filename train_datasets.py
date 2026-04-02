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
            labels[attention_mask == 0] = -100  # Ignore padding in loss

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
                "labels": torch.full((self.max_length,), -100, dtype=torch.long, device="cpu"),
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
            labels[attention_mask == 0] = -100  # Ignore padding in loss

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
                "labels": torch.full((self.max_length,), -100, dtype=torch.long, device="cpu"),
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

        prompt = (
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            f"Answer (yes or no): "
        )
        answer_text = "yes" if answer else "no"
        full_text = prompt + answer_text

        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Force everything to CPU, even if some global default made it CUDA
        input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

        # Tokenize prompt alone to find where the answer starts
        prompt_tokens = self.tokenizer(
            prompt, truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        prompt_ids = prompt_tokens["input_ids"]
        if prompt_ids[0, -1] == self.tokenizer.eos_token_id:
            prompt_ids = prompt_ids[:, :-1]
        prompt_length = prompt_ids.shape[1]

        # Only compute loss on answer tokens
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100
        # If answer was truncated away, keep last non-padding token to avoid NaN loss
        if (labels != -100).sum() == 0:
            last_valid = attention_mask.sum() - 1
            if last_valid >= 0:
                labels[last_valid] = input_ids[last_valid]

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
        )

        input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

        prompt_only_tokens = self.tokenizer(
            prompt, truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        prompt_ids = prompt_only_tokens["input_ids"]
        if prompt_ids[0, -1] == self.tokenizer.eos_token_id:
            prompt_ids = prompt_ids[:, :-1]
        prompt_length = prompt_ids.shape[1]

        # Only compute loss on answer tokens
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100
        # If answer was truncated away, keep last non-padding token to avoid NaN loss
        if (labels != -100).sum() == 0:
            last_valid = attention_mask.sum() - 1
            if last_valid >= 0:
                labels[last_valid] = input_ids[last_valid]

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

class PubMedQADataset(TorchDataset):
    """
    Dataset class for PubMedQA tokenized data with last-token-only loss.

    Uses the qiaojin/PubMedQA dataset on HuggingFace. By default we use the
    labeled split (`pqa_artificial`), which has yes/no/maybe `final_decision`
    labels.

    Each example is formatted roughly as:

        Question: ...
        Context: ...
        Long answer: ...
        Final decision (yes / no / maybe): 

    The loss is computed ONLY on predicting the final decision token (yes/no/maybe).
    All other positions have labels set to -100 (ignored in loss).
    """

    def __init__(
        self,
        config_name: str = "pqa_artificial",
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        split: str = "train",
        num_proc: int = 8,          # kept for API compatibility
        max_examples: int | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # PubMedQA as hosted under qiaojin/PubMedQA uses only a train split
        # for the main configs; in practice you probably want split="train".
        print(f"Loading PubMedQA dataset (config: {config_name}, split: {split})...")
        self.dataset = load_dataset("qiaojin/PubMedQA", config_name, split=split)

        if max_examples is not None:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )

        print(f"PubMedQA dataset loaded: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        question = item.get("question", "")

        # In the HF script, `context` is a structured field with "contexts"
        # as a sequence of strings. We join them into one big context.
        context = item.get("context", {})
        contexts = context.get("contexts", "") if isinstance(context, dict) else context
        if isinstance(contexts, (list, tuple)):
            context_text = " ".join(contexts)
        else:
            context_text = str(contexts)

        long_answer = item.get("long_answer", "")
        final_decision = item.get("final_decision", "")

        # Build prompt without the answer
        parts = [
            f"Question: {question}",
            f"Context: {context_text}",
        ]
        if long_answer:
            parts.append(f"Long answer: {long_answer}")
        
        parts.append("Final decision (yes / no / maybe):")
        
        prompt = "\n".join(parts)
        
        # Add a space before the answer token for proper tokenization
        full_text = prompt + f" {final_decision}"

        # Tokenize the full text
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Force everything to CPU, consistent with BoolQ/Winogrande/etc.
        input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

        # Tokenize just the prompt to find where the answer starts
        prompt_tokens = self.tokenizer(
            prompt, truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        prompt_ids = prompt_tokens["input_ids"]
        if prompt_ids[0, -1] == self.tokenizer.eos_token_id:
            prompt_ids = prompt_ids[:, :-1]
        prompt_length = prompt_ids.shape[1]

        # Only compute loss on answer tokens
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100
        # If answer was truncated away, keep last non-padding token to avoid NaN loss
        if (labels != -100).sum() == 0:
            last_valid = attention_mask.sum() - 1
            if last_valid >= 0:
                labels[last_valid] = input_ids[last_valid]

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


class DROPDataset(TorchDataset):
    """
    Dataset class for DROP (Discrete Reasoning Over Paragraphs) tokenized data.
    
    Uses the ucinlp/drop dataset. Each example contains a passage, question,
    and answer(s). We format it as a reading comprehension task.
    """

    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        split: str = "train",
        num_proc: int = 8,
        max_examples: int | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading DROP dataset (split: {split})...")
        self.dataset = load_dataset("ucinlp/drop", split=split)

        if max_examples is not None:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )

        print(f"DROP dataset loaded: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        passage = item.get("passage", "")
        question = item.get("question", "")
        
        # DROP answers can be in various formats
        answers_dict = item.get("answers_spans", {})
        if isinstance(answers_dict, dict):
            spans = answers_dict.get("spans", [])
            if spans and len(spans) > 0:
                answer = spans[0]  # Use first answer
            else:
                answer = ""
        else:
            answer = str(answers_dict)

        # Format as reading comprehension
        prompt = (
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            f"Answer: "
        )
        full_text = prompt + answer

        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

        # Tokenize prompt alone to find where the answer starts
        prompt_tokens = self.tokenizer(
            prompt, truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        prompt_ids = prompt_tokens["input_ids"]
        if prompt_ids[0, -1] == self.tokenizer.eos_token_id:
            prompt_ids = prompt_ids[:, :-1]
        prompt_length = prompt_ids.shape[1]

        # Only compute loss on answer tokens
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100
        # If answer was truncated away, keep last non-padding token to avoid NaN loss
        if (labels != -100).sum() == 0:
            last_valid = attention_mask.sum() - 1
            if last_valid >= 0:
                labels[last_valid] = input_ids[last_valid]

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


class SQuADv2Dataset(TorchDataset):
    """
    Dataset class for SQuAD v2.0 tokenized data.
    
    Uses the rajpurkar/squad_v2 dataset. Includes both answerable and 
    unanswerable questions.
    """

    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        split: str = "train",
        num_proc: int = 8,
        max_examples: int | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading SQuAD v2 dataset (split: {split})...")
        self.dataset = load_dataset("rajpurkar/squad_v2", split=split)

        if max_examples is not None:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )

        print(f"SQuAD v2 dataset loaded: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        context = item.get("context", "")
        question = item.get("question", "")
        
        # Handle answers
        answers = item.get("answers", {})
        if isinstance(answers, dict):
            answer_texts = answers.get("text", [])
            if answer_texts and len(answer_texts) > 0:
                answer = answer_texts[0]
            else:
                answer = "No answer"
        else:
            answer = "No answer"

        # Format as QA task
        prompt = (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer: "
        )
        full_text = prompt + answer

        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

        # Tokenize prompt alone to find where the answer starts
        prompt_tokens = self.tokenizer(
            prompt, truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        prompt_ids = prompt_tokens["input_ids"]
        if prompt_ids[0, -1] == self.tokenizer.eos_token_id:
            prompt_ids = prompt_ids[:, :-1]
        prompt_length = prompt_ids.shape[1]

        # Only compute loss on answer tokens
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100
        # If answer was truncated away, keep last non-padding token to avoid NaN loss
        if (labels != -100).sum() == 0:
            last_valid = attention_mask.sum() - 1
            if last_valid >= 0:
                labels[last_valid] = input_ids[last_valid]

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


class CommonsenseQADataset(TorchDataset):
    """
    Dataset class for CommonsenseQA tokenized data.
    
    Uses the tau/commonsense_qa dataset. Multiple choice questions requiring
    common sense reasoning.
    """

    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        split: str = "train",
        num_proc: int = 8,
        max_examples: int | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading CommonsenseQA dataset (split: {split})...")
        self.dataset = load_dataset("tau/commonsense_qa", split=split)

        if max_examples is not None:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )

        print(f"CommonsenseQA dataset loaded: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        question = item.get("question", "")
        choices = item.get("choices", {})
        
        # Format choices
        if isinstance(choices, dict):
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            choices_formatted = "\n".join([
                f"{label}. {text}" for label, text in zip(labels, texts)
            ])
        else:
            choices_formatted = str(choices)

        answer_key = item.get("answerKey", "")

        # Format as multiple choice
        prompt = (
            f"Question: {question}\n"
            f"Choices:\n{choices_formatted}\n"
            f"Answer: "
        )
        full_text = prompt + answer_key

        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

        # Tokenize prompt alone to find where the answer starts
        prompt_tokens = self.tokenizer(
            prompt, truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        prompt_ids = prompt_tokens["input_ids"]
        if prompt_ids[0, -1] == self.tokenizer.eos_token_id:
            prompt_ids = prompt_ids[:, :-1]
        prompt_length = prompt_ids.shape[1]

        # Only compute loss on answer tokens
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100
        # If answer was truncated away, keep last non-padding token to avoid NaN loss
        if (labels != -100).sum() == 0:
            last_valid = attention_mask.sum() - 1
            if last_valid >= 0:
                labels[last_valid] = input_ids[last_valid]

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


class HugeMixedDataset(TorchDataset):
    """
    Huge mixed dataset combining DROP, SQuAD v2, CommonsenseQA, Winogrande, and BoolQ.
    
    Samples from all five datasets according to specified ratios.
    """

    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        split: str = "train",
        max_drop_examples: int | None = None,
        max_squad_examples: int | None = None,
        max_csqa_examples: int | None = None,
        max_winogrande_examples: int | None = None,
        max_boolq_examples: int | None = None,
        # Ratios: fraction of total for each dataset (should sum to ~1.0)
        drop_ratio: float = 0.2,
        squad_ratio: float = 0.2,
        csqa_ratio: float = 0.2,
        winogrande_ratio: float = 0.2,
        boolq_ratio: float = 0.2,
    ):
        self.max_length = max_length

        # Load all datasets
        self.drop_dataset = DROPDataset(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split=split,
            max_examples=max_drop_examples,
        )

        self.squad_dataset = SQuADv2Dataset(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split=split,
            max_examples=max_squad_examples,
        )

        self.csqa_dataset = CommonsenseQADataset(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split=split,
            max_examples=max_csqa_examples,
        )

        self.winogrande_dataset = WinograndeDataset(
            winogrande_version="winogrande_xl",
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split=split,
            max_examples=max_winogrande_examples,
        )

        self.boolq_dataset = BoolQDataset(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split=split,
            max_examples=max_boolq_examples,
        )

        # Normalize ratios
        total_ratio = drop_ratio + squad_ratio + csqa_ratio + winogrande_ratio + boolq_ratio
        self.drop_ratio = drop_ratio / total_ratio
        self.squad_ratio = squad_ratio / total_ratio
        self.csqa_ratio = csqa_ratio / total_ratio
        self.winogrande_ratio = winogrande_ratio / total_ratio
        self.boolq_ratio = boolq_ratio / total_ratio

        # Calculate total samples based on largest dataset
        max_dataset_size = max(
            len(self.drop_dataset),
            len(self.squad_dataset),
            len(self.csqa_dataset),
            len(self.winogrande_dataset),
            len(self.boolq_dataset),
        )

        # Allocate samples according to ratios
        self.drop_samples = int(max_dataset_size * self.drop_ratio)
        self.squad_samples = int(max_dataset_size * self.squad_ratio)
        self.csqa_samples = int(max_dataset_size * self.csqa_ratio)
        self.winogrande_samples = int(max_dataset_size * self.winogrande_ratio)
        self.boolq_samples = int(max_dataset_size * self.boolq_ratio)

        # Calculate cumulative indices for dataset selection
        self.drop_end = self.drop_samples
        self.squad_end = self.drop_end + self.squad_samples
        self.csqa_end = self.squad_end + self.csqa_samples
        self.winogrande_end = self.csqa_end + self.winogrande_samples
        self.boolq_end = self.winogrande_end + self.boolq_samples

        print(
            f"HugeMixed dataset created:\n"
            f"  DROP: {self.drop_samples}\n"
            f"  SQuAD v2: {self.squad_samples}\n"
            f"  CommonsenseQA: {self.csqa_samples}\n"
            f"  Winogrande: {self.winogrande_samples}\n"
            f"  BoolQ: {self.boolq_samples}\n"
            f"  Total: {self.__len__()} samples"
        )

    def __len__(self):
        return self.boolq_end

    def __getitem__(self, idx):
        # Route to appropriate dataset based on index
        if idx < self.drop_end:
            dataset_idx = idx % len(self.drop_dataset)
            return self.drop_dataset[dataset_idx]
        elif idx < self.squad_end:
            dataset_idx = (idx - self.drop_end) % len(self.squad_dataset)
            return self.squad_dataset[dataset_idx]
        elif idx < self.csqa_end:
            dataset_idx = (idx - self.squad_end) % len(self.csqa_dataset)
            return self.csqa_dataset[dataset_idx]
        elif idx < self.winogrande_end:
            dataset_idx = (idx - self.csqa_end) % len(self.winogrande_dataset)
            return self.winogrande_dataset[dataset_idx]
        else:
            dataset_idx = (idx - self.winogrande_end) % len(self.boolq_dataset)
            return self.boolq_dataset[dataset_idx]


class CaseHOLDDataset(TorchDataset):
    """
    Dataset class for the CaseHOLD subset of LexGLUE.

    Uses the coastalcph/lex_glue dataset with subset "case_hold".
    Each example contains:
        - context: long legal paragraph
        - endings: list of 5 candidate holdings
        - label: index in [0..4] indicating correct holding

    We format as:

        Context: ...
        Possible holdings:
        A. ...
        B. ...
        C. ...
        D. ...
        E. ...
        Answer: <A-E>
    """

    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        split: str = "train",        # "train", "validation", "test"
        num_proc: int = 8,           # kept for API compatibility
        max_examples: int | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading CaseHOLD dataset (split: {split}) from coastalcph/lex_glue...")
        self.dataset = load_dataset(
            "coastalcph/lex_glue",
            "case_hold",
            split=split,
        )

        if max_examples is not None:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )

        print(f"CaseHOLD dataset loaded: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        context: str = item.get("context", "")
        endings = item.get("endings", [])
        label_idx = int(item.get("label", 0))

        # Just in case, truncate / pad endings to 5
        endings = list(endings)[:5]
        while len(endings) < 5:
            endings.append("")

        letters = ["A", "B", "C", "D", "E"]
        choices_formatted = "\n".join(
            f"{letters[i]}. {ending}" for i, ending in enumerate(endings)
        )

        # Map label index -> letter
        if 0 <= label_idx < len(letters):
            answer_letter = letters[label_idx]
        else:
            assert False, f"Invalid label index {label_idx} for CaseHOLD example {idx}"
            # answer_letter = "A"  # fallback, shouldn't really happen

        prompt = (
            f"Context: {context}\n"
            f"Possible holdings:\n{choices_formatted}\n"
            f"Answer: "
        )
        full_text = prompt + answer_letter

        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

        # Tokenize prompt alone to find where the answer starts
        prompt_tokens = self.tokenizer(
            prompt, truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        prompt_ids = prompt_tokens["input_ids"]
        if prompt_ids[0, -1] == self.tokenizer.eos_token_id:
            prompt_ids = prompt_ids[:, :-1]
        prompt_length = prompt_ids.shape[1]

        # Only compute loss on answer tokens
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100
        # If answer was truncated away, keep last non-padding token to avoid NaN loss
        if (labels != -100).sum() == 0:
            last_valid = attention_mask.sum() - 1
            if last_valid >= 0:
                labels[last_valid] = input_ids[last_valid]

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


class AquaRATDataset(TorchDataset):
    """
    Dataset class for AQUA-RAT in multiple-choice format.

    Uses RikoteMaster/aqua-rat-mcqa:

        - question: str
        - choices: list[str] of length 4 (each usually like 'A. ...', 'B. ...', etc.)
        - answer_index: int in [0..3]
        - answer_text: str with the correct option text (e.g. 'C. 5 and 1')

    We format as:

        Question: ...
        Choices:
        <choice_0>
        <choice_1>
        <choice_2>
        <choice_3>
        Answer: <answer_text>
    """

    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        split: str = "train",        # "train" or "test"
        num_proc: int = 8,           # kept for API compatibility
        max_examples: int | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading AQUA-RAT dataset (deepmind/aqua_rat, split: {split})...")
        self.dataset = load_dataset("deepmind/aqua_rat", split=split)

        if max_examples is not None:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )

        print(f"AQUA-RAT dataset loaded: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        question: str = item.get("question", "")
        choices = item.get("choices", [])
        # answer_index = int(item.get("answer_index", 0))  # not strictly needed here

        options = item.get("options")
        correct = item.get("correct").strip()

        answer_idx = ord(correct[0]) - ord('A')

        prompt = "Question: " + question + "\nOptions:\n"
        letters: List[str] = []
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            letters.append(letter)
            prompt += f"{opt}\n"

        prompt += "Answer: "

        text = prompt + f"{letters[answer_idx]}"

        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        prompt_ids = prompt_tokens["input_ids"]

        if prompt_ids[0, -1] == self.tokenizer.eos_token_id:
            prompt_ids = prompt_ids[:, :-1]

        prompt_length = prompt_ids.shape[1]

        # Initialize labels with -100 (ignored in loss)
        labels = torch.full_like(input_ids, -100)
        
        # Only compute loss on the final decision token(s)
        # The answer starts at prompt_length
        if prompt_length < input_ids.shape[0]:
            # Set labels for the answer tokens (typically just one token for yes/no/maybe)
            # We want to predict the first token of the answer
            labels[prompt_length-1:prompt_length] = input_ids[prompt_length-1:prompt_length]

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
