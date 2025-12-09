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
        )

        input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

        prompt_only_tokens = self.tokenizer(
            prompt,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_length = prompt_only_tokens["input_ids"].shape[1]

        labels = input_ids.clone()
        # if prompt_length < labels.shape[0]:
        #     labels[:prompt_length] = -100

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
    Dataset class for PubMedQA tokenized data, BoolQ-style outputs.

    Uses the qiaojin/PubMedQA dataset on HuggingFace. By default we use the
    labeled split (`pqa_labeled`), which has yes/no/maybe `final_decision`
    labels.

    Each example is formatted roughly as:

        Question: ...
        Context: ...
        Long answer: ...
        Final decision (yes / no / maybe): ...

    and we train with full LM loss over the entire sequence
    (labels = input_ids.clone()).
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

        parts = [
            f"Question: {question}",
            f"Context: {context_text}",
        ]
        if long_answer:
            parts.append(f"Long answer: {long_answer}")
        if final_decision:
            parts.append(
                f"Final decision (yes / no / maybe): {final_decision}"
            )

        text = "\n".join(parts)

        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Force everything to CPU, consistent with BoolQ/Winogrande/etc.
        input_ids = tokens["input_ids"].squeeze(0).to("cpu", non_blocking=False)
        attention_mask = tokens["attention_mask"].squeeze(0).to("cpu", non_blocking=False)

        labels = input_ids.clone()  # full LM loss

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

class CaseHoldDataset(TorchDataset):
    """
    LexGLUE CaseHOLD subset as multiple-choice prompt + single-token answer.
    Follows the Winogrande style: we list the options in the prompt and only
    put loss on the final answer letter (A–E).
    """

    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 512,
        split: str = "train",            # "train", "validation", or "test"
        num_proc: int = 8,               # kept for API compatibility, not used
        max_examples: int | None = None,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading LexGLUE CaseHOLD dataset (split: {split})...")
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

        # Option letters we’ll use in the prompt and answer
        self.option_letters = ["A", "B", "C", "D", "E"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        context = item["context"]

        # HF viewer shows both "endings" and "sequence"; the actual list-of-options
        # is in "sequence" on the parquet version. Fall back to "endings" just in case.
        options = item.get("sequence", None)
        if options is None:
            options = item.get("endings", None)
        if options is None:
            raise KeyError("Expected 'sequence' or 'endings' field in CaseHOLD example.")

        assert len(options) == 5, f"Expected 5 options, got {len(options)}."

        label_idx = int(item["label"])
        assert 0 <= label_idx < len(options), f"Bad label index {label_idx}."

        # Build prompt listing all options.
        # This mirrors your Winogrande style: prompt then a short answer token.
        lines = [f"Context: {context}"]
        for i, opt in enumerate(options):
            letter = self.option_letters[i]
            lines.append(f"Option {letter}: {opt}")
        prompt = "\n".join(lines) + "\nAnswer (A, B, C, D, or E):"

        # Target answer is just the letter, prefixed with a space to be similar
        # to Winogrande’s " 1"/" 2" formatting.
        target_answer = " " + self.option_letters[label_idx]
        full_text = prompt + target_answer

        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Get the length of just the prompt (no answer attached),
        # so we can mask its labels to -100 and only train on the answer.
        prompt_only_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_length = prompt_only_tokens["input_ids"].shape[1]

        labels = input_ids.clone()
        if prompt_length < labels.shape[0]:
            labels[:prompt_length] = -100

        # Force everything to CPU, consistent with your other datasets
        input_ids = input_ids.to("cpu", non_blocking=False)
        attention_mask = attention_mask.to("cpu", non_blocking=False)
        labels = labels.to("cpu", non_blocking=False)

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