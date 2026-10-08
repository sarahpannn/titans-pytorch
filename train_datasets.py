import json
import os
import random

import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset as TorchDataset, IterableDataset


def _ddp_shard():
    """Return (rank, world_size) for the current process.

    Streaming IterableDatasets are replicated to every DDP rank, so each rank
    must select a disjoint slice of the source stream itself -- there is no
    DistributedSampler for iterable datasets.  Returns (0, 1) when torch.distributed
    is not initialized, so single-GPU behavior is bit-identical to before.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1




class PackedFineWebEduDataset(IterableDataset):
    """Dense fixed-length causal-LM blocks packed from FineWeb-Edu documents.

    Documents are tokenized
    without padding, separated by one EOS token, and concatenated directly
    into full blocks.  It keeps long-context continuation training both dense
    and practical.
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        split: str = "train",
        shuffle_buffer_size: int = 1_024,
        seed: int = 42,
    ):
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        self.dataset_name = dataset_name
        self.split = split
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("PackedFineWebEduDataset requires a tokenizer with eos_token_id")

    def __iter__(self):
        rank, world_size = _ddp_shard()
        # A rank-specific shuffle avoids duplicate packed blocks under DDP
        # without requiring each rank to read and discard every other document.
        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        ds = ds.shuffle(
            seed=self.seed + rank,
            buffer_size=self.shuffle_buffer_size,
        )
        token_buffer = []
        eos_token_id = self.tokenizer.eos_token_id
        for item in ds:
            text = item.get("text", "")
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            token_buffer.extend(ids)
            if token_buffer[-1] != eos_token_id:
                token_buffer.append(eos_token_id)
            while len(token_buffer) >= self.max_length:
                input_ids = torch.tensor(
                    token_buffer[:self.max_length], dtype=torch.long
                )
                del token_buffer[:self.max_length]
                yield {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                    "labels": input_ids.clone(),
                }


class MixedLongAlignFineWebPretrainDataset(IterableDataset):
    """An even token mix of causal-LM LongAlign and packed FineWeb-Edu.

    Both sources yield dense ``max_length`` blocks, so alternating blocks gives
    an exact 50/50 token split (apart from a possible final odd block). LongAlign
    supervises every non-padding token and is cycled after a complete pass;
    FineWeb-Edu documents are streamed and packed with EOS separators.
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int,
        seed: int = 42,
        fineweb_shuffle_buffer_size: int = 1_024,
    ):
        self.max_length = max_length
        self.seed = seed
        self.longalign = LongAlignSFTDataset(
            dataset_name="zai-org/LongAlign-10k",
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split="train",
            shuffle_buffer_size=1_024,
            seed=seed,
            causal_lm=True,
        )
        self.fineweb = PackedFineWebEduDataset(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            shuffle_buffer_size=fineweb_shuffle_buffer_size,
            seed=seed,
        )

    def _packed_longalign(self):
        input_buffer = []
        eos_token_id = self.longalign.tokenizer.eos_token_id
        while True:
            for sample in self.longalign:
                valid_ids = sample["input_ids"][sample["attention_mask"].to(torch.bool)].tolist()
                if not valid_ids:
                    continue
                input_buffer.extend(valid_ids)
                if eos_token_id is not None and input_buffer[-1] != eos_token_id:
                    input_buffer.append(eos_token_id)
                while len(input_buffer) >= self.max_length:
                    input_ids = torch.tensor(input_buffer[:self.max_length], dtype=torch.long)
                    del input_buffer[:self.max_length]
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                        "labels": input_ids.clone(),
                    }

    def __iter__(self):
        longalign_iter = self._packed_longalign()
        fineweb_iter = iter(self.fineweb)
        # Alternate fixed-width blocks. The seed controls which source comes
        # first without changing the exact long-run 50/50 token ratio.
        fineweb_first = bool(self.seed % 2)
        while True:
            if fineweb_first:
                yield next(fineweb_iter)
                yield next(longalign_iter)
            else:
                yield next(longalign_iter)
                yield next(fineweb_iter)


class LongAlignSFTDataset(IterableDataset):
    """
    Supervised finetuning dataset for chat/instruction data.

    Supports LongAlign messages, conversations, and prompt/response rows. For SFT we train
    only on assistant tokens and mask user/system/context tokens with -100,
    which is the standard objective for instruction tuning. ``causal_lm=True``
    instead supervises every non-padding token in the formatted conversation.
    """

    def __init__(
        self,
        dataset_name: str = "zai-org/LongAlign-10k",
        tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B",
        max_length: int = 2048,
        split: str = "train",
        max_examples: int | None = None,
        shuffle_buffer_size: int = 1024,
        seed: int = 42,
        causal_lm: bool = False,
    ):
        self.dataset_name = dataset_name
        self.dataset_names = [
            name.strip()
            for name in dataset_name.replace("+", ",").split(",")
            if name.strip()
        ]
        if not self.dataset_names:
            raise ValueError("LongAlignSFTDataset requires at least one dataset name.")
        self.dataset_specs = []
        for name in self.dataset_names:
            if "::" in name:
                dataset_id, dataset_split = name.split("::", 1)
                self.dataset_specs.append((dataset_id.strip(), dataset_split.strip()))
            else:
                self.dataset_specs.append((name, split))
        self.split = split
        self.max_length = max_length
        self.max_examples = max_examples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.causal_lm = causal_lm
        self._stream_epoch = 0
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(
            f"Loading SFT dataset(s) "
            f"{', '.join(f'{name}::{dataset_split}' for name, dataset_split in self.dataset_specs)} "
            "(streaming=True)..."
        )

        print("SFT streaming dataset ready")

    def _fallback_format(self, messages):
        prompt_parts = []
        assistant_text = ""

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "assistant" and not assistant_text:
                assistant_text = content
                break
            prompt_parts.append(f"{role.capitalize()}: {content}\n")

        prompt = "".join(prompt_parts) + "Assistant:"
        full = prompt + " " + assistant_text + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=True)["input_ids"]
        return prompt_ids, full_ids

    def _normalize_role(self, role):
        role = str(role or "").lower()
        if role in {"human", "user", "prompter", "instruction", "input"}:
            return "user"
        if role in {"gpt", "assistant", "model", "bot", "output", "response"}:
            return "assistant"
        if role == "system":
            return "system"
        return role or "user"

    def _normalize_messages(self, messages):
        normalized = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = message.get("role", message.get("from", message.get("speaker", "")))
            content = message.get("content", message.get("value", message.get("text", "")))
            content = str(content or "").strip()
            if not content:
                continue
            normalized.append({
                "role": self._normalize_role(role),
                "content": content,
            })
        return normalized


    def _messages_from_item(self, item):
        if not isinstance(item, dict):
            return []

        for key in ("messages", "conversations"):
            messages = self._normalize_messages(item.get(key))
            if any(message["role"] == "assistant" for message in messages):
                return messages


        prompt = item.get("prompt") or item.get("instruction") or item.get("question") or item.get("input")
        response = (
            item.get("response")
            or item.get("output")
            or item.get("answer")
            or item.get("completion")
        )


        if prompt and response:
            return [
                {"role": "user", "content": str(prompt).strip()},
                {"role": "assistant", "content": str(response).strip()},
            ]
        return []

    def _chat_token_ids(self, messages):
        assistant_idx = next(
            (i for i, message in enumerate(messages) if message.get("role") == "assistant"),
            None,
        )
        if assistant_idx is None:
            return None, None

        if getattr(self.tokenizer, "chat_template", None):
            prompt_messages = messages[:assistant_idx]
            supervised_messages = messages[: assistant_idx + 1]
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = self.tokenizer.apply_chat_template(
                supervised_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            if full_ids[: len(prompt_ids)] == prompt_ids:
                return prompt_ids, full_ids

        return self._fallback_format(messages)


    def _tokenize_item(self, item, idx=None):
        messages = self._messages_from_item(item)
        if not messages:
            return None

        prompt_ids, full_ids = self._chat_token_ids(messages)
        if prompt_ids is None or full_ids is None:
            return None
        if len(prompt_ids) >= self.max_length:
            return None

        assistant_ids = full_ids[len(prompt_ids):]
        if not assistant_ids:
            assistant_ids = [self.tokenizer.eos_token_id]

        if len(full_ids) > self.max_length:
            if len(assistant_ids) >= self.max_length:
                input_ids = assistant_ids[-self.max_length:]
                labels = input_ids.copy()
            else:
                prompt_budget = self.max_length - len(assistant_ids)
                prompt_ids = prompt_ids[-prompt_budget:]
                input_ids = prompt_ids + assistant_ids
                labels = [-100] * len(prompt_ids) + assistant_ids.copy()
        else:
            input_ids = full_ids
            labels = [-100] * len(prompt_ids) + assistant_ids.copy()

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        attention_mask = [1] * len(input_ids)

        if self.causal_lm:
            labels = input_ids.copy()

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.long, device="cpu")
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device="cpu")
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")

        assert input_ids.shape[0] == self.max_length
        assert attention_mask.shape[0] == self.max_length
        assert labels.shape[0] == self.max_length
        assert (labels != -100).sum() > 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


    def _get_stream(self):
        rng = random.Random(self.seed + self._stream_epoch)
        streams = []
        for name, dataset_split in self.dataset_specs:
            ds = load_dataset(name, split=dataset_split, streaming=True)
            if self.shuffle_buffer_size and self.shuffle_buffer_size > 1:
                ds = ds.shuffle(
                    seed=self.seed + self._stream_epoch,
                    buffer_size=self.shuffle_buffer_size,
                )
            streams.append(iter(ds))
        self._stream_epoch += 1

        if len(streams) == 1:
            return streams[0]

        def mixed_stream():
            active = list(range(len(streams)))
            while active:
                stream_idx = rng.choice(active)
                try:
                    yield next(streams[stream_idx])
                except StopIteration:
                    active.remove(stream_idx)

        return mixed_stream()

    def __iter__(self):
        yielded = 0
        rank, world_size = _ddp_shard()
        source = self._get_stream()
        for idx, item in enumerate(source):
            # Shard by raw source index so every rank sees a disjoint subset.
            # Done before tokenization, which is the expensive part.
            if world_size > 1 and idx % world_size != rank:
                continue
            sample = self._tokenize_item(item, idx=idx)
            if sample is None:
                continue
            yield sample
            yielded += 1
            if self.max_examples is not None and yielded >= self.max_examples:
                break


class LoongRLSFTDataset(TorchDataset):
    """LoongRL prompts with loss restricted to canonical gold-answer tokens."""

    def __init__(
        self,
        dataset_name: str = "OldKingMeister/LoongRL-Train-Data",
        config_name: str = "hotpotqa_qwen_0_2500",
        tokenizer_name: str = "unsloth/Llama-3.2-1B-Instruct",
        max_length: int = 16384,
        split: str = "train",
        max_examples: int | None = None,
        memory_forced_query_tokens: int = 0,
    ):
        self.max_length = max_length
        self.memory_forced_query_tokens = int(memory_forced_query_tokens)
        if self.memory_forced_query_tokens < 0:
            raise ValueError("memory_forced_query_tokens must be non-negative")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(
            f"Loading LoongRL SFT dataset {dataset_name} "
            f"(config={config_name!r}, split={split!r})..."
        )
        self.dataset = load_dataset(dataset_name, config_name, split=split)
        if max_examples is not None:
            self.dataset = self.dataset.select(
                range(min(max_examples, len(self.dataset)))
            )
        print(f"LoongRL SFT dataset loaded: {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _prompt_messages(item):
        prompt = item.get("prompt", [])
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt.strip()}]
        messages = []
        for message in prompt or []:
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", "")).strip()
            if content:
                messages.append({
                    "role": str(message.get("role", "user") or "user"),
                    "content": content,
                })
        return messages

    def __getitem__(self, idx):
        item = self.dataset[idx]
        messages = self._prompt_messages(item)
        ground_truth = (item.get("reward_model", {}) or {}).get("ground_truth", [])
        if isinstance(ground_truth, str):
            answers = [ground_truth.strip()]
        else:
            answers = [str(answer).strip() for answer in ground_truth if str(answer).strip()]
        if not messages or not answers:
            raise ValueError(f"LoongRL example {idx} has no prompt or ground truth")

        if getattr(self.tokenizer, "chat_template", None):
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            prompt_text = "".join(
                f"{message['role'].capitalize()}: {message['content']}\n"
                for message in messages
            ) + "Assistant:"
            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=True,
            )["input_ids"]

        # The first answer is LoongRL's canonical target. Alternate answers are
        # useful for reward matching but cannot all be teacher-forced at once.
        answer_ids = self.tokenizer(
            answers[0],
            add_special_tokens=False,
        )["input_ids"]
        if not answer_ids:
            raise ValueError(f"LoongRL example {idx} has an empty tokenized answer")
        if len(answer_ids) >= self.max_length:
            raise ValueError(
                f"LoongRL example {idx} answer is too long: "
                f"{len(answer_ids)} >= {self.max_length}"
            )

        prompt_budget = self.max_length - len(answer_ids)
        if len(prompt_ids) > prompt_budget:
            # Retain BOS plus the prompt tail, which contains the question and
            # the model's assistant-generation marker.
            if prompt_budget > 1:
                prompt_ids = [prompt_ids[0]] + prompt_ids[-(prompt_budget - 1):]
            else:
                prompt_ids = prompt_ids[-prompt_budget:]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids.copy()
        attention_mask = [1] * len(input_ids)
        if self.memory_forced_query_tokens:
            # Keep the full token sequence available to Titan's online NMM store,
            # which does not consume the HF attention mask, but hide the old
            # context from all downstream native-attention layers. Only the tail
            # of the prompt (question/instructions) and the canonical answer stay
            # visible as ordinary attention keys. This removes the global-context
            # bypass while preserving a single differentiable forward pass.
            query_start = max(0, len(prompt_ids) - self.memory_forced_query_tokens)
            attention_mask[:query_start] = [0] * query_start
        pad_len = self.max_length - len(input_ids)
        if pad_len:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.long, device="cpu")
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device="cpu")
        assert input_ids.shape == labels.shape == attention_mask.shape == (self.max_length,)
        assert int((labels != -100).sum()) == len(answer_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MixedLongAlignLoongRLSFTDataset(IterableDataset):
    """Fixed-size, rank-sharded mix of LongAlign and LoongRL SFT samples.

    LongAlign remains streaming so an existing streaming cache is sufficient.
    LoongRL is map-style and each rank receives disjoint indices.  Every rank
    uses the same shuffled source schedule, keeping DDP collectives aligned
    while independently consuming its shard of each source.
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int,
        longalign_examples: int = 9984,
        loongrl_examples: int = 2496,
        loongrl_config_name: str = "hotpotqa_qwen_0_2500",
        loongrl_memory_forced_query_tokens: int = 0,
        seed: int = 42,
    ):
        self.longalign_examples = int(longalign_examples)
        self.loongrl_examples = int(loongrl_examples)
        self.seed = int(seed)
        self.longalign = LongAlignSFTDataset(
            dataset_name="zai-org/LongAlign-10k",
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split="train",
            causal_lm=False,
            seed=seed,
        )
        self.loongrl = LoongRLSFTDataset(
            dataset_name="OldKingMeister/LoongRL-Train-Data",
            config_name=loongrl_config_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            split="train",
            max_examples=loongrl_examples,
            memory_forced_query_tokens=loongrl_memory_forced_query_tokens,
        )
        if len(self.loongrl) != self.loongrl_examples:
            raise ValueError(
                f"Requested {self.loongrl_examples} LoongRL examples but loaded "
                f"{len(self.loongrl)}"
            )

    def __iter__(self):
        rank, world_size = _ddp_shard()
        if self.longalign_examples % world_size:
            raise ValueError("LongAlign example count must be divisible by DDP world size")
        if self.loongrl_examples % world_size:
            raise ValueError("LoongRL example count must be divisible by DDP world size")


        longalign_count = self.longalign_examples // world_size
        loongrl_indices = list(range(rank, self.loongrl_examples, world_size))
        rng = random.Random(self.seed)
        rng.shuffle(loongrl_indices)
        source_schedule = [0] * longalign_count + [1] * len(loongrl_indices)
        rng.shuffle(source_schedule)

        longalign_iter = iter(self.longalign)
        loongrl_iter = iter(loongrl_indices)
        for source in source_schedule:
            if source == 1:
                yield self.loongrl[next(loongrl_iter)]
                continue
            try:
                yield next(longalign_iter)
            except StopIteration:
                # A small number of over-length/invalid rows can make the
                # valid streaming epoch shorter than the requested 9,984.
                # Continue from a fresh deterministic shuffle to preserve the
                # exact 80/20 optimizer mix on every rank.
                longalign_iter = iter(self.longalign)
                yield next(longalign_iter)


