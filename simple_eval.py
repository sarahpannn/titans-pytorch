import math
from typing import Tuple, List, Dict, Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm


def load_small_boolq_winogrande(
    max_examples: int = 1000,
    boolq_split: str = "validation",
    winogrande_version: str = "winogrande_xl",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load small subsets of BoolQ and Winogrande as plain Python lists.

    Returns:
        boolq_data: list of {"passage", "question", "label" (bool)}
        winogrande_data: list of {"sentence", "option1", "option2", "answer" (0 or 1)}
    """
    # --- BoolQ ---
    boolq_raw = load_dataset("boolq", split=boolq_split)
    if max_examples is not None:
        boolq_raw = boolq_raw.select(range(min(max_examples, len(boolq_raw))))

    boolq_data: List[Dict] = []
    for ex in boolq_raw:
        boolq_data.append(
            {
                "passage": ex["passage"],
                "question": ex["question"],
                "label": bool(ex["answer"]),  # 1 = True, 0 = False
            }
        )

    # --- Winogrande ---
    winogrande_raw = load_dataset("winogrande", winogrande_version, split="validation")
    if max_examples is not None:
        winogrande_raw = winogrande_raw.select(
            range(min(max_examples, len(winogrande_raw)))
        )

    winogrande_data: List[Dict] = []
    for ex in winogrande_raw:
        winogrande_data.append(
            {
                "sentence": ex["sentence"],
                "option1": ex["option1"],
                "option2": ex["option2"],
                # HF stores answer as "1"/"2"
                "answer": int(ex["answer"]) - 1,  # -> 0 or 1
            }
        )

    return boolq_data, winogrande_data


def eval_boolq(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    boolq_data: List[Dict],
    device: torch.device | str = "cuda",
    batch_size: int = 8,
    max_input_len: int = 512,
) -> float:
    """
    Simple BoolQ eval using next-token scoring for ' yes' vs ' no'.
    Returns accuracy in [0, 1].

    max_input_len and batch_size should typically come from training config:
      - max_input_len ~ config.sequence_length
      - batch_size    ~ config.micro_batch_size (or smaller if OOM)
    """

    # Reset Titan memory once at the start of eval (safe even for non-Titan models)
    if hasattr(model, "reset_memory_states"):
        model.reset_memory_states()
    elif hasattr(model, "module") and hasattr(model.module, "reset_memory_states"):
        model.module.reset_memory_states()

    yes_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode(" no", add_special_tokens=False)[0]

    total = 0
    correct = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(boolq_data), batch_size), desc="BoolQ"):
            batch = boolq_data[i : i + batch_size]

            prompts = [
                f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer (yes or no):"
                for ex in batch
            ]
            labels = torch.tensor(
                [1 if ex["label"] else 0 for ex in batch],
                device=device,
                dtype=torch.long,
            )  # 1 = yes, 0 = no

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            # Extra safety if tokenizer ever returns 1D tensors
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

            # Optionally reset memory per batch to avoid cross-example bleed
            if hasattr(model, "reset_memory_states"):
                model.reset_memory_states()
            elif hasattr(model, "module") and hasattr(model.module, "reset_memory_states"):
                model.module.reset_memory_states()

            # print("DURING TEST: ", input_ids.shape)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # logits for the *next* token
            next_logits = outputs["logits"][:, -1, :]  # (batch, vocab)

            yes_logits = next_logits[:, yes_id]
            no_logits = next_logits[:, no_id]

            preds = (yes_logits > no_logits).long()  # shape (batch,)

            # Guard for shape bugs
            if preds.shape != labels.shape:
                raise RuntimeError(
                    f"BoolQ eval shape mismatch: preds {preds.shape}, labels {labels.shape}"
                )

            total += labels.numel()
            correct += (preds == labels).sum().item()

    return correct / max(1, total)


def _sequence_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    device: torch.device,
    max_length: int = 256,
) -> float:
    """
    Log p(text) under the model, summed over tokens.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = enc["input_ids"]  # (1, T)

    # Safety: reset memory before each sequence for Titan
    if hasattr(model, "reset_memory_states"):
        model.reset_memory_states()
    elif hasattr(model, "module") and hasattr(model.module, "reset_memory_states"):
        model.module.reset_memory_states()

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs["logits"][:, :-1, :]  # predict token t+1 from t
        log_probs = torch.log_softmax(logits, dim=-1)  # (1, T-1, V)
        target = input_ids[:, 1:]  # shift by 1

        token_logprobs = log_probs.gather(
            2, target.unsqueeze(-1)
        ).squeeze(-1)  # (1, T-1)

        return token_logprobs.sum().item()


def eval_winogrande(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    winogrande_data: List[Dict],
    device: torch.device | str = "cuda",
    max_input_len: int = 256,
) -> float:
    """
    Winogrande eval by comparing log p(sentence with option1) vs log p(sentence with option2).
    Returns accuracy in [0, 1].

    max_input_len can be set based on training seq_len, but 256 is usually enough.
    """
    model.eval()
    device = torch.device(device)

    # Reset memory at start of eval
    if hasattr(model, "reset_memory_states"):
        model.reset_memory_states()
    elif hasattr(model, "module") and hasattr(model.module, "reset_memory_states"):
        model.module.reset_memory_states()

    total = 0
    correct = 0

    with torch.no_grad():
        for ex in tqdm(winogrande_data, desc="Winogrande"):
            sent_template = ex["sentence"]

            # Replace the blank with each option
            sent1 = sent_template.replace("_", ex["option1"])
            sent2 = sent_template.replace("_", ex["option2"])

            lp1 = _sequence_logprob(
                model, tokenizer, sent1, device, max_length=max_input_len
            )
            lp2 = _sequence_logprob(
                model, tokenizer, sent2, device, max_length=max_input_len
            )

            pred = 0 if lp1 >= lp2 else 1
            label = ex["answer"]  # 0 or 1

            total += 1
            if pred == label:
                correct += 1

    return correct / max(1, total)


def eval_winogrande_boolq(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device | str = "cuda",
    max_examples: int = 1000,
    *,
    # NEW: hook these up to training config
    seq_len: Optional[int] = None,
    boolq_batch_size: Optional[int] = None,
    winogrande_max_len: Optional[int] = None,
) -> dict[str, float]:
    """
    Convenience wrapper: load up to `max_examples` from BoolQ + Winogrande
    and return both accuracies.

    Pass training args in here, e.g.:

        eval_winogrande_boolq(
            model, tokenizer,
            device=device,
            max_examples=config.intermittent_eval_limit,
            seq_len=config.sequence_length,
            boolq_batch_size=config.micro_batch_size,
        )
    """
    # Sensible defaults if you *don't* pass training args
    if seq_len is None:
        seq_len = 8192  # default to your training seq len
    if boolq_batch_size is None:
        boolq_batch_size = 8
    if winogrande_max_len is None:
        # WG sentences are short; no need for full 8k
        winogrande_max_len = min(512, seq_len)

    boolq_data, winogrande_data = load_small_boolq_winogrande(
        max_examples=max_examples
    )

    print("evaluating BoolQ...")
    boolq_acc = eval_boolq(
        model,
        tokenizer,
        boolq_data,
        device=device,
        batch_size=boolq_batch_size,
        max_input_len=seq_len,
    )

    print("evaluating Winogrande...")
    winogrande_acc = eval_winogrande(
        model,
        tokenizer,
        winogrande_data,
        device=device,
        max_input_len=winogrande_max_len,
    )

    return {
        "boolq_acc": boolq_acc,
        "winogrande_acc": winogrande_acc,
    }
