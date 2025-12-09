"""
Unified evaluation module for multiple-choice NLP benchmarks.

All evaluations use the same core approach:
- Batched sequence log-probability scoring via `_batched_sequence_logprob`
- Consistent memory reset handling for Titan models
- Unified interface via `MultipleChoiceEvaluator`

Supported datasets: BoolQ, Winogrande, PubMedQA, MMLU, HellaSwag, ARC, PIQA, AQuA-RAT
"""

import math
import time
import gc
import sys
import json
import os
import tempfile
import subprocess
from typing import Tuple, List, Dict, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoTokenizer
from tqdm import tqdm

try:
    from lm_eval import evaluator, tasks
    from lm_eval.api.model import LM
except ImportError:
    print("Warning: lm-eval not available. Install with: pip install lm-eval==0.4.2")
    evaluator = tasks = LM = None

try:
    from baseline_eval import TitanSegmentedLM
except ImportError:
    print("Warning: baseline_eval not available for advanced evaluation functions")
    TitanSegmentedLM = None


# =============================================================================
# Core batched log-probability computation
# =============================================================================

def _reset_memory(model: PreTrainedModel) -> None:
    """Reset Titan memory states if available."""
    if hasattr(model, "reset_memory_states"):
        model.reset_memory_states()
    elif hasattr(model, "module") and hasattr(model.module, "reset_memory_states"):
        model.module.reset_memory_states()


def _batched_sequence_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
    reset_memory: bool = True,
) -> List[float]:
    """
    Compute log p(text) for a batch of texts, summed over tokens.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of text strings to score
        device: Device to run on
        max_length: Maximum sequence length
        reset_memory: Whether to reset Titan memory before computation
    
    Returns:
        List of log probabilities (one per text)
    """
    if not texts:
        return []
    
    if reset_memory:
        _reset_memory(model)
    
    # Tokenize batch with padding
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    
    input_ids = enc["input_ids"]  # (B, T)
    attention_mask = enc["attention_mask"]  # (B, T)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        
        # Shift for next-token prediction: logits[:, :-1] predicts input_ids[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = input_ids[:, 1:].contiguous()   # (B, T-1)
        shift_mask = attention_mask[:, 1:].contiguous()  # (B, T-1)
        
        # Compute log probabilities
        log_probs = torch.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)
        
        # Gather the log probs for actual tokens
        token_logprobs = log_probs.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)
        
        # Mask out padding tokens and sum
        token_logprobs = token_logprobs * shift_mask.float()
        sequence_logprobs = token_logprobs.sum(dim=-1)  # (B,)
        
        return sequence_logprobs.tolist()


def _sequence_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    device: torch.device,
    max_length: int = 256,
) -> float:
    """
    Log p(text) under the model, summed over tokens.
    Single-sequence wrapper for backward compatibility.
    """
    return _batched_sequence_logprob(
        model, tokenizer, [text], device, max_length, reset_memory=True
    )[0]


# =============================================================================
# Data loading functions
# =============================================================================

def load_boolq(max_examples: Optional[int] = None, split: str = "validation") -> List[Dict]:
    """Load BoolQ dataset."""
    ds = load_dataset("boolq", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    return [
        {
            "passage": ex["passage"],
            "question": ex["question"],
            "label": int(ex["answer"]),  # 1 = True, 0 = False
        }
        for ex in ds
    ]


def load_winogrande(
    max_examples: Optional[int] = None,
    version: str = "winogrande_xl",
    split: str = "validation",
) -> List[Dict]:
    """Load Winogrande dataset."""
    ds = load_dataset("winogrande", version, split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    return [
        {
            "sentence": ex["sentence"],
            "option1": ex["option1"],
            "option2": ex["option2"],
            "answer": int(ex["answer"]) - 1,  # Convert "1"/"2" to 0/1
        }
        for ex in ds
    ]


def load_pubmedqa(
    max_examples: Optional[int] = None,
    config_name: str = "pqa_labeled",
    split: str = "train",
) -> List[Dict]:
    """Load PubMedQA dataset."""
    ds = load_dataset("pubmed_qa", config_name, split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    data = []
    for ex in ds:
        context = ex.get("context", "")
        if isinstance(context, dict):
            contexts = context.get("contexts", "")
        else:
            contexts = context
        
        if isinstance(contexts, (list, tuple)):
            context_text = " ".join(contexts)
        else:
            context_text = str(contexts)
        
        data.append({
            "question": ex["question"],
            "context": context_text,
            "long_answer": ex.get("long_answer", ""),
            "label": ex["final_decision"].strip().lower(),  # "yes", "no", "maybe"
        })
    
    return data


def load_mmlu(
    max_examples: Optional[int] = None,
    subject: str = "all",
    split: str = "test",
) -> List[Dict]:
    """Load MMLU dataset."""
    if subject == "all":
        ds = load_dataset("cais/mmlu", "all", split=split)
    else:
        ds = load_dataset("cais/mmlu", subject, split=split)
    
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    return [
        {
            "question": ex["question"],
            "choices": ex["choices"],
            "answer": ex["answer"],  # 0-3
            "subject": ex.get("subject", subject),
        }
        for ex in ds
    ]


def load_hellaswag(max_examples: Optional[int] = None, split: str = "validation") -> List[Dict]:
    """Load HellaSwag dataset."""
    ds = load_dataset("Rowan/hellaswag", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    return [
        {
            "context": ex["ctx"],
            "endings": ex["endings"],
            "answer": int(ex["label"]),
        }
        for ex in ds
    ]


def load_arc(
    max_examples: Optional[int] = None,
    challenge: bool = True,
    split: str = "test",
) -> List[Dict]:
    """Load ARC dataset (Challenge or Easy)."""
    config = "ARC-Challenge" if challenge else "ARC-Easy"
    ds = load_dataset("allenai/ai2_arc", config, split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    
    return [
        {
            "question": ex["question"],
            "choices": ex["choices"]["text"],
            "answer": label_map.get(ex["answerKey"], 0),
        }
        for ex in ds
    ]


def load_piqa(max_examples: Optional[int] = None, split: str = "validation") -> List[Dict]:
    """Load PIQA dataset."""
    ds = load_dataset("piqa", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    return [
        {
            "goal": ex["goal"],
            "choices": [ex["sol1"], ex["sol2"]],
            "answer": ex["label"],  # 0 or 1
        }
        for ex in ds
    ]


def load_aqua_rat(max_examples: Optional[int] = None, split: str = "validation") -> List[Dict]:
    """Load AQuA-RAT dataset (algebraic reasoning with rationales)."""
    ds = load_dataset("deepmind/aqua_rat", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    return [
        {
            "question": ex["question"],
            "choices": ex["options"],  # List of 5 options (A-E)
            "answer": ord(ex["correct"]) - ord("A"),  # Convert "A"-"E" to 0-4
            "rationale": ex.get("rationale", ""),
        }
        for ex in ds
    ]


def load_casehold(max_examples: Optional[int] = None, split: str = "validation") -> List[Dict]:
    """Load LexGLUE CaseHOLD dataset."""
    ds = load_dataset("coastalcph/lex_glue", "case_hold", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    data = []
    for ex in ds:
        # CaseHOLD stores options in 'endings' field
        options = ex.get("endings", None)
        if options is None:
            options = ex.get("sequence", None)
        if options is None:
            raise KeyError("Expected 'endings' or 'sequence' field in CaseHOLD example.")
        
        data.append({
            "context": ex["context"],
            "choices": options,  # List of 5 holdings
            "answer": int(ex["label"]),  # 0-4
        })
    
    return data


def load_commonsenseqa(max_examples: Optional[int] = None, split: str = "validation") -> List[Dict]:
    """Load CommonsenseQA dataset."""
    ds = load_dataset("tau/commonsense_qa", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    
    return [
        {
            "question": ex["question"],
            "choices": ex["choices"]["text"],
            "choice_labels": ex["choices"]["label"],  # ['A', 'B', 'C', 'D', 'E']
            "answer": label_map.get(ex["answerKey"], 0),
        }
        for ex in ds
    ]


def load_drop(max_examples: Optional[int] = None, split: str = "validation") -> List[Dict]:
    """
    Load DROP dataset.
    
    DROP is an extractive/generative QA task, but we convert it to multiple-choice
    by treating the gold answers as one option and generating distractors.
    For simplicity, we evaluate using exact match on generated text.
    """
    ds = load_dataset("ucinlp/drop", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    data = []
    for ex in ds:
        # DROP has multiple valid answers
        answers_spans = ex.get("answers_spans", {})
        spans = answers_spans.get("spans", [])
        
        # Also check for number answers
        number_answer = ex.get("answer", {}).get("number", "")
        
        # Collect all valid answers
        valid_answers = []
        if spans:
            valid_answers.extend(spans)
        if number_answer:
            valid_answers.append(str(number_answer))
        
        if not valid_answers:
            continue
        
        data.append({
            "passage": ex["passage"],
            "question": ex["question"],
            "answers": valid_answers,  # List of acceptable answers
        })
    
    return data


def load_squadv2(max_examples: Optional[int] = None, split: str = "validation") -> List[Dict]:
    """
    Load SQuAD v2 dataset.
    
    SQuAD v2 includes unanswerable questions (where answers is empty).
    """
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    data = []
    for ex in ds:
        answers = ex["answers"]["text"]
        is_impossible = len(answers) == 0
        
        data.append({
            "context": ex["context"],
            "question": ex["question"],
            "answers": answers if answers else [""],  # Empty string for unanswerable
            "is_impossible": is_impossible,
        })
    
    return data


# =============================================================================
# Unified Multiple Choice Evaluator
# =============================================================================

# Safe default batch sizes for evaluation (to prevent OOM when training batch sizes are passed)
# These are conservative defaults that work on most GPUs
EVAL_BATCH_SIZE_DEFAULTS = {
    "boolq": 8,
    "winogrande": 16,
    "pubmedqa": 2,      # Long contexts
    "mmlu": 4,
    "hellaswag": 4,
    "arc": 8,
    "arc_easy": 8,
    "arc_challenge": 8,
    "piqa": 8,
    "aqua_rat": 4,       # Math reasoning
    "casehold": 2,       # Long legal contexts
    "commonsenseqa": 8,
    "drop": 2,           # Generative, long contexts
    "squadv2": 2,        # Generative, long contexts
}

# Maximum batch size for evaluation (safety cap)
MAX_EVAL_BATCH_SIZE = 16


def _get_safe_batch_size(batch_size: int, dataset: Optional[str] = None) -> int:
    """
    Clamp batch size to a safe value for evaluation.
    
    Training batch sizes are often large (32, 64, etc.) but evaluation needs
    to score multiple candidates per example, which multiplies memory usage.
    
    Args:
        batch_size: Requested batch size
        dataset: Optional dataset name for dataset-specific defaults
    
    Returns:
        Safe batch size clamped to reasonable limits
    """
    if dataset and dataset in EVAL_BATCH_SIZE_DEFAULTS:
        max_for_dataset = EVAL_BATCH_SIZE_DEFAULTS[dataset]
    else:
        max_for_dataset = MAX_EVAL_BATCH_SIZE
    
    return min(batch_size, max_for_dataset)


@dataclass
class EvalResult:
    """Result from a single evaluation."""
    accuracy: float
    total: int
    correct: int
    per_example: Optional[List[Dict]] = None  # For detailed analysis


class MultipleChoiceEvaluator:
    """
    Unified evaluator for multiple-choice benchmarks using sequence log-probabilities.
    
    All tasks are evaluated by:
    1. Constructing full sequences for each choice
    2. Computing log p(sequence) for each choice
    3. Selecting the highest-scoring choice as the prediction
    
    This approach is consistent with how language models naturally work and
    allows for efficient batched evaluation.
    
    Note: batch_size is automatically clamped to safe values to prevent OOM.
    If you pass your training micro_batch_size, it will be reduced appropriately.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Union[torch.device, str] = "cuda",
        batch_size: int = 8,
        max_length: int = 512,
        show_progress: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self._requested_batch_size = batch_size  # Store original request
        self.batch_size = min(batch_size, MAX_EVAL_BATCH_SIZE)  # Apply global cap
        self.max_length = max_length
        self.show_progress = show_progress
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _format_boolq(self, ex: Dict) -> Tuple[List[str], int]:
        """Format BoolQ example into candidate sequences."""
        prompt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
        candidates = [f"{prompt} no", f"{prompt} yes"]  # 0=no, 1=yes
        return candidates, ex["label"]
    
    def _format_winogrande(self, ex: Dict) -> Tuple[List[str], int]:
        """Format Winogrande example into candidate sequences."""
        sent1 = ex["sentence"].replace("_", ex["option1"])
        sent2 = ex["sentence"].replace("_", ex["option2"])
        return [sent1, sent2], ex["answer"]
    
    def _format_pubmedqa(self, ex: Dict, include_long_answer: bool = False) -> Tuple[List[str], int]:
        """Format PubMedQA example into candidate sequences."""
        prompt_parts = [
            f"Question: {ex['question']}",
            f"Context: {ex['context']}",
        ]
        if include_long_answer and ex.get("long_answer"):
            prompt_parts.append(f"Long answer: {ex['long_answer']}")
        
        prompt = "\n".join(prompt_parts) + "\nFinal decision (yes / no / maybe):"
        
        answer_texts = ["yes", "no", "maybe"]
        candidates = [f"{prompt} {ans}" for ans in answer_texts]
        
        label_map = {"yes": 0, "no": 1, "maybe": 2}
        label = label_map.get(ex["label"], -1)
        
        return candidates, label
    
    def _format_mmlu(self, ex: Dict) -> Tuple[List[str], int]:
        """Format MMLU example into candidate sequences."""
        question = ex["question"]
        choices = ex["choices"]
        
        prompt = f"Question: {question}\n"
        candidates = []
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            candidates.append(f"{prompt}Answer: {letter}. {choice}")
        
        return candidates, ex["answer"]
    
    def _format_hellaswag(self, ex: Dict) -> Tuple[List[str], int]:
        """Format HellaSwag example into candidate sequences."""
        context = ex["context"]
        candidates = [f"{context} {ending}" for ending in ex["endings"]]
        return candidates, ex["answer"]
    
    def _format_arc(self, ex: Dict) -> Tuple[List[str], int]:
        """Format ARC example into candidate sequences."""
        question = ex["question"]
        choices = ex["choices"]
        
        prompt = f"Question: {question}\n"
        candidates = []
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            candidates.append(f"{prompt}Answer: {letter}. {choice}")
        
        return candidates, ex["answer"]
    
    def _format_piqa(self, ex: Dict) -> Tuple[List[str], int]:
        """Format PIQA example into candidate sequences."""
        goal = ex["goal"]
        candidates = [f"Goal: {goal}\nSolution: {choice}" for choice in ex["choices"]]
        return candidates, ex["answer"]
    
    def _format_aqua_rat(self, ex: Dict) -> Tuple[List[str], int]:
        """Format AQuA-RAT example into candidate sequences."""
        question = ex["question"]
        choices = ex["choices"]
        
        prompt = f"Question: {question}\n"
        candidates = []
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            candidates.append(f"{prompt}Answer: {letter}. {choice}")
        
        return candidates, ex["answer"]
    
    def _format_casehold(self, ex: Dict) -> Tuple[List[str], int]:
        """Format CaseHOLD example into candidate sequences."""
        context = ex["context"]
        choices = ex["choices"]
        option_letters = ["A", "B", "C", "D", "E"]
        
        # Build the prompt with all options listed
        lines = [f"Context: {context}"]
        for i, opt in enumerate(choices):
            letter = option_letters[i]
            lines.append(f"Option {letter}: {opt}")
        prompt = "\n".join(lines) + "\nAnswer (A, B, C, D, or E):"
        
        # Each candidate is the prompt + the answer letter
        candidates = [f"{prompt} {option_letters[i]}" for i in range(len(choices))]
        return candidates, ex["answer"]
    
    def _format_commonsenseqa(self, ex: Dict) -> Tuple[List[str], int]:
        """Format CommonsenseQA example into candidate sequences."""
        question = ex["question"]
        choices = ex["choices"]
        choice_labels = ex.get("choice_labels", ["A", "B", "C", "D", "E"][:len(choices)])
        
        prompt = f"Question: {question}\n"
        candidates = []
        for i, choice in enumerate(choices):
            letter = choice_labels[i] if i < len(choice_labels) else chr(ord('A') + i)
            candidates.append(f"{prompt}Answer: {letter}. {choice}")
        
        return candidates, ex["answer"]
    
    def _evaluate_batch(
        self,
        examples: List[Dict],
        format_fn,
        format_kwargs: Optional[Dict] = None,
    ) -> List[Tuple[int, int]]:
        """
        Evaluate a batch of examples.
        
        Returns list of (prediction, label) tuples.
        """
        format_kwargs = format_kwargs or {}
        results = []
        
        # Format all examples
        all_candidates = []
        all_labels = []
        example_indices = []  # Track which candidates belong to which example
        
        for i, ex in enumerate(examples):
            candidates, label = format_fn(ex, **format_kwargs) if format_kwargs else format_fn(ex)
            if label == -1:  # Skip invalid examples
                continue
            all_candidates.extend(candidates)
            all_labels.append(label)
            example_indices.append((len(all_candidates) - len(candidates), len(all_candidates)))
        
        if not all_candidates:
            return []
        
        # Compute log probs for all candidates at once
        all_logprobs = _batched_sequence_logprob(
            self.model,
            self.tokenizer,
            all_candidates,
            self.device,
            self.max_length,
            reset_memory=True,
        )
        
        # Group by example and find predictions
        for (start, end), label in zip(example_indices, all_labels):
            candidate_logprobs = all_logprobs[start:end]
            pred = int(torch.tensor(candidate_logprobs).argmax().item())
            results.append((pred, label))
        
        return results
    
    def evaluate(
        self,
        dataset: str,
        data: Optional[List[Dict]] = None,
        max_examples: Optional[int] = None,
        **kwargs,
    ) -> EvalResult:
        """
        Evaluate on a dataset.
        
        Args:
            dataset: Name of dataset ("boolq", "winogrande", "pubmedqa", "mmlu", 
                     "hellaswag", "arc", "arc_easy", "piqa")
            data: Pre-loaded data (optional, will load if not provided)
            max_examples: Maximum examples to evaluate
            **kwargs: Additional arguments passed to format function
        
        Returns:
            EvalResult with accuracy and counts
        """
        # Load data if not provided
        if data is None:
            loaders = {
                "boolq": load_boolq,
                "winogrande": load_winogrande,
                "pubmedqa": load_pubmedqa,
                "mmlu": load_mmlu,
                "hellaswag": load_hellaswag,
                "arc": lambda **kw: load_arc(challenge=True, **kw),
                "arc_easy": lambda **kw: load_arc(challenge=False, **kw),
                "arc_challenge": lambda **kw: load_arc(challenge=True, **kw),
                "piqa": load_piqa,
                "aqua_rat": load_aqua_rat,
                "casehold": load_casehold,
                "commonsenseqa": load_commonsenseqa,
            }
            if dataset not in loaders:
                raise ValueError(f"Unknown dataset: {dataset}. Available: {list(loaders.keys())}")
            data = loaders[dataset](max_examples=max_examples)
        elif max_examples is not None:
            data = data[:max_examples]
        
        # Get format function
        formatters = {
            "boolq": self._format_boolq,
            "winogrande": self._format_winogrande,
            "pubmedqa": self._format_pubmedqa,
            "mmlu": self._format_mmlu,
            "hellaswag": self._format_hellaswag,
            "arc": self._format_arc,
            "arc_easy": self._format_arc,
            "arc_challenge": self._format_arc,
            "piqa": self._format_piqa,
            "aqua_rat": self._format_aqua_rat,
            "casehold": self._format_casehold,
            "commonsenseqa": self._format_commonsenseqa,
        }
        format_fn = formatters[dataset]
        
        # Get safe batch size for this dataset
        effective_batch_size = _get_safe_batch_size(self._requested_batch_size, dataset)
        
        # Evaluate in batches
        was_training = self.model.training
        self.model.eval()
        
        all_results = []
        iterator = range(0, len(data), effective_batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc=f"Evaluating {dataset}")
        
        for i in iterator:
            batch = data[i:i + effective_batch_size]
            batch_results = self._evaluate_batch(batch, format_fn, kwargs)
            all_results.extend(batch_results)
        
        if was_training:
            self.model.train()
        
        # Compute accuracy
        correct = sum(1 for pred, label in all_results if pred == label)
        total = len(all_results)
        accuracy = correct / total if total > 0 else 0.0
        
        return EvalResult(
            accuracy=accuracy,
            total=total,
            correct=correct,
            per_example=[{"pred": p, "label": l} for p, l in all_results],
        )
    
    def evaluate_multiple(
        self,
        datasets: List[str],
        max_examples: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, EvalResult]:
        """
        Evaluate on multiple datasets.
        
        Returns:
            Dictionary mapping dataset name to EvalResult
        """
        results = {}
        for dataset in datasets:
            results[dataset] = self.evaluate(dataset, max_examples=max_examples, **kwargs)
        return results


# =============================================================================
# Generative Evaluation (for DROP, SQuAD v2)
# =============================================================================

def _normalize_answer(s: str) -> str:
    """Normalize answer for comparison (lowercase, strip, remove articles/punctuation)."""
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)
    
    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Check if prediction exactly matches any ground truth."""
    pred_norm = _normalize_answer(prediction)
    return float(any(_normalize_answer(gt) == pred_norm for gt in ground_truths))


def _compute_f1_max(prediction: str, ground_truths: List[str]) -> float:
    """Compute max F1 across all ground truths."""
    return max(_compute_f1(prediction, gt) for gt in ground_truths)


def _generate_answers_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 32,
    max_input_len: int = 512,
) -> List[str]:
    """Generate answers for a batch of prompts."""
    _reset_memory(model)
    
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len,
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    input_len = enc["input_ids"].shape[1]
    generated = outputs[:, input_len:]
    answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
    
    # Clean up answers (take first line, strip)
    cleaned = []
    for ans in answers:
        ans = ans.strip()
        if "\n" in ans:
            ans = ans.split("\n")[0]
        cleaned.append(ans)
    
    return cleaned


@dataclass
class GenerativeEvalResult:
    """Result from generative evaluation."""
    exact_match: float
    f1: float
    total: int
    per_example: Optional[List[Dict]] = None


class GenerativeEvaluator:
    """
    Evaluator for generative QA tasks (DROP, SQuAD v2).
    
    Uses text generation and computes exact match / F1 metrics.
    
    Note: batch_size is automatically clamped to safe values to prevent OOM.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Union[torch.device, str] = "cuda",
        batch_size: int = 4,
        max_input_len: int = 512,
        max_new_tokens: int = 32,
        show_progress: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self._requested_batch_size = batch_size
        self.batch_size = min(batch_size, MAX_EVAL_BATCH_SIZE)
        self.max_input_len = max_input_len
        self.max_new_tokens = max_new_tokens
        self.show_progress = show_progress
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _format_drop(self, ex: Dict) -> Tuple[str, List[str]]:
        """Format DROP example into prompt and ground truth answers."""
        prompt = f"Passage: {ex['passage']}\n\nQuestion: {ex['question']}\n\nAnswer:"
        return prompt, ex["answers"]
    
    def _format_squadv2(self, ex: Dict) -> Tuple[str, List[str]]:
        """Format SQuAD v2 example into prompt and ground truth answers."""
        prompt = f"Context: {ex['context']}\n\nQuestion: {ex['question']}\n\nAnswer:"
        # For unanswerable questions, the expected answer is empty/unanswerable
        if ex["is_impossible"]:
            return prompt, ["unanswerable", "no answer", ""]
        return prompt, ex["answers"]
    
    def evaluate(
        self,
        dataset: str,
        data: Optional[List[Dict]] = None,
        max_examples: Optional[int] = None,
    ) -> GenerativeEvalResult:
        """
        Evaluate on a generative QA dataset.
        
        Args:
            dataset: "drop" or "squadv2"
            data: Pre-loaded data (optional)
            max_examples: Maximum examples to evaluate
        
        Returns:
            GenerativeEvalResult with EM, F1, and counts
        """
        # Load data if not provided
        if data is None:
            loaders = {
                "drop": load_drop,
                "squadv2": load_squadv2,
            }
            if dataset not in loaders:
                raise ValueError(f"Unknown generative dataset: {dataset}")
            data = loaders[dataset](max_examples=max_examples)
        elif max_examples is not None:
            data = data[:max_examples]
        
        formatters = {
            "drop": self._format_drop,
            "squadv2": self._format_squadv2,
        }
        format_fn = formatters[dataset]
        
        # Get safe batch size for this dataset
        effective_batch_size = _get_safe_batch_size(self._requested_batch_size, dataset)
        
        was_training = self.model.training
        self.model.eval()
        
        all_em = []
        all_f1 = []
        per_example = []
        
        iterator = range(0, len(data), effective_batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc=f"Evaluating {dataset}")
        
        for i in iterator:
            batch = data[i:i + effective_batch_size]
            
            prompts = []
            ground_truths = []
            for ex in batch:
                prompt, answers = format_fn(ex)
                prompts.append(prompt)
                ground_truths.append(answers)
            
            predictions = _generate_answers_batch(
                self.model,
                self.tokenizer,
                prompts,
                self.device,
                max_new_tokens=self.max_new_tokens,
                max_input_len=self.max_input_len,
            )
            
            for pred, gts in zip(predictions, ground_truths):
                em = _compute_exact_match(pred, gts)
                f1 = _compute_f1_max(pred, gts)
                all_em.append(em)
                all_f1.append(f1)
                per_example.append({"prediction": pred, "ground_truths": gts, "em": em, "f1": f1})
        
        if was_training:
            self.model.train()
        
        return GenerativeEvalResult(
            exact_match=sum(all_em) / len(all_em) if all_em else 0.0,
            f1=sum(all_f1) / len(all_f1) if all_f1 else 0.0,
            total=len(all_em),
            per_example=per_example,
        )


# =============================================================================
# Convenience functions (backward compatible API)
# =============================================================================

def eval_boolq(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    boolq_data: Optional[List[Dict]] = None,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 8,
    max_input_len: int = 512,
    max_examples: Optional[int] = None,
) -> float:
    """
    Evaluate BoolQ using sequence log-probability scoring.
    Returns accuracy in [0, 1].
    """
    evaluator = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_input_len,
    )
    result = evaluator.evaluate("boolq", data=boolq_data, max_examples=max_examples)
    return result.accuracy


def eval_winogrande(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    winogrande_data: Optional[List[Dict]] = None,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 8,
    max_input_len: int = 256,
    max_examples: Optional[int] = None,
) -> float:
    """
    Evaluate Winogrande using sequence log-probability scoring.
    Returns accuracy in [0, 1].
    """
    evaluator = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_input_len,
    )
    result = evaluator.evaluate("winogrande", data=winogrande_data, max_examples=max_examples)
    return result.accuracy


def eval_pubmedqa(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 4,
    max_input_len: int = 1024,
    max_examples: Optional[int] = None,
    include_long_answer_in_prompt: bool = False,
) -> Dict[str, float]:
    """
    Evaluate PubMedQA as a 3-way multiple choice task.
    Returns {"pubmedqa_acc": float}.
    """
    evaluator_obj = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_input_len,
    )
    result = evaluator_obj.evaluate(
        "pubmedqa",
        max_examples=max_examples,
        include_long_answer=include_long_answer_in_prompt,
    )
    return {"pubmedqa_acc": result.accuracy}


def eval_mmlu(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 4,
    max_input_len: int = 512,
    max_examples: Optional[int] = None,
    subject: str = "all",
) -> Dict[str, float]:
    """
    Evaluate MMLU.
    Returns {"mmlu_acc": float}.
    """
    data = load_mmlu(max_examples=max_examples, subject=subject)
    evaluator_obj = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_input_len,
    )
    result = evaluator_obj.evaluate("mmlu", data=data)
    return {"mmlu_acc": result.accuracy}


def eval_hellaswag(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 4,
    max_input_len: int = 512,
    max_examples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate HellaSwag.
    Returns {"hellaswag_acc": float}.
    """
    evaluator_obj = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_input_len,
    )
    result = evaluator_obj.evaluate("hellaswag", max_examples=max_examples)
    return {"hellaswag_acc": result.accuracy}


def eval_arc(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 4,
    max_input_len: int = 512,
    max_examples: Optional[int] = None,
    challenge: bool = True,
) -> Dict[str, float]:
    """
    Evaluate ARC (Challenge or Easy).
    Returns {"arc_acc": float} or {"arc_easy_acc": float}.
    """
    dataset_name = "arc_challenge" if challenge else "arc_easy"
    evaluator_obj = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_input_len,
    )
    result = evaluator_obj.evaluate(dataset_name, max_examples=max_examples)
    key = "arc_acc" if challenge else "arc_easy_acc"
    return {key: result.accuracy}


def eval_piqa(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 4,
    max_input_len: int = 512,
    max_examples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate PIQA.
    Returns {"piqa_acc": float}.
    """
    evaluator_obj = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_input_len,
    )
    result = evaluator_obj.evaluate("piqa", max_examples=max_examples)
    return {"piqa_acc": result.accuracy}


def eval_aqua_rat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 4,
    max_input_len: int = 512,
    max_examples: Optional[int] = None,
    split: str = "validation",
) -> Dict[str, float]:
    """
    Evaluate AQuA-RAT (algebraic reasoning with rationales).
    Returns {"aqua_rat_acc": float}.
    """
    data = load_aqua_rat(max_examples=max_examples, split=split)
    evaluator_obj = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_input_len,
    )
    result = evaluator_obj.evaluate("aqua_rat", data=data)
    return {"aqua_rat_acc": result.accuracy}


def eval_casehold(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 4,
    max_input_len: int = 512,
    max_examples: Optional[int] = None,
    split: str = "validation",
) -> Dict[str, float]:
    """
    Evaluate LexGLUE CaseHOLD (5-way multiple choice).
    Returns {"casehold_acc": float}.
    """
    data = load_casehold(max_examples=max_examples, split=split)
    evaluator_obj = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_input_len,
    )
    result = evaluator_obj.evaluate("casehold", data=data)
    return {"casehold_acc": result.accuracy}


def eval_commonsenseqa(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 4,
    max_input_len: int = 512,
    max_examples: Optional[int] = None,
    split: str = "validation",
) -> Dict[str, float]:
    """
    Evaluate CommonsenseQA (5-way multiple choice).
    Returns {"commonsenseqa_acc": float}.
    """
    data = load_commonsenseqa(max_examples=max_examples, split=split)
    evaluator_obj = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_input_len,
    )
    result = evaluator_obj.evaluate("commonsenseqa", data=data)
    return {"commonsenseqa_acc": result.accuracy}


def eval_drop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 4,
    max_input_len: int = 512,
    max_new_tokens: int = 32,
    max_examples: Optional[int] = None,
    split: str = "validation",
) -> Dict[str, float]:
    """
    Evaluate DROP (generative QA).
    Returns {"drop_em": float, "drop_f1": float}.
    """
    data = load_drop(max_examples=max_examples, split=split)
    evaluator_obj = GenerativeEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_input_len=max_input_len,
        max_new_tokens=max_new_tokens,
    )
    result = evaluator_obj.evaluate("drop", data=data)
    return {"drop_em": result.exact_match, "drop_f1": result.f1}


def eval_squadv2(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 4,
    max_input_len: int = 512,
    max_new_tokens: int = 32,
    max_examples: Optional[int] = None,
    split: str = "validation",
) -> Dict[str, float]:
    """
    Evaluate SQuAD v2 (generative QA with unanswerable questions).
    Returns {"squadv2_em": float, "squadv2_f1": float}.
    """
    data = load_squadv2(max_examples=max_examples, split=split)
    evaluator_obj = GenerativeEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_input_len=max_input_len,
        max_new_tokens=max_new_tokens,
    )
    result = evaluator_obj.evaluate("squadv2", data=data)
    return {"squadv2_em": result.exact_match, "squadv2_f1": result.f1}


# =============================================================================
# Combined evaluation functions
# =============================================================================

def load_small_boolq_winogrande(
    max_examples: int = 1000,
    boolq_split: str = "validation",
    winogrande_version: str = "winogrande_xl",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load small subsets of BoolQ and Winogrande as plain Python lists.
    Backward compatible API.
    """
    boolq_data = load_boolq(max_examples=max_examples, split=boolq_split)
    winogrande_data = load_winogrande(max_examples=max_examples, version=winogrande_version)
    return boolq_data, winogrande_data


def load_small_pubmed(max_examples: int = 1000) -> List[Dict]:
    """
    Load small subset of PubMedQA as plain Python list.
    Backward compatible API.
    """
    return load_pubmedqa(max_examples=max_examples)


def eval_winogrande_boolq(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    max_examples: int = 1000,
    *,
    seq_len: Optional[int] = None,
    boolq_batch_size: Optional[int] = None,
    winogrande_max_len: Optional[int] = None,
    winogrande_batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper: evaluate both BoolQ and Winogrande.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        device: Device to run on
        max_examples: Maximum examples per dataset
        seq_len: Maximum sequence length for BoolQ
        boolq_batch_size: Batch size for BoolQ
        winogrande_max_len: Maximum sequence length for Winogrande
        winogrande_batch_size: Batch size for Winogrande
    
    Returns:
        {"boolq_acc": float, "winogrande_acc": float}
    """
    if seq_len is None:
        seq_len = 8192
    if boolq_batch_size is None:
        boolq_batch_size = 8
    if winogrande_max_len is None:
        winogrande_max_len = min(512, seq_len)
    if winogrande_batch_size is None:
        winogrande_batch_size = boolq_batch_size
    
    boolq_data, winogrande_data = load_small_boolq_winogrande(max_examples=max_examples)
    
    print("Evaluating BoolQ...")
    boolq_acc = eval_boolq(
        model, tokenizer, boolq_data,
        device=device,
        batch_size=boolq_batch_size,
        max_input_len=seq_len,
    )
    
    print("Evaluating Winogrande...")
    winogrande_acc = eval_winogrande(
        model, tokenizer, winogrande_data,
        device=device,
        batch_size=winogrande_batch_size,
        max_input_len=winogrande_max_len,
    )
    
    return {
        "boolq_acc": boolq_acc,
        "winogrande_acc": winogrande_acc,
    }


def eval_all_benchmarks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 4,
    max_length: int = 512,
    max_examples: Optional[int] = None,
    datasets: Optional[List[str]] = None,
    include_generative: bool = False,
) -> Dict[str, float]:
    """
    Evaluate on all (or specified) benchmarks.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        device: Device to run on
        batch_size: Batch size for evaluation (automatically clamped to safe values)
        max_length: Maximum sequence length
        max_examples: Maximum examples per dataset
        datasets: List of dataset names to evaluate (default: all multiple-choice)
        include_generative: Whether to include generative tasks (DROP, SQuAD v2)
    
    Returns:
        Dictionary mapping "{dataset}_acc" (or _em/_f1) to scores
    """
    # Multiple-choice datasets
    mc_datasets = ["boolq", "winogrande", "pubmedqa", "mmlu", "hellaswag", 
                   "arc", "piqa", "aqua_rat", "casehold", "commonsenseqa"]
    # Generative datasets
    gen_datasets = ["drop", "squadv2"]
    
    if datasets is None:
        datasets = mc_datasets.copy()
        if include_generative:
            datasets.extend(gen_datasets)
    
    mc_evaluator = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    
    gen_evaluator = GenerativeEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_input_len=max_length,
    ) if any(d in gen_datasets for d in datasets) else None
    
    results = {}
    for dataset in datasets:
        print(f"Evaluating {dataset}...")
        try:
            if dataset in gen_datasets:
                result = gen_evaluator.evaluate(dataset, max_examples=max_examples)
                results[f"{dataset}_em"] = result.exact_match
                results[f"{dataset}_f1"] = result.f1
                results[f"{dataset}_total"] = result.total
            else:
                result = mc_evaluator.evaluate(dataset, max_examples=max_examples)
                results[f"{dataset}_acc"] = result.accuracy
                results[f"{dataset}_total"] = result.total
        except Exception as e:
            print(f"Error evaluating {dataset}: {e}")
            results[f"{dataset}_error"] = str(e)
    
    return results


# =============================================================================
# Quick evaluation functions (for training loops)
# =============================================================================

def quick_eval_boolq(
    model,
    tokenizer,
    limit: int = 200,
    batch_size: int = 8,
    max_gen_toks: int = 32,  # Unused, kept for API compatibility
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Quick BoolQ evaluation during training.
    Uses the unified batched evaluator.
    """
    if device is None:
        device = next(model.parameters()).device
    
    start_time = time.time()
    acc = eval_boolq(
        model, tokenizer,
        device=device,
        batch_size=batch_size,
        max_examples=limit,
    )
    eval_time = time.time() - start_time
    
    return {
        "boolq_acc": acc,
        "eval_time_sec": eval_time,
        "questions_evaluated": limit,
    }


def quick_eval_multiple_tasks(
    model,
    tokenizer,
    tasks_to_eval: List[str] = ["boolq", "winogrande"],
    limit: int = 100,
    batch_size: int = 8,
    max_gen_toks: int = 32,  # Unused, kept for API compatibility
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quick evaluation on multiple tasks during training.
    Uses the unified batched evaluator.
    """
    if device is None:
        device = next(model.parameters()).device
    
    evaluator_obj = MultipleChoiceEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=512,
    )
    
    all_metrics = {}
    total_time = 0
    
    for task_name in tasks_to_eval:
        try:
            start_time = time.time()
            result = evaluator_obj.evaluate(task_name, max_examples=limit)
            eval_time = time.time() - start_time
            
            all_metrics[f"{task_name}_acc"] = result.accuracy
            all_metrics[f"{task_name}_eval_time"] = eval_time
            total_time += eval_time
            
        except Exception as e:
            all_metrics[f"{task_name}_error"] = str(e)
    
    all_metrics["total_eval_time"] = total_time
    all_metrics["questions_per_task"] = limit
    
    return all_metrics


# =============================================================================
# Logging utilities
# =============================================================================

def log_eval_metrics(metrics: Dict[str, Any], step: int, logger=None, wandb=None):
    """
    Log evaluation metrics to logger and wandb.
    """
    if "error" in metrics:
        if logger:
            logger.warning(f"Evaluation error at step {step}: {metrics['error']}")
        return
    
    if logger:
        log_str = f"Step {step} eval metrics: "
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f"{key}={value:.3f} "
        logger.info(log_str)
    
    if wandb and hasattr(wandb, 'log'):
        wandb_metrics = {
            f"eval/{key}": value 
            for key, value in metrics.items() 
            if isinstance(value, (int, float))
        }
        wandb_metrics["eval/step"] = step
        wandb.log(wandb_metrics)


def should_run_intermittent_eval(step: int, eval_frequency: int = 15, start_step: int = 50) -> bool:
    """
    Determine if we should run intermittent evaluation at this step.
    """
    return step >= start_step and step % eval_frequency == 0


# =============================================================================
# Subprocess-based evaluation (kept for compatibility but not recommended)
# =============================================================================

def quick_eval_boolq_subprocess(
    model,
    tokenizer,
    limit: int = 200,
    batch_size: int = 1,
    max_gen_toks: int = 32,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Quick BoolQ evaluation in subprocess to isolate from training.
    DEPRECATED: Use quick_eval_boolq() instead for better performance.
    """
    # Fall back to in-process evaluation
    return quick_eval_boolq(
        model=model,
        tokenizer=tokenizer,
        limit=limit,
        batch_size=batch_size,
        device=device,
    )


# =============================================================================
# Main / CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a model on multiple-choice benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--datasets", type=str, nargs="+", 
                        default=["boolq", "winogrande"],
                        help="Datasets to evaluate")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Maximum examples per dataset")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    from transformers import AutoModelForCausalLM
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Run evaluation
    results = eval_all_benchmarks(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_examples=args.max_examples,
        datasets=args.datasets,
    )
    
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")