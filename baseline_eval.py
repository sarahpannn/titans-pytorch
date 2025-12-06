#!/usr/bin/env python3
"""
Benchmark LLaMA 3.1 8B with segmented attention (no neural memory).

This script builds a TitanLLaMA model with segmented attention only, then runs
the LM Evaluation Harness over the following tasks:
- MMLU, MMLU_pro, AGIEval, CommonsenseQA, Winogrande, BBH, ARC-Challenge,
  Trivia-QA Wiki, SQuAD, QuAC, BoolQ, DROP.

Metrics are filtered to `acc_norm`/`acc_per_char`/`acc` or `f1` where applicable.

Usage (example):
    python baseline_eval.py --batch-size 1 --limit 25

Requires: pip install lm-eval==0.4.2 transformers datasets torch
"""

import argparse
import json
from typing import Iterable, List, Sequence, Tuple
import tqdm
from tqdm import tqdm

import torch
from transformers import AutoConfig, AutoTokenizer

from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM

try:
    from lm_eval import evaluator, tasks
    from lm_eval.api.model import LM
except ImportError as exc:  # pragma: no cover - dependency is optional at import time
    raise SystemExit(
        "lm-eval is required for benchmarking."
    ) from exc



DEFAULT_TASKS: Sequence[str] = (
    # "mmlu",
    # "mmlu_pro",
    # "agieval",
    # "commonsense_qa",
    "winogrande",
    # "bbh",
    # "arc_challenge",
    # "triviaqa_wiki",
    # "squadv2", # limit 100
    # "quac",
    # "boolq",
    # "drop", # limit 100
)


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device) if tensor.device != device else tensor


class TitanSegmentedLM(LM):
    """Minimal LM Eval wrapper around Titan segmented-attention LLaMA."""

    def __init__(
        self,
        model: TitanLLaMAForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int = 1,
        max_gen_toks: int = 128,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks
        self._device = next(model.parameters()).device
        # Get actual distributed environment if available
        import os
        self._rank = int(os.environ.get('RANK', 0))
        self._world_size = int(os.environ.get('WORLD_SIZE', 1))

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return getattr(self.model.config, "max_position_embeddings", 2048)

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _reset_memory_states(self):
        """Reset model memory states to prevent memory corruption."""
        try:
            if hasattr(self.model, 'reset_memory_states'):
                self.model.reset_memory_states()
            torch.cuda.empty_cache()
        except Exception:
            pass  # Graceful degradation if reset fails

    # --- LM Eval required methods -------------------------------------------------
    def loglikelihood(self, requests: Iterable) -> List[Tuple[float, bool]]:
        self._reset_memory_states()  # Reset memory before evaluation
        outputs = []
        for request in tqdm(requests):
            context, continuation = request.args if hasattr(request, "args") else request
            ll, greedy = self._loglikelihood_single(context, continuation)
            outputs.append((ll, greedy))
        return outputs

    def loglikelihood_rolling(self, requests: Iterable) -> List[Tuple[float, bool]]:
        self._reset_memory_states()  # Reset memory before evaluation
        results: List[Tuple[float, bool]] = []
        stride = self.max_length - 1
        for request in requests:
            (text,) = request.args if hasattr(request, "args") else request
            tokens = self.tok_encode(text)
            if not tokens:
                results.append((0.0, True))
                continue

            total_logprob = 0.0
            idx = 0
            while idx < len(tokens):
                window = tokens[max(0, idx - stride) : idx + 1]
                input_ids = torch.tensor([window], device=self.device)
                attention_mask = torch.ones_like(input_ids, device=self.device)
                with torch.no_grad():
                    logits = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )["logits"]
                logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                target_tokens = torch.tensor(window[1:], device=self.device)
                token_lp = logprobs[0, -len(target_tokens) :, :].gather(
                    -1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                total_logprob += token_lp.sum().item()
                idx += stride

            results.append((float(total_logprob), False))
        return results

    def generate_until(self, requests: Iterable) -> List[str]:
        self._reset_memory_states()  # Reset memory before evaluation
        generations: List[str] = []
        for request in tqdm(requests):
            context, gen_args = request.args if hasattr(request, "args") else request
            until = gen_args.get("until", []) if isinstance(gen_args, dict) else []
            max_new_tokens = gen_args.get("max_gen_toks", self.max_gen_toks) if isinstance(
                gen_args, dict
            ) else self.max_gen_toks

            inputs = self.tokenizer(
                context, return_tensors="pt", add_special_tokens=False
            )
            inputs = {k: _to_device(v, self.device) for k, v in inputs.items()}
            gen_text = self._greedy_generate(inputs, max_new_tokens=max_new_tokens, stop_tokens=until)
            generations.append(gen_text)
        return generations

    def _greedy_generate(self, inputs: dict, max_new_tokens: int, stop_tokens: Sequence[str]) -> str:
        """Lightweight greedy decode since TitanLLaMAForCausalLM lacks HF .generate()."""
        # Reset memory states before generation
        try:
            if hasattr(self.model, 'reset_memory_states'):
                self.model.reset_memory_states()
        except Exception:
            pass
            
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids, device=self.device))
        generated = input_ids

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.model(input_ids=generated, attention_mask=attention_mask)
                logits = out["logits"][:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.ones_like(generated, device=self.device)

            # Check stop tokens in decoded text
            text = self.tok_decode(generated[0].tolist()[input_ids.shape[1] :])
            for stopper in stop_tokens:
                stop_idx = text.find(stopper)
                if stop_idx != -1:
                    return text[:stop_idx]

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tok_decode(generated[0].tolist()[input_ids.shape[1] :])

    # --- Helpers ------------------------------------------------------------------
    def _loglikelihood_single(self, context: str, continuation: str) -> Tuple[float, bool]:
        # Reset memory states for each request to prevent corruption
        try:
            if hasattr(self.model, 'reset_memory_states'):
                self.model.reset_memory_states()
        except Exception:
            pass
            
        context_ids = self.tok_encode(context)
        continuation_ids = self.tok_encode(continuation)
        if not context_ids:
            bos = self.tokenizer.bos_token_id
            context_ids = [bos if bos is not None else self.eot_token_id]

        input_ids = torch.tensor(
            [context_ids + continuation_ids],
            device=self.device,
        )
        attention_mask = torch.ones_like(input_ids, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)

        cont_start = len(context_ids)
        cont_end = cont_start + len(continuation_ids)
        target_slice = logprobs[0, cont_start - 1 : cont_end - 1, :]
        target_tokens = torch.tensor(continuation_ids, device=self.device)
        token_logprobs = target_slice.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)

        greedy = bool(
            torch.all(
                target_tokens
                == torch.argmax(target_slice, dim=-1)
            ).item()
        )
        return float(token_logprobs.sum().item()), greedy


def build_segmented_llama(
    base_model_name: str,
    segment_len: int = 512,
    num_persist_mem_tokens: int = 4,
    num_longterm_mem_tokens: int = 4,
    use_flex_attn: bool = False,
    sliding_window_attn: bool = False,
    dtype: str = "bfloat16",
) -> Tuple[TitanLLaMAForCausalLM, AutoTokenizer]:
    """Load Meta-LLaMA-3.1-8B weights into Titan segmented attention without NMM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    base_cfg = AutoConfig.from_pretrained(base_model_name)
    titan_cfg = TitanLLaMAConfig.from_llama_config(
        base_cfg,
        segment_len=segment_len,
        num_persist_mem_tokens=num_persist_mem_tokens,
        num_longterm_mem_tokens=num_longterm_mem_tokens,
        neural_memory_layers=(),  # disable neural memory module
        neural_memory_segment_len=0,
        neural_memory_batch_size=0,
        neural_memory_depth=0,
        use_flex_attn=use_flex_attn,
        sliding_window_attn=sliding_window_attn,
        neural_mem_gate_attn_output=False,
        neural_mem_weight_residual=False,
        neural_mem_qkv_receives_diff_view=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = TitanLLaMAForCausalLM.from_pretrained_llama(
        base_model_name_or_path=base_model_name,
        titan_config=titan_cfg,
        freeze_backbone=True,
        dtype=torch_dtype,
        device_map="cuda",
    )
    model.to(device)
    return model, tokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LLaMA 3.1 8B with segmented attention (no NMM).")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B", help="Base HF model to load.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="LM Eval task names to run.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--max-gen-toks", type=int, default=128, help="Max new tokens for generation tasks.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task example limit for a quick smoke test.")
    parser.add_argument("--segment-len", type=int, default=512, help="Segment/window length for segmented attention.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16", help="Model dtype.")
    parser.add_argument("--eval_out", type=str, default="baseline_eval.txt", help="Where eval summary saves to.")
    parser.add_argument("--bootstrap-iters", type=int, default=0, help="Confidence-interval bootstrap iterations (lower to finish faster).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = build_segmented_llama(
        base_model_name=args.model,
        segment_len=args.segment_len,
        dtype=args.dtype,
    )
    lm = TitanSegmentedLM(model=model, tokenizer=tokenizer, batch_size=args.batch_size, max_gen_toks=args.max_gen_toks)

    task_dict = tasks.get_task_dict(args.tasks)
    results = evaluator.evaluate(
        lm=lm, task_dict=task_dict, 
        limit=args.limit, 
        bootstrap_iters=args.bootstrap_iters,
    )

    with open(args.eval_out, "w") as f:
        f.write(str(results))
    
    # print("\n=== Full lm-eval output ===")
    # print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
